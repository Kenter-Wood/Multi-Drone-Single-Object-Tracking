import os
import cv2
import time
from PIL import Image, ImageDraw
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.tensor import TensorDict
import utils.data_processing as prutils
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        # mode = 'template' if mode == 'template_eva' else 'search' if mode == 'search_eva' else mode
        if mode == 'template_eva':
            mode = 'template'
        elif mode == 'search_eva':
            mode = 'search'
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        # data['template_eva_images'] = data['template_images']
        # data['template_eva_anno'] = data['template_anno']
        # data['template_eva_masks'] = data['template_masks']
        # data['search_eva_images'] = data['search_images']
        # data['search_eva_anno'] = data['search_anno']
        # data['search_eva_masks'] = data['search_masks']

        for s in ['template', 'search']:
        # for s in ['template', 'search', 'template_eva', 'search_eva']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            if s in ['template_eva', 'search_eva']:
                factor_key = 'template' if s == 'template_eva' else 'search'
            else:
                factor_key = s

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[factor_key])

            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[factor_key],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[factor_key](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data


        # save_dir = '/data/yakun/MultiTrack/output/debug/multitrain'
        # save_visual(data['template_images'], 1, data['template_anno'], save_dir, prefix="template")
        # save_visual(data['search_images'], 1, data['search_anno'], save_dir, prefix="search")
        # time.sleep(10)





        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
    
class STARKProcessingThree(BaseProcessing):
    """ 
        Changed this processing function to obtain three templates for searching
        Used when training Multi-Drone methods with LaSOT/GOT...
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        # mode = 'template' if mode == 'template_eva' else 'search' if mode == 'search_eva' else mode
        if mode == 'template_eva':
            mode = 'template'
        elif mode == 'search_eva':
            mode = 'search'
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        # data['template_eva_images'] = data['template_images']
        # data['template_eva_anno'] = data['template_anno']
        # data['template_eva_masks'] = data['template_masks']
        # data['search_eva_images'] = data['search_images']
        # data['search_eva_anno'] = data['search_anno']
        # data['search_eva_masks'] = data['search_masks']

        for s in ['template', 'search']:
        # for s in ['template', 'search', 'template_eva', 'search_eva']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            if s in ['template_eva', 'search_eva']:
                factor_key = 'template' if s == 'template_eva' else 'search'
            else:
                factor_key = s

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[factor_key])

            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            # save_frame希望保留原始图像，用于后续接着裁剪更多的模板
            if s == 'template':
                save_frame = data['template_images']
                save_anno = data['template_anno']
                save_masks = data['template_masks']
                
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[factor_key],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # if s == 'template':
            #     validator = AugmentationValidator()
            #     validator.save_sample(save_frame[0], crops[0], boxes[0], att_mask[0], mask_crops[0])
            if s == 'template':
                save_frame = data['template_images']
                save_anno = data['template_anno']
                save_masks = data['template_masks']
                save_jitter = jittered_anno
                save_crops_temp = crops
                anno_temp = boxes
            else:
                save_crops_search = crops
                anno_search = boxes


            # Apply transforms
            # 在这里会产生增强变换以及array到tensor的转变。
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[factor_key](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

#######第二种透视变换的增强方法
    # 先对原图进行一定的透视变换，得到新的图像和标签
        H, W, C = save_frame[0].shape
        # absolute_bboxes按照x1y1,x2y2表示
        absolute_bboxes = [
            BoundingBox(x1=anno[0], y1=anno[1], 
                x2=anno[0] + anno[2], y2=anno[1] + anno[3])
            for anno in save_anno
        ]
        abs_boxes_np = []
        for bb in absolute_bboxes:
            abs_boxes_np.append([bb.x1, bb.y1, bb.x2-bb.x1, bb.y2-bb.y1])

        # print(abs_boxes_np)
        # 应用第一次透视变换
        aug_img1, aug_boxes1, aug_masks1 = apply_perspective_transform(
            save_frame, abs_boxes_np, [mask for mask in save_masks], scale=(0.1, 0.2), direction='horizontal')

        # 应用第二次透视变换
        aug_img2, aug_boxes2, aug_masks2 = apply_perspective_transform(
            save_frame, abs_boxes_np, [mask for mask in save_masks], scale=(-0.2, -0.1), direction='horizontal')

        aug_anno1 = []
        aug_anno2 = []
        for box in aug_boxes1:
            x, y, w, h = box
            aug_anno1.append(torch.tensor([x, y, w, h], dtype=torch.float32))
        for box in aug_boxes2:
            x, y, w, h = box
            aug_anno2.append(torch.tensor([x, y, w, h], dtype=torch.float32))
        # print('augaug', aug_anno1)
    # 然后在增强中，用新的图像代替之前的

        jittered_anno1 = [self._get_jittered_box(a, 'template') for a in aug_anno1]
        w, h = torch.stack(jittered_anno1, dim=0)[:, 2], torch.stack(jittered_anno1, dim=0)[:, 3]
        crop_sz2 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*1.15)

        jittered_anno2 = [self._get_jittered_box(a, 'template') for a in aug_anno2]
        w, h = torch.stack(jittered_anno2, dim=0)[:, 2], torch.stack(jittered_anno2, dim=0)[:, 3]        
        crop_sz3 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*0.85)
        if (crop_sz2 < 1 or crop_sz3 < 1).any():
            data['valid'] = False
            return data
        crops2, boxes2, att_mask2, mask_crops2 = prutils.jittered_center_crop(aug_img1, jittered_anno1,
                                                                              aug_anno1, self.search_area_factor['template']*1.15,
                                                                              self.output_sz['template'], masks=aug_masks1)
        crops3, boxes3, att_mask3, mask_crops3 = prutils.jittered_center_crop(aug_img2, jittered_anno2,
                                                                              aug_anno2, self.search_area_factor['template']*0.85,
                                                                              self.output_sz['template'], masks=aug_masks2)
       
        keys = ['template_images', 'template_anno', 'template_att', 'template_masks']
        new_data2 = self.transform['template'](image=crops2, bbox=boxes2, att=att_mask2, mask=mask_crops2, joint=False)
        new_data3 = self.transform['template'](image=crops3, bbox=boxes3, att=att_mask3, mask=mask_crops3, joint=False)
        
        for k ,ls in zip(keys, new_data2):
            data[k] += ls
        for k ,ls in zip(keys, new_data3):
            data[k] += ls    


#######第一种随即旋转的增强方法
        # jittered_anno = [self._get_jittered_box(a, 'template') for a in data['template_anno']]
        # w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
        # crop_sz2 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*1.3)
        # crop_sz3 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*0.8)
        # if (crop_sz2 < 1 or crop_sz3 < 1).any():
        #     data['valid'] = False
        #     return data
        # # print('22222', data['template_images'][0].shape, len(data['template_images']))
        # crops2, boxes2, att_mask2, mask_crops2 = prutils.jittered_center_crop(save_frame, save_jitter,
        #                                                                       save_anno, self.search_area_factor['template']*1.3,
        #                                                                       self.output_sz['template'], masks=save_masks)
        # # print('Save_frame:', type(save_frame[0]), save_frame[0].shape)

        # crops2, boxes2, att_mask2, mask_crops2 = prutils.rotate_all(crops2, boxes2, att_mask2, mask_crops2, 
        #                                                             self.output_sz['template'], self.output_sz['template'], k=3)

        # crops3, boxes3, att_mask3, mask_crops3 = prutils.jittered_center_crop(save_frame, save_jitter,
        #                                                                       save_anno, self.search_area_factor['template']*0.8,
        #                                                                       self.output_sz['template'], masks=save_masks)
        # crops3, boxes3, att_mask3, mask_crops3 = prutils.rotate_all(crops3, boxes3, att_mask3, mask_crops3, self.output_sz['template'], 
        #                                                             self.output_sz['template'], k=1)
        
        # # 为 crops2 添加随机光照噪声
        # crops2 = [prutils.add_light_gradient(crop, direction='horizontal', intensity=10) for crop in crops2]
        # # 为 crops3 添加随机光照噪声
        # crops3 = [prutils.add_light_gradient(crop, direction='vertical', intensity=15) for crop in crops3]

        # keys = ['template_images', 'template_anno', 'template_att', 'template_masks']
        # new_data2 = self.transform['template'](image=crops2, bbox=boxes2, att=att_mask2, mask=mask_crops2, joint=False)
        # new_data3 = self.transform['template'](image=crops3, bbox=boxes3, att=att_mask3, mask=mask_crops3, joint=False)
        
        # for k ,ls in zip(keys, new_data2):
        #     data[k] += ls
        # for k ,ls in zip(keys, new_data3):
        #     data[k] += ls     
####### 保存图片可视化
        # save_dir = '/data/yakun/MultiTrack/output/debug/augment'
        # self.save_visual(save_crops_search, 0, anno_search, save_dir, prefix="visual")
        # self.save_visual(save_crops_temp, 1, anno_temp, save_dir, prefix="visual")
        # self.save_visual(crops2, 2, boxes2, save_dir, prefix="visual")
        # self.save_visual(crops3, 3, boxes3, save_dir, prefix="visual")
        # print(save_frame[0].shape)
        # time.sleep(10)
####### 

        data['valid'] = True
        # print('Yes, data is valid')
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
    


    # 为了检查模板是否正确定义的可视化函数



class SimpleMask:
    """简化的掩码类，用于替代 MasksOnImage"""
    def __init__(self, masks, shape):
        self.masks = [mask.astype(np.uint8) if isinstance(mask, np.ndarray) else mask for mask in masks]
        self.shape = shape
        
    def draw_on_image(self, image):
        result = image.copy()
        for mask in self.masks:
            mask_region = mask > 0
            result[mask_region] = result[mask_region] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
        return result

def apply_perspective_transform(images, boxes, masks=None, scale=(0.01, 0.1), direction=None):
    """
    应用透视变换到图像、边界框和掩码
    
    参数:
        images: 图像列表
        boxes: 边界框列表，格式为 [[x1,y1,w,h], ...]
        masks: 掩码列表（可选）
        scale: 透视变换的强度范围
        direction: 变换的主要方向，可以是 'horizontal', 'vertical', 'diagonal_up', 'diagonal_down' 或 None（随机）
    
    返回:
        transformed_images, transformed_boxes, transformed_masks
    """
    results_img = []
    results_boxes = []
    results_masks = []
    
    for i, (img, box) in enumerate(zip(images, boxes)):
        try:
            # 检查图像是否为张量，如果是则转换为numpy数组
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()                
            h, w = img.shape[:2]
            
            # 生成透视变换矩阵，确保点格式正确
            scale_value = np.random.uniform(scale[0], scale[1])
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # 根据指定方向生成扰动
            displacement_direction = 1 if scale_value > 0 else -1
            max_displacement = abs(scale_value) * min(w, h)
            if direction == 'horizontal':
                # 水平方向扰动较大
                displacement = np.zeros((4, 2))
                displacement[:, 0] = np.random.normal(0, max_displacement, 4)* displacement_direction  # x方向扰动大
                displacement[:, 1] = np.random.normal(0, max_displacement * 0.3, 4)* displacement_direction  # y方向扰动小
            elif direction == 'vertical':
                # 垂直方向扰动较大
                displacement = np.zeros((4, 2))
                displacement[:, 0] = np.random.normal(0, max_displacement * 0.3, 4)* displacement_direction  # x方向扰动小
                displacement[:, 1] = np.random.normal(0, max_displacement, 4)* displacement_direction  # y方向扰动大
            elif direction == 'diagonal_up':
                # 右上-左下对角线扰动
                displacement = np.random.normal(0, max_displacement, (4, 2))
                # 右上和左下点沿对角线方向移动
                displacement[1, :] *= np.array([1, -1])  # 右上角点
                displacement[2, :] *= np.array([1, -1])  # 左下角点
            elif direction == 'diagonal_down':
                # 左上-右下对角线扰动
                displacement = np.random.normal(0, max_displacement, (4, 2))
                # 左上和右下点沿对角线方向移动
                displacement[0, :] *= np.array([1, 1])  # 左上角点
                displacement[3, :] *= np.array([1, 1])  # 右下角点
            else:
                # 随机方向
                displacement = np.clip(
                    np.random.normal(0, max_displacement, (4, 2)), 
                    -max_displacement * 2, 
                    max_displacement * 2
                )
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) + displacement
            
            # 确保pts2中的点不会超出图像边界
            pts2[:, 0] = np.clip(pts2[:, 0], 0, w-1)
            pts2[:, 1] = np.clip(pts2[:, 1], 0, h-1)
            
            # 其余代码保持不变...
            if pts1.shape != (4, 2) or pts2.shape != (4, 2):
                print(f"点集形状错误: pts1.shape={pts1.shape}, pts2.shape={pts2.shape}")
                raise ValueError("透视变换需要4个2D点")

            # 检查是否有无效值
            if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
                print("点集包含NaN或Inf值")
                raise ValueError("透视变换点集包含无效值")           
            pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
            pts2 = np.ascontiguousarray(pts2, dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            
            # 应用到图像
            transformed_img = cv2.warpPerspective(img, M, (w, h))
            results_img.append(transformed_img)
            

            # 应用到边界框
            x, y, box_w, box_h = box
            box_pts = np.float32([[x, y], [x+box_w, y], [x, y+box_h], [x+box_w, y+box_h]])
            transformed_pts = cv2.perspectiveTransform(box_pts.reshape(1, -1, 2), M)[0]


            # 计算新边界框
            min_x = max(0, np.min(transformed_pts[:, 0]))
            min_y = max(0, np.min(transformed_pts[:, 1]))
            max_x = min(w, np.max(transformed_pts[:, 0]))
            max_y = min(h, np.max(transformed_pts[:, 1]))
            
            new_box = [min_x, min_y, max_x - min_x, max_y - min_y]
            results_boxes.append(new_box)
            
            # 应用到掩码
            if masks is not None and i < len(masks):
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                transformed_mask = cv2.warpPerspective(mask.astype(np.uint8), M, (w, h))
                results_masks.append(transformed_mask)
                
        except Exception as e:
            print(f"透视变换过程中出错: {e}")
            # 如果发生错误，使用原始图像和边界框
            results_img.append(img.copy())
            results_boxes.append(box)
            if masks is not None and i < len(masks):
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                results_masks.append(mask)
    
    return results_img, results_boxes, results_masks if masks is not None else None

def save_visual(crops, imageid, boxes, save_dir, prefix="visual"):
    """
    保存裁剪结果和对应的bbox到指定路径。
    """
    print(len(crops), crops[0].shape)
    print(len(boxes), boxes[0].shape)
    os.makedirs(save_dir, exist_ok=True)
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        # 先处理图像数据
        if isinstance(crop, torch.Tensor):
            crop = crop.detach().cpu().numpy()
        
        if crop.shape[0] == 3 and len(crop.shape) == 3:
            # 从(C,H,W)转为(H,W,C)
            crop = np.transpose(crop, (1, 2, 0))
        print(f"形状={crop.shape}, 类型={crop.dtype}, 范围=({np.min(crop)}-{np.max(crop)})")

        # 确保是三通道RGB图像
        if len(crop.shape) == 2:  # 单通道灰度图
            crop = np.stack([crop, crop, crop], axis=2)
        elif crop.shape[2] == 4:  # RGBA图像
            crop = crop[:, :, :3]  # 只用RGB通道
            
        # 数值范围处理
        if crop.dtype == np.float32 or crop.dtype == np.float64:
            crop = ((crop - np.min(crop))/(np.max(crop)-np.min(crop))*255).astype(np.uint8)
        
        # 确保是uint8类型
        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)
            
        # 确保图像连续性，解决"被一条横线分割"问题
        if not crop.flags['C_CONTIGUOUS']:
            crop = np.ascontiguousarray(crop)
            
        # 现在才创建PIL图像
        try:
            image = Image.fromarray(crop)
            
            # 反归一化bbox到像素坐标
            x1, y1, w, h = box.tolist()
            x1, y1 = x1 * crop.shape[1], y1 * crop.shape[0]
            w, h = w * crop.shape[1], h * crop.shape[0]
            x2, y2 = x1 + w, y1 + h
            
            # 在图像上绘制bbox
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
            
            # 保存图像
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(save_dir, f"{prefix}_{i}_{timestamp}_{imageid}.png")
            image.save(save_path)
            print(f"Saved visualized crop to {save_path}")
        except Exception as e:
            print(f"图像保存失败: {e}, 形状={crop.shape}, 类型={crop.dtype}, 范围=({np.min(crop)}-{np.max(crop)})")