import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms
from utils.tensor import TensorDict
import utils.data_processing as prutils
import torch.nn.functional as F
from datetime import datetime
from PIL import Image, ImageDraw


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


class STARKProcessingThree(BaseProcessing):
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
                print("Too small box is found. Replace it with new data.")
                return data
            # save_frame希望保留原始图像，用于后续接着裁剪更多的模板
            if s == 'template':
                save_frame = data['template_images']
                save_anno = data['template_anno']
                save_jitter = jittered_anno
                save_masks = data['template_masks']
                
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[factor_key],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # if s == 'template':
            #     validator = AugmentationValidator()
            #     validator.save_sample(save_frame[0], crops[0], boxes[0], att_mask[0], mask_crops[0])

            # Apply transforms
            # 在这里会产生增强变换以及array到tensor的转变。
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[factor_key](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    print("Values of down-sampled attention mask are all one. "
                          "Replace it with new data.")
                    return data
#######
        jittered_anno = [self._get_jittered_box(a, 'template') for a in data['template_anno']]
        w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
        crop_sz2 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*1.3)
        crop_sz3 = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template']*0.8)
        if (crop_sz2 < 1 or crop_sz3 < 1).any():
            data['valid'] = False
            return data
        # print('22222', data['template_images'][0].shape, len(data['template_images']))
        crops2, boxes2, att_mask2, mask_crops2 = prutils.jittered_center_crop(save_frame, save_jitter,
                                                                              save_anno, self.search_area_factor['template']*1.3,
                                                                              self.output_sz['template'], masks=save_masks)
        # print('Save_frame:', type(save_frame[0]), save_frame[0].shape)

        crops2, boxes2, att_mask2, mask_crops2 = prutils.rotate_all(crops2, boxes2, att_mask2, mask_crops2, 
                                                                    self.output_sz['template'], self.output_sz['template'], k=3)

        crops3, boxes3, att_mask3, mask_crops3 = prutils.jittered_center_crop(save_frame, save_jitter,
                                                                              save_anno, self.search_area_factor['template']*0.8,
                                                                              self.output_sz['template'], masks=save_masks)
        crops3, boxes3, att_mask3, mask_crops3 = prutils.rotate_all(crops3, boxes3, att_mask3, mask_crops3, self.output_sz['template'], 
                                                                    self.output_sz['template'], k=1)
        
        # 为 crops2 添加随机光照噪声
        crops2 = [prutils.add_light_gradient(crop, direction='horizontal', intensity=10) for crop in crops2]
        # 为 crops3 添加随机光照噪声
        crops3 = [prutils.add_light_gradient(crop, direction='vertical', intensity=15) for crop in crops3]

        keys = ['template_images', 'template_anno', 'template_att', 'template_masks']
        new_data2 = self.transform['template'](image=crops2, bbox=boxes2, att=att_mask2, mask=mask_crops2, joint=False)
        new_data3 = self.transform['template'](image=crops3, bbox=boxes3, att=att_mask3, mask=mask_crops3, joint=False)
        
        for k ,ls in zip(keys, new_data2):
            data[k] += ls
        for k ,ls in zip(keys, new_data3):
            data[k] += ls     

        # save_dir = '/data/yakun/test_augment/123'
        # Image.fromarray(crops2[0]).save("debug_crops2.png")
        # Image.fromarray(crops3[0]).save("debug_crops3.png")
        # self.save_visual(crops2, 2, boxes2, save_dir, prefix="visual")
        # self.save_visual(crops3, 3, boxes3, save_dir, prefix="visual")

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

    def __call__000(self, data: TensorDict):
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


        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
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
    


# 为了检查模板是否正确定义的可视化函数
    def save_visual(self, crops, imageid, boxes, save_dir, prefix="visual"):
        """
        保存裁剪结果和对应的bbox到指定路径。

        参数:
            crops (list of np.ndarray): 裁剪后的图像列表。
            boxes (list of torch.Tensor): 对应的bbox列表，格式为 [x1, y1, w, h]，归一化到 [0, 1]。
            save_dir (str): 保存图像的目录。
            prefix (str): 保存文件的前缀。
        """
        os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
        for i, (crop, box) in enumerate(zip(crops, boxes)):
            # 将裁剪后的图像转换为PIL格式
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


