import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target_new(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    max_scale_relative_to_image = 1.1


    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # 2025.4.20 在这里添加一个限定扩展尺寸的逻辑，这样不会把图像缩放的太显著
    # 计算图像的最长边
    img_max_dim = max(im.shape[0], im.shape[1])
    
    # 计算原始扩展尺寸
    original_crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    
    # 限制裁剪尺寸不超过图像最长边乘以限制系数
    max_allowed_crop_sz = int(img_max_dim * max_scale_relative_to_image)
    crop_sz = min(original_crop_sz, max_allowed_crop_sz)
    
    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    # 计算裁剪区域的实际搜索因子（用于日志或调试）
    actual_search_factor = crop_sz / math.sqrt(w * h) if w*h > 0 else search_area_factor
    # Crop image

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
    # 检查mask_crop的类型，确保其为PyTorch张量
        if isinstance(mask_crop, np.ndarray):
            mask_crop = torch.from_numpy(mask_crop).float()
        elif isinstance(mask_crop, torch.Tensor) and mask_crop.dtype != torch.float:
            mask_crop = mask_crop.float()
        
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    
        if isinstance(mask_crop, np.ndarray):
            mask_crop = torch.from_numpy(mask_crop).float()
        elif isinstance(mask_crop, torch.Tensor) and mask_crop.dtype != torch.float:
            mask_crop = mask_crop.float()

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded



def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

def add_light_noise(image, intensity=30):
    """
    添加随机光照噪声。
    
    参数:
        image (np.ndarray): 输入图像，形状为 (H, W, C)，像素值范围为 [0, 255]。
        intensity (int): 噪声强度，值越大，光照变化越明显。
    
    返回:
        np.ndarray: 添加光照噪声后的图像。
    """
    # 生成与图像大小相同的随机噪声
    noise = np.random.randint(-intensity, intensity, image.shape, dtype=np.int16)
    # 添加噪声
    image = image.astype(np.int16) + noise
    # 限制像素值范围在 [0, 255]
    image = np.clip(image, 0, 255)
    # 转换回 uint8 类型
    return image.astype(np.uint8)

def add_light_gradient(image, direction='horizontal', intensity=100):
    """
    添加光照梯度。
    
    参数:
        image (np.ndarray): 输入图像，形状为 (H, W, C)，像素值范围为 [0, 255]。
        direction (str): 梯度方向，'horizontal' 或 'vertical'。
        intensity (int): 梯度强度，值越大，光照变化越明显。
    
    返回:
        np.ndarray: 添加光照梯度后的图像。
    """
    h, w, c = image.shape
    if direction == 'horizontal':
        gradient = np.linspace(0, intensity, w, dtype=np.float32).reshape(1, w, 1)
    elif direction == 'vertical':
        gradient = np.linspace(0, intensity, h, dtype=np.float32).reshape(h, 1, 1)
    else:
        raise ValueError("Invalid direction. Choose 'horizontal' or 'vertical'.")
    
    # 添加梯度到图像
    image = image.astype(np.float32) + gradient
    # 限制像素值范围在 [0, 255]
    image = np.clip(image, 0, 255)
    # 转换回 uint8 类型
    return image.astype(np.uint8)



def rotate_image_and_bbox(array, bbox, w_image, h_image, k=1):
    """
    旋转图像和bbox。

    参数:
        array (np.ndarray): 输入的图像数组，形状为 (h_image, w_image, 3)。
        bbox (torch.Tensor): 输入的bbox，格式为 [x1, y1, w, h]，归一化到 [0, 1]。
        w_image (int): 图像宽度。
        h_image (int): 图像高度。
        k (int): 顺时针旋转次数，每次90度。

    返回:
        rotated_array (np.ndarray): 旋转后的图像数组。
        rotated_bbox (torch.Tensor): 旋转后的bbox，格式为 [x1, y1, w, h]，归一化到 [0, 1]。
    """
    # 确保数组形状正确
    assert array.shape == (h_image, w_image, 3), f"数组形状应为 {h_image}x{w_image}x3"

    # 处理数据类型转换
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    # 在array中顺时针旋转 k * 90 度
    rotated_array = np.rot90(array, k=4 - k).copy()  # 顺时针旋转 k 次等价于逆时针旋转 (4 - k) 次

    # 调整bbox坐标
    x1, y1, w, h = bbox[:4]
    x1, y1 = x1 * w_image, y1 * h_image
    w, h = w * w_image, h * h_image
    x2, y2 = x1 + w, y1 + h

    # 根据旋转次数调整bbox
    if k == 1:  # 顺时针旋转90度
        new_x1 = y1
        new_y1 = w_image - x2
        new_x2 = y2
        new_y2 = w_image - x1
    elif k == 2:  # 顺时针旋转180度
        new_x1 = w_image - x2
        new_y1 = h_image - y2
        new_x2 = w_image - x1
        new_y2 = h_image - y1
    elif k == 3:  # 顺时针旋转270度
        new_x1 = h_image - y2
        new_y1 = x1
        new_x2 = h_image - y1
        new_y2 = x2
    else:  # 不旋转
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

    # 将旋转后的bbox转换回归一化格式
    rotated_bbox = torch.tensor([new_x1 / w_image, new_y1 / h_image, 
                                 (new_x2 - new_x1) / w_image, (new_y2 - new_y1) / h_image])

    return rotated_array, rotated_bbox

def rotate_all(images, bboxs, att_masks, mask_crops, w_image, h_image, k):

    rotated_images = []
    rotated_bboxes = []
    rotated_att_masks = []
    rotated_mask_crops = []

    for rotated_image, rotated_bbox, att_mask, mask_crop in zip(images, bboxs, att_masks, mask_crops):
        rotated_image, rotated_bbox = rotate_image_and_bbox(rotated_image, rotated_bbox, w_image, h_image, k)
        rotated_att_mask = np.rot90(att_mask, k=4 - k).copy()
        rotated_mask_crop = np.rot90(mask_crop, k=4 - k).copy()

        rotated_images.append(rotated_image)
        rotated_bboxes.append(rotated_bbox)
        rotated_att_masks.append(rotated_att_mask)
        rotated_mask_crops.append(rotated_mask_crop)
    return rotated_images, rotated_bboxes, rotated_att_masks, rotated_mask_crops

