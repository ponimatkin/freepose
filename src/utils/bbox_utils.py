import logging
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class CropResizePad:
    def __init__(self, target_size: Union[Tuple, int], orig_size: Union[Tuple, int], bbox_extend: int = 0) -> torch.Tensor:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)
        self.bbox_extend = bbox_extend
        self.h, self.w = orig_size

    def __call__(self, images: torch.Tensor, boxes: torch.Tensor):
        boxes = boxes.clone()
        for box in boxes:
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            box[0] = max(0, box[0] - self.bbox_extend * box_w)
            box[2] = min(self.w, box[2] + self.bbox_extend * box_w)
            box[1] = max(0, box[1] - self.bbox_extend * box_h)
            box[3] = min(self.h, box[3] + self.bbox_extend * box_h)
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            image = image[:, box[1] : box[3], box[0] : box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom)
                )
            assert image.shape[1] == image.shape[2], logging.info(
                f"image {image.shape} is not square after padding"
            )
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


def xyxy_to_xywh(bbox):
    if len(bbox.shape) == 1:
        """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    elif len(bbox.shape) == 2:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    if len(bbox.shape) == 1:
        x, y, w, h = bbox
        return [x, y, x + w - 1, y + h - 1]
    elif len(bbox.shape) == 2:
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def get_bbox_size(bbox):
    return [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def make_bbox_dividable(bbox_size, dividable_size, ceil=True):
    if ceil:
        new_size = np.ceil(np.array(bbox_size) / dividable_size) * dividable_size
    else:
        new_size = np.floor(np.array(bbox_size) / dividable_size) * dividable_size
    return new_size


def make_bbox_square(old_bbox):
    size_to_fit = np.max([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]])
    new_bbox = np.array(old_bbox)
    old_bbox_size = [old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]
    # Add padding into y axis
    displacement = int((size_to_fit - old_bbox_size[1]) / 2)
    new_bbox[1] = old_bbox[1] - displacement
    new_bbox[3] = old_bbox[3] + displacement
    # Add padding into x axis
    displacement = int((size_to_fit - old_bbox_size[0]) / 2)
    new_bbox[0] = old_bbox[0] - displacement
    new_bbox[2] = old_bbox[2] + displacement
    return new_bbox


def crop_image(image, bbox, format="xyxy"):
    if format == "xyxy":
        image_cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    elif format == "xywh":
        image_cropped = image[
            bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
        ]
    return image_cropped


def force_binary_mask(mask, threshold=0.):
    mask = np.where(mask > threshold, 1, 0)
    return mask


def bbox_iou(bb_a, bb_b):
    # [x1, y1, width, height] --> [x1, y1, x2, y2]
    tl_a, br_a = (bb_a[0], bb_a[1]), (bb_a[0] + bb_a[2], bb_a[1] + bb_a[3])
    tl_b, br_b = (bb_b[0], bb_b[1]), (bb_b[0] + bb_b[2], bb_b[1] + bb_b[3])
    
    # Intersection rectangle.
    tl_inter = max(tl_a[0], tl_b[0]), max(tl_a[1], tl_b[1])
    br_inter = min(br_a[0], br_b[0]), min(br_a[1], br_b[1])

    # Width and height of the intersection rectangle.
    w_inter = br_inter[0] - tl_inter[0]
    h_inter = br_inter[1] - tl_inter[1]

    if w_inter > 0 and h_inter > 0:
        area_inter = w_inter * h_inter
        area_a = bb_a[2] * bb_a[3]
        area_b = bb_b[2] * bb_b[3]
        iou = area_inter / float(area_a + area_b - area_inter)
    else:
        iou = 0.0

    return iou