import cv2
import math
import torch
import pickle
import open3d
import trimesh
import dataclasses
import torchvision
import numpy as np
import pyrender

from PIL import Image
from typing import Sequence
from torchvision import transforms
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
from functools import partial

@dataclasses.dataclass
class Mesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    color: torch.Tensor
    texture: torch.Tensor


class MaybeToTensor(transforms.ToTensor):

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


pil2torch = transforms.Compose([
    MaybeToTensor(),
    make_normalize_transform()
])


def overlay_pcd(img, vertices, T, K):
    OUT = (np.pad(np.array(vertices), ((0, 0), (0, 1)), constant_values=1.) @ T.T)[:, :3] @ K.T
    OUT = OUT[:, :2] / OUT[:, 2:]

    fc = lambda x: np.maximum(0, np.minimum(x, 518-1))
    contour2 = np.array(img)
    contour2[fc(np.int32(OUT)[:, 1]), fc(np.int32(OUT)[:, 0])] = np.array([255, 255, 255])
    return contour2


def show_feats(pil_image, output, mask=None):
    H, W, C = output.shape
    features_rgb_ = PCA(n_components=3).fit_transform(output.view(-1, C).numpy())
    features_rgb_ = features_rgb_ - features_rgb_.min(0, keepdims=True)
    features_rgb_ = features_rgb_ / features_rgb_.max(0, keepdims=True)

    features_pil_ = Image.fromarray(np.uint8(features_rgb_.reshape(H, W, 3) * 255.)).resize((W * 14, H * 14),
                                                                                            Image.Resampling.NEAREST)
    output_list = [np.array(pil_image), np.array(features_pil_)]

    if mask is not None:
        upscaled_mask = np.stack(
            [np.array(Image.fromarray(np.uint8(mask) * 255).resize((W * 14, H * 14), Image.Resampling.NEAREST))] * 3, 2)
        output_list.append(upscaled_mask)

        masked_feats_rgb_ = output.numpy().copy()
        masked_feats_rgb_[np.logical_not(mask)] = 0
        masked_feats_rgb_ = PCA(n_components=3).fit_transform(masked_feats_rgb_.reshape(-1, C))
        masked_feats_rgb_ = masked_feats_rgb_ - masked_feats_rgb_.min(0, keepdims=True)
        masked_feats_rgb_ = masked_feats_rgb_ / masked_feats_rgb_.max(0, keepdims=True)

        masked_features = np.uint8(masked_feats_rgb_.reshape(H, W, 3) * 255.)
        masked_features[np.logical_not(mask)] = 0
        output_list.append(
            np.array(Image.fromarray(masked_features).resize((W * 14, H * 14), Image.Resampling.NEAREST)))

    return Image.fromarray(np.concatenate(output_list, 1))


def crop_image(image, Ts, points, K, render_width, render_height, lamb=1.4):
    assert len(image.shape) == 3 and image.shape[0] in [1, 3, 4] and image.dtype == torch.float32
    assert Ts.shape[1:] == torch.Size([4, 4])
    assert points.shape[1:] == torch.Size([4])
    assert K.shape == torch.Size([3, 3])

    T = torch.matmul(
        torch.nn.functional.pad(K, (0, 1, 0, 0), value=0.).unsqueeze(0),
        Ts
    )
    # project object points to the image and take their bounding box
    points_transformed = torch.matmul(points.unsqueeze(0), T.permute(0, 2, 1))
    uv = points_transformed[:, :, :2] / torch.maximum(points_transformed[:, :, [2]], torch.tensor(0.01))
    bboxes = torch.cat([uv.min(dim=1).values, uv.max(dim=1).values], dim=1)

    # project object center to the image
    centers_transformed = torch.matmul(torch.mean(points, dim=0, keepdim=True).unsqueeze(0), T.permute(0, 2, 1)).squeeze(1)
    centers_uv = centers_transformed[:, :2] / torch.maximum(centers_transformed[:, [2]], torch.tensor(0.01))

    # compute distances from object center to the bounding box corners
    dists = torch.maximum((bboxes[:, [0, 1]] - centers_uv).abs_(), (bboxes[:, [2, 3]] - centers_uv).abs_())
    xdists, ydists = dists[:, 0], dists[:, 1]

    # compute image crop
    r = render_width / render_height
    # r = 2464/ 2056
    width = torch.max(xdists, ydists * r) * 2 * lamb
    height = torch.max(xdists / r, ydists) * 2 * lamb
    x1, y1 = centers_uv[:, 0] - width / 2, centers_uv[:, 1] - height / 2
    x2, y2 = centers_uv[:, 0] + width / 2, centers_uv[:, 1] + height / 2

    bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    bboxes_with_batchdim = torch.cat(
        [torch.zeros((len(bboxes), 1), device=bboxes.device), bboxes], 1).to(image.device)

    crops = torchvision.ops.roi_align(
        image.unsqueeze(0),
        bboxes_with_batchdim,
        output_size=(render_height, render_width),
        sampling_ratio=2
    )
    return crops, bboxes


def update_K_with_crop(K, bboxes, render_width, render_height):
    # Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    # ! Skew is not handled.
    assert K.shape == torch.Size([3, 3])
    assert bboxes.shape[1:] == torch.Size([4])
    new_K = K.unsqueeze(0).repeat(len(bboxes), 1, 1)

    crop_width = bboxes[:, 2] - bboxes[:, 0]
    crop_height = bboxes[:, 3] - bboxes[:, 1]
    crop_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    crop_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

    # Crop
    cx = K[0, 2] + (crop_width - 1) / 2 - crop_cx
    cy = K[1, 2] + (crop_height - 1) / 2 - crop_cy

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = render_width / crop_width
    scale_y = render_height / crop_height
    scaled_center_x = (render_width - 1) / 2
    scaled_center_y = (render_height - 1) / 2
    fx = scale_x * K[0, 0]
    fy = scale_y * K[1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def average_quaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:,eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigen_vectors[:,0])


def moving_average(data, window_size=5, fun=np.mean):
    data_smooth = np.zeros_like(data)
    half_window = window_size // 2
    for i in range(len(data)):
        # Determine the window indices
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        # Calculate the moving average of the quaternions in the window
        data_smooth[i] = fun(data[start:end])
    return data_smooth

def smooth_quaternions(data, window_size=5):
    return moving_average(data, window_size=window_size, fun=average_quaternions)

def smooth_3dvec(data, window_size=5):
    return moving_average(data, window_size=window_size, fun=partial(np.mean, axis=0))

def smooth_transforms(TCOs):
    TCOs = TCOs.copy()
    xyz = TCOs[:,:3,3]

    # Smooth translations:
    TCOs[:,:3,3] = smooth_3dvec(xyz, window_size=5)
    
    # Smooth rotations
    quats = Rotation.from_matrix(TCOs[:,:3,:3]).as_quat()
    quats = smooth_quaternions(quats, window_size=9)
    TCOs[:,:3,:3] = Rotation.from_quat(quats).as_matrix()
    return TCOs