import json
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from sam2.utils.amg import mask_to_rle_pytorch
from scipy.ndimage import label
from scipy.spatial import KDTree
from skimage.measure import regionprops
from skimage.morphology import isotropic_erosion

from src.utils.bbox_utils import CropResizePad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Proposals:
    def __init__(self, image, detections_output, target_size=350, scene_id=None, frame_id=None, bbox_extend=0.2, mask_rgb=True):
        self.image = (torch.tensor(image).float()/255).permute(2, 0, 1).to(detections_output['masks'].device)
        self.masks = detections_output['masks'].bool()
        self.boxes = detections_output['boxes'].int()
        self.rgb_proposal_processor = CropResizePad(target_size=target_size, orig_size=(image.shape[0], image.shape[1]), bbox_extend=bbox_extend)

        self.proposals, self.proposals_masks = self.extract_proposals(mask_rgb=mask_rgb)
        self.features = None
        self.scores = []
        self.meshes = []
        self.scene_id = scene_id
        self.frame_id = frame_id

    def extract_proposals(self, mask_rgb=True):
        num_proposals = len(self.masks)
        rgbs = self.image.unsqueeze(0).repeat(num_proposals, 1, 1, 1).float()
        masks = self.image.unsqueeze(0).repeat(num_proposals, 1, 1, 1).float()
        masks[~self.masks.unsqueeze(1).repeat(1, 3, 1, 1)] = 0
        masks[self.masks.unsqueeze(1).repeat(1, 3, 1, 1)] = 1
        
        if mask_rgb:
            masked_rgbs = rgbs * self.masks.unsqueeze(1)
        else:
            masked_rgbs = rgbs# * self.masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, self.boxes
        )  # [N, 3, target_size, target_size]

        processed_masks = self.rgb_proposal_processor(
            masks, self.boxes
        )  # [N, 3, target_size, target_size]

        processed_masks = processed_masks[:, 0] > 0.5
        return processed_masked_rgbs, processed_masks
    
    def to_bop_dict(self):
        bop_dict = []
        for i in range(len(self.boxes)):
            bbox = self.boxes[i].cpu().numpy().tolist()
            # convert x1 y1 x2 y2 to x y w h
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            bop_dict.append({
                'bbox': bbox,
                'segmentation': mask_to_rle_pytorch(self.masks[i][None])[0],
                'mesh': self.meshes[i],
                'score': self.scores[i],
                'scene_id': int(self.scene_id),
                'image_id': int(self.frame_id),
                'time': 0.01
            })
        return bop_dict
            
def extract_largest_component(mask):
    # Label the connected components in the mask
    labeled_mask, num_labels = label(mask)

    # Get the properties of the connected components
    properties = regionprops(labeled_mask)

    # Find the label of the largest connected component
    largest_component_label = max(properties, key=lambda x: x.area).label

    # Create a mask for the largest connected component
    largest_component_mask = (labeled_mask == largest_component_label)

    return largest_component_mask

def generate_point_cloud(rgb_image, depth_image, K, mask, erosion_radius=1, std_dev=3.0, align=True):
    mask = isotropic_erosion(extract_largest_component(mask), radius=erosion_radius)
    # Apply the mask to the images
    rgb_image = rgb_image[mask]
    depth_image = depth_image[mask]

    # Get the indices of the mask
    v, u = np.where(mask)

    # Get the camera parameters from the K matrix
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Convert the depth image to meters
    Z = depth_image

    # Convert the pixel coordinates to 3D world coordinates
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
        
    # Stack the 3D points and the colors into a single array
    point_cloud = np.column_stack((X, Y, Z, rgb_image/255)).reshape(-1, 6)

    # get median depth
    mask = np.abs(Z-np.median(Z)) < np.std(Z)*std_dev
    point_cloud = point_cloud[mask]

    if align:
        X = point_cloud[:, :3] - np.mean(point_cloud[:, :3], axis=0)
        _, _, V = np.linalg.svd(X.T@X)
        point_cloud[:, :3] = point_cloud[:, :3]@V.T

    return point_cloud

def depthmap_to_pointcloud(depth_map, K):
    # Invert K
    K_inv = np.linalg.inv(K)

    # Create a meshgrid for pixel coordinates
    height, width = depth_map.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Stack x, y, and ones to create homogeneous coordinates
    homogeneous_coordinates = np.stack((x, y, np.ones_like(x)), axis=2)

    # Reshape to (number of points, 3)
    homogeneous_coordinates = homogeneous_coordinates.reshape(-1, 3)

    # Backproject to 3D space
    depth_map_flat = depth_map.flatten()  # Depth map is saved in 16-bit PNG in millimeters
    point_cloud = np.dot(K_inv, homogeneous_coordinates.T) * depth_map_flat

    # Transpose to get the point cloud in shape (number of points, 3)
    point_cloud = point_cloud.T
    point_cloud = point_cloud[~np.all(point_cloud == 0, axis=1)]
    return point_cloud
    

def get_z_from_pointcloud(bbox, pointcloud, K, TCO_init):
    TCO = TCO_init.copy()
    z_guess = TCO[2, 3]
    fxfy = K[[0, 1], [0, 1]]
    cxcy = K[[0, 1], [2, 2]]
    bb_xy_centers = (bbox[0:2] + bbox[2:4]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy

    TCO[:2, 3] = xy_init
    deltax_3d = pointcloud[:, 0].max() - pointcloud[:, 0].min()
    deltay_3d = pointcloud[:, 1].max() - pointcloud[:, 1].min()
    bb_deltax = (bbox[2] - bbox[0]) + 1
    bb_deltay = (bbox[3] - bbox[1]) + 1

    z_from_dx = fxfy[0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[1] * deltay_3d / bb_deltay

    z = (z_from_dy + z_from_dx) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:2, 3] = xy_init
    TCO[2, 3] = z
    return TCO

def mask_to_bbox(mask):
        # Get the indices of non-zero elements
        y_indices, x_indices = np.nonzero(mask)
        
        # Get the min and max indices for each dimension
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Return the coordinates of the bounding box
        return np.array([x_min, y_min, x_max, y_max])