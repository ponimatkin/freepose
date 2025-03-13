import json

import numpy as np
import torch
from numpy.linalg import svd
from scipy.spatial import KDTree
from skimage.morphology import isotropic_erosion

from src.pipeline.utils import extract_largest_component


class ConstantScaleEstimator:
    def __init__(self, const) -> None:
        self.const = const

    def estimate(self, proposals, depth_image=None, K=None):
        return self.const
        

class MeanScaleEstimator:
    def __init__(self, mean_scale, svd=True):
        self.mean_scale = mean_scale
        self.svd = svd

    def estimate(self, proposals, depth_image, K):
        masks = [mask.cpu().numpy().astype(bool) for mask in proposals.masks]
        pointclouds = [generate_pointcloud(depth_image, K, mask, svd=self.svd) for mask in masks]
        scales = np.array([get_scale(pointcloud) for pointcloud in pointclouds])
        # Calculate the scale correction to match self.mean_scale
        correction = self.mean_scale/(2*np.mean(scales))
        scales *= correction
        return scales
    

class GPT4ScaleEstimator:
    def __init__(self, clip, query_k=11, scale_file=None, feats_path="data/scale_feats.pt", svd=True) -> None:
        self.clip = clip
        self.query_k = query_k
        self.svd = svd
        self.device = next(clip.parameters()).device

        if scale_file is not None:
            gpt_feats_scales = self.generate_clip_features(scale_file, clip, feats_path=feats_path)
        else:
            gpt_feats_scales = torch.load(feats_path)
        self.text_features = gpt_feats_scales["feats"]
        self.scales = gpt_feats_scales["scales"]
        self.kdtree = KDTree(self.text_features)
    
    def estimate(self, proposals, depth_image=None, K=None):
        assert (depth_image is None) == (K is None)
        use_depth = depth_image is not None and len(proposals.masks) > 1

        if use_depth:
            masks = [mask.cpu().numpy().astype(bool) for mask in proposals.masks]
            pointclouds = [generate_pointcloud(depth_image, K, mask, svd=self.svd) for mask in masks]
            depth_scales = np.array([get_scale(pointcloud) for pointcloud in pointclouds])

        image_features = []
        for prop in proposals.proposals:
            feats = self.clip(prop[None].to('cuda', dtype=torch.bfloat16))
            with torch.inference_mode():
                feats /= feats.norm(dim=-1, keepdim=True)
            image_features.append(feats[0].float().cpu().numpy())

        _, idx = self.kdtree.query(image_features, k=self.query_k)

        if self.query_k == 1:
            chatgpt_scales = self.scales[idx].numpy()
        else:
            # Get median scale over args.query_k most similar descriptors for each mask
            chatgpt_scales = self.scales[idx.reshape(-1)].reshape(idx.shape).median(axis=1).values

        if use_depth:
            correction = np.median(chatgpt_scales/depth_scales)
            scales = depth_scales*correction
        else:
            scales = chatgpt_scales
        
        return scales/2.0
        
    @staticmethod
    def generate_clip_features(scale_file, clip, feats_path="data/scale_feats.pt"):
        with open(scale_file) as f:
            gpt_scales = json.load(f)

        gpt_scales = list(gpt_scales.items())
        object_names = [x[0] for x in gpt_scales]
        scales = [x[1] for x in gpt_scales]

        text = clip.tokenizer(object_names)
        device = next(clip.parameters()).device
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip.model.encode_text(text.to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu()
    
        feats_scales = {"feats" : text_features, "scales" : torch.Tensor(scales)}
        if feats_path is not None:
            torch.save(feats_scales, feats_path)

        return feats_scales
        
        
    @staticmethod
    def mask_to_bbox(mask):
        # convert to numpy array if not already
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax


def get_scale(vertices):
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin)/2.0
    return scale

def svd_align(pointcloud):
    assert pointcloud.shape[1] == 3

    X = pointcloud - np.mean(pointcloud, axis=0)
    _, _, V = svd(X.T@X)
    return pointcloud@V.T


def generate_pointcloud(depth, K, mask, erosion_radius=8, std_factor=1.5, min_vertices=25, svd=False, rgb=None):
    mask = extract_largest_component(mask)

    # Mask erosion
    radius = erosion_radius
    m = isotropic_erosion(mask,radius)
    while np.sum(m) <= min_vertices:
        if radius < 1:
            m = mask
            break
        radius /= 2
        m = isotropic_erosion(mask,radius)
    
    # Get u,v coordinates and corresponding depth values
    v,u =  np.where(m)
    Z = depth[v,u]

    # Remove outliers : inliers = np.abs(z-np.median(z)) < np.std(z)*std_factor
    dists = np.abs(Z-np.median(Z))
    sort = np.argsort(dists)
    dists = dists[sort]
    Z = Z[sort]
    num_inliers = np.argmax(dists>np.std(Z)*std_factor)
    num_inliers = max(num_inliers, min_vertices) # make sure we have at least `min_vertices` vertices left

    Z = Z[:num_inliers]
    u = u[sort][:num_inliers]
    v = v[sort][:num_inliers]

    # Backproject points to 3D
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pointcloud = np.column_stack((X, Y, Z)).reshape(-1, 3)
    
    if svd:
        pointcloud = svd_align(pointcloud)

    if rgb is not None:
        return pointcloud, rgb[u,v]
    else:
        return pointcloud
    
