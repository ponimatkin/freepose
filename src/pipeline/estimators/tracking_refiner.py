import math
import os
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger
from PIL import Image

from src.pipeline import refiner_utils

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender


class TrackingRefiner:
    def __init__(self, dino_model="dinov2_vitb14_reg", dino_device="cpu", cotracker_device="cpu"):
        self.dino_device = dino_device
        self.cotracker_device = cotracker_device
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dino_model).to(dino_device).eval()
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(cotracker_device).eval()
        
        self.image_size = int(math.sqrt(1370 - 1) * self.dinov2.patch_embed.proj.kernel_size[0])  # 518
        self.patch_size = self.dinov2.patch_embed.proj.kernel_size[0]  # 14
        self.feats_size = self.image_size // self.patch_size  # 37

    def _render(self, mesh, width, height, K, transform):
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.0001, zfar=9999)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[5.0, 5.0, 5.0])

        opencv2opengl = np.eye(4)
        opencv2opengl[1, 1] = -1
        opencv2opengl[2, 2] = -1

        scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), matrix=transform))
        scene.add_node(pyrender.Node(camera=camera, matrix=opencv2opengl))

        color_buff, depth_buff = renderer.render(scene)
        return color_buff, depth_buff

    def _crop_image(self, mesh, image, K, transform):
        vertices = np.asarray(mesh.vertices)
        np.random.seed(42)
        vertices = vertices[np.random.choice(np.arange(len(vertices)), 100)]
        vertices = torch.from_numpy(np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.).copy()).float()

        image = refiner_utils.MaybeToTensor()(image)
        K = torch.from_numpy(K).view(3, 3).float()
        transform = torch.from_numpy(transform).view(1, 4, 4).float()

        cropped_images, recomputed_bboxes = refiner_utils.crop_image(image, transform, vertices, K, 518, 518)
        new_Ks = refiner_utils.update_K_with_crop(K, recomputed_bboxes, 518, 518)

        return cropped_images[0], recomputed_bboxes[0], new_Ks[0]
    
    def _get_threshold_for_confidence(self, similarity_matrices, top_quantile=0.2):
        counts, values = np.histogram(similarity_matrices[similarity_matrices>0], bins=50)
        cutoff_value = counts.sum() * top_quantile
        cum_ = 0
        for c, v in zip(counts[::-1], values[:-1][::-1]):
            cum_ += c
            if cum_ > cutoff_value:
                break
        return v

    def pose_confidence(self, mesh, photo, K, transform):
        cropped_photo, new_bbox, new_K = self._crop_image(mesh, photo, K, transform)
        
        rendered_detection, rendered_depth = self._render(mesh, 518, 518, new_K, transform)
        rendered_detection = Image.fromarray(rendered_detection)
        render_valid_37x37_mask = cv2.resize((rendered_depth > 0).astype(np.float32), (37, 37), interpolation=cv2.INTER_CUBIC) > 0.5

        with torch.no_grad():
            input_ = refiner_utils.pil2torch(cropped_photo).unsqueeze(0).to(self.dino_device)
            photo_feats = self.dinov2.forward_features(input_)["x_norm_patchtokens"].cpu()
            photo_feats = photo_feats.squeeze(0).view(37, 37, photo_feats.shape[-1])
            photo_feats = photo_feats / torch.linalg.norm(photo_feats, dim=-1, keepdim=True)

            input_ = refiner_utils.pil2torch(rendered_detection).unsqueeze(0).to(self.dino_device)
            render_feats = self.dinov2.forward_features(input_)["x_norm_patchtokens"].cpu()
            render_feats = render_feats.squeeze(0).view(37, 37, render_feats.shape[-1])
            render_feats = render_feats / torch.linalg.norm(render_feats, dim=-1, keepdim=True)
        
        cosine_sim = (photo_feats * render_feats).sum(-1)
        cosine_sim = cosine_sim * torch.from_numpy(render_valid_37x37_mask).float()
        return cosine_sim.numpy()
    
    def n_inliers_per_pose(self, mesh, frames, K, transforms):
        confidences = []
        for frame, transform in tqdm.tqdm(zip(frames, transforms)):
            confidences.append(
                self.pose_confidence(mesh, Image.fromarray(frame), K, transform))
        confidences = np.stack(confidences)
        
        thr = self._get_threshold_for_confidence(confidences)
        return (confidences > thr).sum(-1).sum(-1), thr
    
    def _compute_3d_points(self, mesh, render_valid_coords, K, transform):
        real_coords = np.asarray(mesh.sample(10000))
        transformed_coords = (np.pad(real_coords, ((0, 0), (0, 1)), constant_values=1.) @ transform.T)[:, :3]
        projected_coords = transformed_coords @ K.T
        projected_coords = projected_coords[:, :2] / projected_coords[:, 2:]

        coords2indices = defaultdict(list)
        for i, p in enumerate(np.floor(projected_coords / self.patch_size).astype(np.int32)):
            coords2indices[tuple(p)].append(i)
        coords2indices = dict(coords2indices)

        render_real_coords = []
        for p in render_valid_coords:
            if tuple(p) not in coords2indices:
                logger.error(f"ERROR: 'pixel' {p} not in projected pointcloud!")
                render_real_coords.append(np.array([0, 0, 0]))
            else:
                indices = np.array(coords2indices[tuple(p)])
                # take points that project to given patch
                local_proj_coords = projected_coords[indices] / self.patch_size
                # select 25% of those points that are closest to the center of the patch
                closest_to_center_indices = np.argsort(
                    np.square(local_proj_coords - np.floor(local_proj_coords) - 0.5).sum(1))[
                                            :int(math.ceil(len(local_proj_coords) * 0.25))]
                # out of them, select the point closest to the camera
                min_idx = np.argmin(transformed_coords[indices[closest_to_center_indices]][:, 2])

                render_real_coords.append(real_coords[indices[closest_to_center_indices[min_idx]]])
        return np.stack(render_real_coords)
    
    def compute_2d3d_correspondences(self, mesh, photo, K, transform, mask=None):
        cropped_photo, new_bbox, new_K = self._crop_image(mesh, photo, K, transform)
        if mask is not None:
            cropped_mask, _, _, = self._crop_image(mesh, mask[:,:,None].astype(np.float32), K, transform)

            cropped_mask = cv2.resize(cropped_mask[0].numpy(), (37, 37), interpolation=cv2.INTER_CUBIC) > 0.5
            
        
        mesh_smaller = deepcopy(mesh.copy())
        mesh_smaller.vertices = mesh_smaller.vertices * 0.8
        
        _, rendered_depth = self._render(mesh_smaller, 518, 518, new_K, transform)
        render_valid_37x37_mask = cv2.resize((rendered_depth > 0).astype(np.float32), (37, 37), interpolation=cv2.INTER_CUBIC) > 0.5
        if mask is None:
            image_valid_coords = np.stack(np.where(render_valid_37x37_mask)[::-1], 1)
        else:
            image_valid_coords = np.stack(np.where(render_valid_37x37_mask & cropped_mask)[::-1], 1)
            if len(image_valid_coords) < 4:
                logger.warning("Not enought valid points in the mask, using unmasked points")
                image_valid_coords = np.stack(np.where(render_valid_37x37_mask)[::-1], 1)
        render_real_coords = self._compute_3d_points(mesh, image_valid_coords, new_K.numpy(), transform)
        
        # recompute image coords to the full (uncropped) image
        x1, y1, x2, y2 = new_bbox.numpy()
        tracking_query_points = np.float32(image_valid_coords) * self.patch_size + (self.patch_size * .5)
        tracking_query_points = tracking_query_points / self.image_size * np.array([[(x2 - x1), (y2 - y1)]]) + np.array([[x1, y1]])
        return tracking_query_points, render_real_coords
    
    def _track_frames(self, frames, query_points):
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.cotracker_device)  # B T C H W

        with torch.no_grad():
            # pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size, segm_mask=torch.from_numpy(masks_np[0])[None, None].float()) # B T N 2,  B T N 1
            pred_tracks, pred_visibility = self.cotracker(video, queries=torch.from_numpy(query_points)[None], backward_tracking=True)
        return pred_tracks.squeeze(0).cpu().numpy(), pred_visibility.squeeze(0).cpu().numpy()
        
    def _compute_pnp(self, image_keypoints, render_real_3d_points, matches, K):
        image_matching_2dpoints = np.array([image_keypoints[m.queryIdx] for m in matches])
        rendr_matching_3dpoints = np.array([render_real_3d_points[m.trainIdx] for m in matches])

        # _, rot, trans, kept_matches = cv2.solvePnPRansac(rendr_matching_3dpoints, image_matching_2dpoints, K, np.array([]), iterationsCount=300, confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
        _, rot, trans = cv2.solvePnP(rendr_matching_3dpoints, image_matching_2dpoints, K, np.array([]), flags=cv2.SOLVEPNP_EPNP)
        kept_matches = np.arange(len(matches))

        predicted_transform = np.eye(4)
        predicted_transform[:3, :3] = cv2.Rodrigues(rot)[0]
        predicted_transform[:3, 3] = trans.reshape(-1)
        return predicted_transform, kept_matches

    def compute_pnp_or_need_resample(self, mesh, photo, points2d, points2d_visibility, points3d, K):
        matches = [cv2.DMatch(i, i, 1) for i, bool_ in enumerate(points2d_visibility) if bool_]
        if len(matches) < 0.5 * len(points2d_visibility):
            return True, None
        predicted_transform, _ = self._compute_pnp(points2d, points3d, matches, K)
        new_points2d, _ = self.compute_2d3d_correspondences(mesh, photo, K, predicted_transform)

        old_vs_new = np.array([np.min(((p[np.newaxis] - points2d) ** 2).sum(1))
                               for p in new_points2d]) ** (1/2)
        new_vs_new = np.array([np.min(((p[np.newaxis] - new_points2d[np.arange(len(new_points2d)) != i]) ** 2).sum(1))
                               for i, p in enumerate(new_points2d)]) ** (1/2)

        return np.median(old_vs_new) > np.median(new_vs_new), predicted_transform
    
    def get_query_frames(self, n_inliers_per_frame, n_reference_detections=8):
        selected_query_frames = []
        n_frames = len(n_inliers_per_frame)
        span_ = int(n_frames / n_reference_detections / 2)
        arr_ = n_inliers_per_frame.copy()
        while len(selected_query_frames) < n_reference_detections:
            idx_ = np.argmax(arr_)
            selected_query_frames.append(idx_)
            arr_[max(idx_ - span_, 0):idx_ + span_ + 1] = 0

        return np.sort(selected_query_frames)
