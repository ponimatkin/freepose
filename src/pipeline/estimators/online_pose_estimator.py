import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from scipy.spatial.transform import Rotation as Rot

from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.pipeline.retrieval.dino import DINOv2FeatureExtractor
from src.pipeline.retrieval.renderer import MeshRenderer
from src.pipeline.utils import depthmap_to_pointcloud, get_z_from_pointcloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoOnlinePoseEstimator(nn.Module):
    def __init__(self, n_coarse_poses=600, n_fine_poses=10000, cache_size=50, save_all=False, cache_dir='./data/cache'):
        super(DinoOnlinePoseEstimator, self).__init__()
        self.coarse_estimator = DinoPoseEstimator(n_coarse_poses, cache_size, save_all, cache_dir)
        self.feature_extractor = DINOv2FeatureExtractor().to('cuda', dtype=torch.bfloat16)
        self.fine_mesh_poses = np.array(self.coarse_estimator.generate_poses(n_fine_poses))
        self.renderer = MeshRenderer(n_fine_poses)
        self.rendering_scale = 0.25

    @staticmethod
    def geodesic_distance(render_poses, query_pose, degrees=True):
        render_rots = render_poses[:,:3,:3]
        query_rot = query_pose[:3,:3]
        diffs = render_rots @ query_rot.T
        dists = np.linalg.norm(Rot.from_matrix(diffs).as_rotvec(),axis=1)
        if degrees:
            dists = np.rad2deg(dists)

        return dists
    
    def forward(self, proposal, proposal_mask, template_dict, mesh, K, bbox, est_scale, prev_pose=None, neighborhood=15, layer=22, batch_size=128, mask_scores=False):
        if prev_pose is None:
            # Run the coarse pose estimator first
            #TODO: Include proposal_mask in the coarse pose estimator?
            coarse_results = self.coarse_estimator.forward(proposal, template_dict, K, bbox, est_scale, layer, batch_size, return_query_feat=True)
            query_feat = coarse_results['query_feat']
            prev_pose = coarse_results['TCO'][0]
        else:
            query_feat = None
        
        # Run the refiner on the closest poses
        return self.forward_fine(proposal, proposal_mask, template_dict, mesh, K, bbox, est_scale, prev_pose, neighborhood, layer, mask_scores, query_feat)
    
    def forward_fine(self, proposal, proposal_mask, template_dict, mesh, K, bbox, est_scale, prev_pose, neighborhood=15, layer=22, mask_scores=False, query_feat=None):
        if query_feat is None:
            query_feat = self.feature_extractor(proposal[None].cuda().half(), layer=layer, feature_type='patch')
            query_feat = F.normalize(query_feat, dim=-1)
        
        # find the closest poses in the fine sampling
        dists = self.geodesic_distance(self.fine_mesh_poses[:, :3, :3], prev_pose)
        close_poses = np.where(dists < neighborhood)[0]

        # render the fine poses
        selected_poses = self.fine_mesh_poses[close_poses]
        mesh.apply_scale(self.rendering_scale)
        renders = self.renderer.render_from_poses(mesh, selected_poses)
        ren_props, poses, masks_fine_template = self.renderer.generate_proposals(renders)
        masks_fine_template = torch.Tensor(np.array(masks_fine_template)).bool()
        mesh.apply_scale(1/self.rendering_scale)

        feats_fine_template = self.feature_extractor(ren_props.cuda().half(), layer=layer, feature_type='patch')

        signature = 'b n d, b n d -> b n'
        if mask_scores:
            scores = einsum(query_feat, F.normalize(feats_fine_template, dim=-1), signature)
            masks = torch.logical_or(masks_fine_template,proposal_mask[None]).to(torch.float16)
            n_views = feats_fine_template.shape[0]
            masks = F.interpolate(masks[None].float(), size=(30, 30), mode='bilinear').reshape(n_views,900).cuda()
            scores = (scores*masks).sum(dim=-1)/masks.sum(dim=-1)
        else:
            scores = einsum(query_feat, F.normalize(feats_fine_template, dim=-1), signature).mean(dim=-1)

        top_score = torch.max(scores)
        top_index = torch.argmax(scores)

        point_cloud = depthmap_to_pointcloud(renders[top_index][1], template_dict['intrinsic'].numpy())
        # rescale point cloud from 0.25 (rendering scale) to 1.0
        point_cloud /= 0.25
        # rescale point cloud to the estimated scale
        point_cloud *= est_scale
        TCO = get_z_from_pointcloud(bbox, point_cloud, K, poses[top_index])

        out_dict = {
            'TCO': [TCO],
            'scores': [top_score.float().cpu().numpy()],
            'proposal': proposal,
            'K': K,
            'bbox': bbox,
        }

        return out_dict