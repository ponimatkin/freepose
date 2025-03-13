import shutil
from collections import OrderedDict
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from scipy.spatial.transform import Rotation as Rot

from src.pipeline.retrieval.dino import DINOv2FeatureExtractor
from src.pipeline.utils import depthmap_to_pointcloud, get_z_from_pointcloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoPoseEstimator(nn.Module):
    def __init__(self, n_poses=600, cache_size=50, save_all=False, cache_dir='./data/cache'):
        super(DinoPoseEstimator, self).__init__()
        self.feature_extractor = DINOv2FeatureExtractor().to('cuda', dtype=torch.bfloat16)
        self.mesh_poses = self.generate_poses(n_poses)
        self.feature_cache = OrderedDict()
        self.cache_size = cache_size
        self.save_all = save_all
        self.cache_dir = cache_dir
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def _extract_features(self, proposals, layer=22, batch_size=128):
        feats = []
        for i in range(0, len(proposals), batch_size):
            feats.append(self.feature_extractor(proposals[i:i+batch_size].to('cuda', dtype=torch.bfloat16), layer=layer, feature_type='patch'))
        feats = torch.cat(feats, dim=0)
        return feats
    
    def _cache_features(self, key, features):
        features_cpu = features.to('cpu')
        self.feature_cache[key] = features_cpu
        self.feature_cache.move_to_end(key)

        cache_path = self.cache_dir / f"{key}.pth"
        if self.save_all and not cache_path.exists():
            with open(cache_path, 'w') as f:
                flock(f, LOCK_EX)
                torch.save(features_cpu, f)
                flock(f, LOCK_UN)

        if len(self.feature_cache) > self.cache_size:
            oldest_key, oldest_feature = self.feature_cache.popitem(last=False)
            cache_path = self.cache_dir / f"{oldest_key}.pth"
            torch.save(oldest_feature, cache_path)

    def _get_template_features(self, template_dict, layer=22, batch_size=128):
        # check RAM cache
        if template_dict['model_name'] in self.feature_cache:
            features = self.feature_cache[template_dict['model_name']]
            self.feature_cache.move_to_end(template_dict['model_name'])
            return features.to('cuda', dtype=torch.bfloat16)
        
        # check disk cache
        cache_path = self.cache_dir / f"{template_dict['model_name']}.pth"
        if cache_path.exists():
            features = torch.load(cache_path)
            self._cache_features(template_dict['model_name'], features)
            self.feature_cache.move_to_end(template_dict['model_name'])
            return features.to('cuda', dtype=torch.bfloat16)
        
        # extract features
        features = self._extract_features(template_dict['templates'], layer=layer, batch_size=batch_size)
        self._cache_features(template_dict['model_name'], features)
        self.feature_cache.move_to_end(template_dict['model_name'])
        return features.to('cuda', dtype=torch.bfloat16)
    
    def __del__(self):
        shutil.rmtree(self.cache_dir)

    def forward(self, proposal, template_dict, K, bbox, est_scale, layer=22, batch_size=128, return_query_feat=False):
        if self.cache_size > 0:
            feats_template = self._get_template_features(template_dict, layer=layer, batch_size=batch_size)
        elif self.cache_size == 0:
            feats_template = self._extract_features(template_dict['templates'], layer=layer, batch_size=batch_size)
        query_feat = self.feature_extractor(proposal[None].to('cuda', dtype=torch.bfloat16), layer=layer, feature_type='patch')
        signature = 'b n d, b n d -> b n'

        # do product between NxD templates and 1xD query
        scores = einsum(F.normalize(feats_template, dim=-1), F.normalize(query_feat, dim=-1), signature).mean(dim=-1)

        top_scores, top_indices = torch.topk(scores, 3)
        top_scores = top_scores.float().cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        out_dict = {
            'TCO': [],
            'scores': top_scores,
            'proposal': proposal,
            'K': K,
            'bbox': bbox,
            'retrieved_proposals': [template_dict['templates'][idx] for idx in top_indices],
        }

        for idx in top_indices:
            point_cloud = depthmap_to_pointcloud(template_dict['depths'][idx].numpy(), template_dict['intrinsic'].numpy())
            mean_translation = point_cloud.mean(axis=0)
            point_cloud -= mean_translation
            # rescale point cloud from 0.25 (rendering scale) to 1.0
            point_cloud /= 0.25
            # rescale point cloud to the estimated scale
            point_cloud *= est_scale
            point_cloud += mean_translation
            TCO = get_z_from_pointcloud(bbox, point_cloud, K, self.mesh_poses[idx])
            out_dict['TCO'].append(TCO)

        if return_query_feat:
            out_dict['query_feat'] = query_feat

        return out_dict
            

    @staticmethod
    def generate_poses(n_poses=600):
        phi = np.sqrt(2.0)
        psi = 1.533751168755204288118041
        
        Q = np.empty(shape=(n_poses,4), dtype=float)
        rotations = [] # opencv
        mesh_poses = [] # opengl
        for i in range(n_poses):
            s = i+0.5
            r = np.sqrt(s/n_poses)
            R = np.sqrt(1.0-s/n_poses)
            alpha = 2.0 * np.pi * s / phi
            beta = 2.0 * np.pi * s / psi
            Q[i,0] = r*np.sin(alpha)
            Q[i,1] = r*np.cos(alpha)
            Q[i,2] = R*np.sin(beta)
            Q[i,3] = R*np.cos(beta)

            rotations.append(Rot.from_quat(Q[i]).as_matrix())

            mesh_pose = np.eye(4)
            mesh_pose[:3, 3] = np.array([0, 0, 1.1])
            mesh_pose[:3, :3] = rotations[-1]
            mesh_poses.append(mesh_pose)

        return mesh_poses




