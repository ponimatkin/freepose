import os
import cv2
import json
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from sam2.utils.amg import rle_to_mask
from loguru import logger
import argparse

from src.pipeline.utils import Proposals
from src.dataloader.bop import BOPDataset
from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.dataloader.template import WebTemplateDataset

from src.pipeline.estimators.scale_estimators import generate_pointcloud, get_scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def run():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str)
    args.add_argument('--split', type=str, default='test')
    args.add_argument('--proposals', type=str)
    args.add_argument('--layer', type=int, default=22)
    args.add_argument('--depth_method', type=str, default='zoedepth')
    args.add_argument('--bbox_extend', type=float, default=0.05)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--cache_size', type=int, default=50)
    args.add_argument('--save_all_cache', action='store_true')
    args = args.parse_args()

    proposals_path = Path('./data/results').resolve() / args.dataset / args.proposals

    array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
    pose_outputs = Path('./data/results').resolve() / args.dataset / args.proposals.replace('.json', f'_dinopose_layer_{args.layer}_bbext_{args.bbox_extend}_depth_{args.depth_method}_cache_{args.cache_size}')
    pose_outputs.mkdir(parents=True, exist_ok=True)
    pose_outputs = pose_outputs / f'pose_outputs_{array_task_id}.csv'

    dataset = BOPDataset(f'data/datasets/{args.dataset}/', args.split)
    templates = WebTemplateDataset('data/datasets/objaverse_shards', 'data/mesh_cache.csv', bbox_extend=args.bbox_extend)

    cache_dir = f'./data/cache_{array_task_id}_{args.dataset}'
    model = DinoPoseEstimator(n_poses=600, cache_size=args.cache_size, save_all=args.save_all_cache, cache_dir=cache_dir).to('cuda', dtype=torch.bfloat16)

    with open(proposals_path, 'r') as f:
        props = json.load(f)

    dataset_size = len(dataset)
    scenes_per_array = 30
    from_idx = array_task_id * scenes_per_array
    to_idx = min((array_task_id + 1) * scenes_per_array, dataset_size)
    
    results_dict = {
        "scene_id": [],
        "im_id": [],
        "obj_id": [],
        "score": [],
        "R": [],
        "t": [],
        'bbox_visib': [],
        'scale': [],
        "time": [],
    }

    for scene_idx in range(from_idx, to_idx):
        entry = dataset[scene_idx]

        img = entry['image']
        scene_id = int(entry['scene_id'])
        frame_id = int(entry['frame_id'])

        scene_proposals = [p for p in props if p['scene_id'] == scene_id and p['image_id'] == frame_id]
        if len(scene_proposals) == 0:
            continue
        masks = [rle_to_mask(prop['segmentation']) for prop in scene_proposals]
        boxes = [np.array(prop['bbox']) for prop in scene_proposals]
        scores = [prop['score'] for prop in scene_proposals]
        meshes = [prop['mesh'] for prop in scene_proposals]

        if args.depth_method == 'depthmap':
            pointclouds = [generate_pointcloud(entry['depth'], entry['intrinsic'], mask, svd=True) for mask in masks]
            scales = np.array([get_scale(pointcloud) for pointcloud in pointclouds])
            for prop, scale in zip(scene_proposals, scales):
                prop['scale'] = scale
        elif args.depth_method == 'const-0.05':
            scales = [0.05]*len(scene_proposals)
            for prop, scale in zip(scene_proposals, scales):
                prop['scale'] = scale
        elif args.depth_method == 'const-0.1':
            scales = [0.1]*len(scene_proposals)
            for prop, scale in zip(scene_proposals, scales):
                prop['scale'] = scale
        elif args.depth_method == 'zoedepth':
            scales = [np.clip(p['scale'], a_min=0.01, a_max=None) for p in scene_proposals]

        masks = torch.from_numpy(np.stack(masks))
        boxes = torch.from_numpy(np.stack(boxes))
        # convert boxes from xywh to xyxy
        boxes[:, 2:] += boxes[:, :2]
        out = {'boxes': boxes, 'masks': masks}
        proposals = Proposals(img, out, 420, bbox_extend=args.bbox_extend)
        proposals.scores = scores
        proposals.meshes = meshes

        for prop_idx, (prop, mask) in enumerate(zip(proposals.proposals, proposals.proposals_masks)):
            logger.info(f'Processing proposal {prop_idx+1}/{len(proposals.proposals)}, scene {scene_idx+1} / {dataset_size}')
            mesh_entry = templates.get_template_by_name(proposals.meshes[prop_idx])
            out = model(prop, mesh_entry, entry['intrinsic'], boxes[prop_idx], scales[prop_idx], layer=args.layer, batch_size=args.batch_size)

            results_dict["scene_id"].append(int(scene_id))
            results_dict["im_id"].append(int(frame_id))
            results_dict["obj_id"].append(proposals.meshes[prop_idx])
            results_dict["score"].append(out['scores'][0])
            R = out['TCO'][0][:3, :3].flatten().tolist()
            t = out['TCO'][0][:3, 3].tolist()

            bbox = out['bbox'].cpu().numpy()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            results_dict["R"].append(" ".join([str(x) for x in R]))
            results_dict["t"].append(" ".join([str(x*1000) for x in t]))
            results_dict["bbox_visib"].append(" ".join([str(x) for x in bbox]))
            results_dict["scale"].append(scales[prop_idx])
            results_dict["time"].append(0.2)

    df = pd.DataFrame(results_dict)
    df.to_csv(pose_outputs, index=False, header=True)
            
if __name__ == "__main__":
    run()