import json
import torch
import numpy as np
from pathlib import Path
import argparse

from loguru import logger
from src.dataloader.bop import BOPDataset
from sam2.utils.amg import rle_to_mask
from src.pipeline.utils import Proposals
from src.pipeline.retrieval.clip import CLIPFeatureExtractor
from src.pipeline.estimators.scale_estimators import GPT4ScaleEstimator


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str)
    args.add_argument("--proposals", type=str)
    args.add_argument("--split", type=str, default='test')
    args = args.parse_args()

    results_path = Path('./data/results').resolve()  / args.dataset / args.proposals
    results_path_out = Path('./data/results').resolve() / args.dataset / args.proposals.replace('.json', '_gpt4_scaled.json')

    with open(results_path, "r") as f:
        props = json.load(f)

    clip = CLIPFeatureExtractor().to('cuda', dtype=torch.bfloat16)
    scale_estimator = GPT4ScaleEstimator(clip, scale_file="data/gpt4_scales.json")

    dataset = BOPDataset(f'data/datasets/{args.dataset}/', args.split)

    for e_idx, entry in enumerate(dataset):
        logger.info(f"Processing {e_idx}/{len(dataset)}")
        image = entry['image']
        gt_masks = entry['masks']
        K = entry['intrinsic']
        depth_pred = entry['depth_pred']

        scene_id = entry["scene_id"]
        image_id = entry["frame_id"]

        scene_proposals = [p for p in props if p['scene_id'] == scene_id and p['image_id'] == image_id]
        masks = [rle_to_mask(prop['segmentation']) for prop in scene_proposals]
        boxes = [np.array(prop['bbox']) for prop in scene_proposals]
        scores = [prop['score'] for prop in scene_proposals]
        meshes = [prop['mesh'] for prop in scene_proposals]

        masks = torch.from_numpy(np.stack(masks))
        boxes = torch.from_numpy(np.stack(boxes))
        # convert bbox from xywh to xyxy
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        out = {'boxes': boxes, 'masks': masks}
        proposals = Proposals(image, out, 224, bbox_extend=0.05)

        img_pred_scales = scale_estimator.estimate(proposals, depth_pred, K)

        for i, scale in enumerate(img_pred_scales):
            scene_proposals[i]["scale"] = float(scale.item())

    with open(str(results_path_out), "w") as f:
        json.dump(props, f)




    