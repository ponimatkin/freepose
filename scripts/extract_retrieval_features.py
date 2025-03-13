import os
import cv2
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import argparse

from src.dataloader.template import WebTemplateDataset
from src.pipeline.retrieval.dino import DINOv2FeatureExtractor

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--shards_folder", type=str, default='objaverse_shards')
    args.add_argument("--filelist", type=str, default='mesh_cache.csv')
    args.add_argument("--feature", type=str, default='ffa', choices=['ffa', 'cls'])
    args.add_argument("--layer", type=int, default=22)
    args.add_argument("--mesh_per_job", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=128)
    args = args.parse_args()

    shards_path = Path('data/datasets').resolve() / args.shards_folder
    features_path = Path('data/datasets').resolve() / f'{args.shards_folder}_{args.feature}_{args.layer}'
    features_path.mkdir(parents=True, exist_ok=True)

    filelist_path = Path('data').resolve() / args.filelist
    model = DINOv2FeatureExtractor().to('cuda', dtype=torch.bfloat16)
    feature_type = 'cls' if args.feature == 'cls' else 'patch'

    dataset = WebTemplateDataset(shards_path.as_posix(), filelist_path.as_posix(), crop=False)

    job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    start_idx = job_id * args.mesh_per_job
    end_idx = min(start_idx + args.mesh_per_job, len(dataset))

    for idx in range(start_idx, end_idx):
        logger.info(f'Processing {idx+1} / {len(dataset)}')
        sample = dataset[idx]
        model_name = sample['model_name']
        templates = sample['templates'].to('cuda', dtype=torch.bfloat16)
        if sample['templates'] is not None:
            features = []
            for i in range(0, len(templates), args.batch_size):
                batch = templates[i:i+args.batch_size]
                features.append(model(batch, layer=args.layer, feature_type=feature_type))

            features = torch.cat(features, dim=0)

            if args.feature == 'ffa':
                avg_feats = []
                for feat, mask in zip(features, sample['masks']):
                    # resize mask from 420x420 to 30x30
                    mask_orig = mask.clone()
                    mask = cv2.resize(mask.float().numpy(), (30, 30), interpolation=cv2.INTER_AREA) > 0
                    avg_feat = feat[mask.flatten()]
                    # average features
                    avg_feat = avg_feat.mean(dim=0).float().cpu().numpy()

                    if np.isnan(avg_feat).any():
                        logger.warning(f'Feature {model_name} contains NaNs')
                        logger.warning(f'Mask sum is {mask.sum()}')
                        logger.warning(f'Mask orig sum is {mask_orig.sum()}')
                        cv2.imwrite(f'{model_name}_mask_orig.png', mask_orig.numpy().astype(np.uint8) * 255)
                        cv2.imwrite(f'{model_name}_mask.png', mask.astype(np.uint8) * 255)
                        continue
                    else:
                        avg_feats.append(avg_feat)

                avg_feats = np.stack(avg_feats)
                np.save((features_path / f'{model_name}.npy').as_posix(), avg_feats)
            else:
                np.save((features_path / f'{model_name}.npy').as_posix(), features.float().cpu().numpy())
        else:
            logger.warning(f"Skipping {sample['model_name']}")

    logger.info('Done')