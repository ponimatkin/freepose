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
        # Skip if we already processed the file (useful in case of error and restart)
        if os.path.exists((features_path / f'{dataset.get_model_name(idx)}.npy').as_posix()):
            logger.info("Model already processed and present on disk. Skipping.")
            continue

        try:
            sample = dataset[idx]
        except KeyError as e:
            logger.error(f"File missing {e}")
            continue
        model_name = sample['model_name']

    if sample['templates'] is not None and not os.path.exists((features_path / f'{model_name}.npy').as_posix()):
        # Move to GPU once, as the batch processing happens
        templates = sample['templates'].to('cuda', dtype=torch.bfloat16)

        # Pre-allocate list for features and compute all batches in one go
        features = []
        num_batches = len(templates) // args.batch_size + (len(templates) % args.batch_size != 0)

        # Process features in batches
        for i in range(num_batches):
            batch = templates[i * args.batch_size : (i + 1) * args.batch_size]
            features.append(model(batch, layer=args.layer, feature_type=feature_type))

        features = torch.cat(features, dim=0)  # Concatenate features after all batches are processed

        if args.feature == 'ffa':
            avg_feats = []
            for feat, mask in zip(features, sample['masks']):
                # Resize mask from 420x420 to 30x30 using torch for potentially better performance
                mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(30, 30), mode='bilinear', align_corners=False).squeeze()
                mask_resized = mask_resized > 0

                # Extract features where the mask is valid
                avg_feat = feat[mask_resized.flatten()]
                if avg_feat.numel() > 0:
                    avg_feat = avg_feat.mean(dim=0).float().cpu().numpy()
                else:
                    avg_feat = np.zeros(feat.shape[0], dtype=np.float32)

                # Check for NaNs (fewer checks after the loop)
                if np.isnan(avg_feat).any():
                    logger.warning(f'Feature {model_name} contains NaNs')
                    cv2.imwrite(f'{model_name}_mask_orig.png', mask.numpy().astype(np.uint8) * 255)
                    cv2.imwrite(f'{model_name}_mask.png', mask_resized.numpy().astype(np.uint8) * 255)
                else:
                    avg_feats.append(avg_feat)

            avg_feats = np.stack(avg_feats)
            np.save((features_path / f'{model_name}.npy').as_posix(), avg_feats)
        else:
            # Directly save features if 'ffa' is not the feature type
            np.save((features_path / f'{model_name}.npy').as_posix(), features.float().cpu().numpy())
    else:
        logger.warning(f"Skipping {sample['model_name']}")