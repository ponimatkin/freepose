import argparse
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--features_folder", type=str, default='objaverse_features_ffa_22')
    args.add_argument("--filelist", type=str, default='mesh_cache.txt')
    args = args.parse_args()

    objaverse_features_path = Path('data/datasets/').resolve() / args.features_folder

    with open(f'data/{args.filelist}', 'r', encoding='utf-8') as f:
        mesh_ids = f.read().splitlines()

    accumulated_features = []

    for mesh_id in mesh_ids:
        feature_file = objaverse_features_path / f'{mesh_id}.npy'
        if not feature_file.exists():
            print(f'Feature {feature_file} does not exist')
            continue

        features = np.load(feature_file)
        avg_feat = np.mean(features, axis=0)

        if np.isnan(avg_feat).any():
            print(f'Feature {feature_file} contains NaNs')
            continue

        accumulated_features.append(avg_feat)

    accumulated_features = np.stack(accumulated_features, axis=0)
    np.save(f'data/{args.features_folder}.npy', accumulated_features)