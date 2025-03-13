import torch
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from loguru import logger

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str)
    args.add_argument("--split", type=str, default="test")
    args = args.parse_args()

    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    dataset_path = Path('data/datasets').resolve() / args.dataset / args.split
    
    for scene in dataset_path.iterdir():
        logger.info(f"Processing {scene}")
        depth_pred = scene / 'depth_pred'
        depth_pred.mkdir(exist_ok=True)
        
        rgb_folder = scene / 'rgb'
        for rgb_path in rgb_folder.iterdir():
            image = Image.open(rgb_path)
            depth = model.infer_pil(image)
            depth = (depth * (2**16-1)/np.max(depth)).astype(np.uint16)
            depth_path = depth_pred / rgb_path.name
            Image.fromarray(depth).save(depth_path)