import argparse
import json
from collections import defaultdict
from itertools import takewhile
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from sam2.utils.amg import rle_to_mask

from src.pipeline.estimators.scale_estimators import GPT4ScaleEstimator
from src.pipeline.retrieval.clip import CLIPFeatureExtractor
from src.pipeline.utils import Proposals

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--video", type=str)
    args.add_argument("--proposals", type=str)
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path("data") / "datasets" / "videos" / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])

    results_dir = (Path("data") / "results" / "videos" / args.video).resolve()
    input_path = results_dir / args.proposals
    output_path = results_dir / args.proposals.replace(".json", "_gpt4_scaled.json")

    with open(input_path, "r") as f:
        proposals_all = json.load(f)

    N_objects = len(list(takewhile(lambda x: x['image_id']==0, proposals_all)))
    

    clip = CLIPFeatureExtractor().to(device, dtype=torch.bfloat16)
    scale_estimator = GPT4ScaleEstimator(clip, scale_file="data/gpt4_scales.json")
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)

    image_path = frame_paths[0]
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(
        np.uint8
    )
    image_h, image_w, _ = image.shape
    cx = image_w / 2.0
    cy = image_h / 2.0
    f = np.sqrt(image_h**2 + image_w**2)
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]]
    ).astype(float)

    props = defaultdict(list)
    for p in proposals_all:
        props[p["image_id"]].append(p)

    for frame_idx, frame_path in enumerate(frame_paths):
        logger.info(f"Processing frame {frame_idx}/{len(frame_paths)}")

        image = cv2.cvtColor(
            cv2.imread(str(frame_paths[frame_idx])), cv2.COLOR_BGR2RGB
        ).astype(np.uint8)

        frame_props = props[frame_idx]
        masks = [rle_to_mask(p["segmentation"]) for p in frame_props]
        boxes = [np.array(p["bbox"]) for p in frame_props]
        scores = [p["score"] for p in frame_props]
        meshes = [p["mesh"] for p in frame_props]

        masks = torch.from_numpy(np.stack(masks))
        boxes = torch.from_numpy(np.stack(boxes))
        # convert bbox from xywh to xyxy
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        out = {"boxes": boxes, "masks": masks}
        proposals = Proposals(image, out, 224, bbox_extend=0.05)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            depth_pred = zoe.infer(proposals.image[None])[0, 0].detach().cpu().numpy()
            img_pred_scales = scale_estimator.estimate(proposals, depth_pred, K)

        for i, scale in enumerate(img_pred_scales):
            frame_props[i]["scale"] = float(scale.item())

    # replace scales with medians for each tracked object
    for object_idx in range(N_objects):
        object_proposals = proposals_all[object_idx::N_objects]
        scales = [x["scale"] for x in object_proposals]
        scale = np.median(scales)
        for p in object_proposals:
            p["scale"] = scale
    with open(str(output_path), "w") as f:
        json.dump(proposals_all, f)
