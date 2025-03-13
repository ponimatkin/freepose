import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.pipeline.retrieval.dino import DINOv2FeatureExtractor
from src.pipeline.utils import Proposals, mask_to_bbox


def get_init_bboxes(image, text_prompt, box_thresh, text_thresh, device="cuda"):
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3 and image.shape[2] == 3

    logger.info("Loading Grounding DINO model")

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    objectness_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = objectness_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.shape[:2]],
    )[0]

    bboxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    idxs = np.where(np.array(labels) != '')[0]
    bboxes = [bboxes[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    labels = [labels[i] for i in idxs]

    return bboxes, scores, labels


def track_with_sam2(video_dir, boxes, scores, reverse=False, device="cuda"):
    logger.info("Loading SAM 2 model")
    checkpoint = "./data/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(video_dir))

    logger.info("Tracking masks with SAM2")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for object_id, (bbox, score) in enumerate(zip(bboxes, scores)):
            _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                box=bboxes[object_id],
            )

        ignore_objects = set()
        tracking_output = dict()

        start_frame = len(frame_paths) - 1 if reverse else 0
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state, reverse=reverse, start_frame_idx=start_frame):
            scores = [1.0 for i in range(len(obj_ids))]
            masks = [(mask_logits[i] > 0.0)[0] for i in range(len(obj_ids))]

            boxes = []
            for i, mask in enumerate(masks):
                if mask.sum() < 100:
                    ignore_objects.add(i)
                    boxes.append(None)
                    continue

                bbox = mask_to_bbox(mask.cpu().numpy())
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                if w < 10 or h < 10:
                    ignore_objects.add(i)
                    boxes.append(None)
                    continue

                boxes.append(bbox)

            tracking_output[frame_idx] = {
                "boxes": boxes,
                "masks": masks,
                "scores": scores,
            }

    # Remove ignored objects
    if len(ignore_objects) > 0:
        logger.info(f"Ignoring objects: {ignore_objects}")

    obj_idxs = sorted(list(ignore_objects))[::-1]
    for frame_idx, output in tracking_output.items():
        for idx in obj_idxs:
            output["boxes"].pop(idx)
            output["masks"].pop(idx)
            output["scores"].pop(idx)
        output["boxes"] = torch.tensor(np.array(output["boxes"])).cuda()
        output["masks"] = torch.stack(output["masks"]).cuda()

    return tracking_output


def retrieve_meshes(tracking_output, retrieval, retrieval_features, filelist_path, layer=22, topk=0, device='cuda'):
    logger.info("Retrieving meshes")
    feature_extractor = DINOv2FeatureExtractor().to(device, dtype=torch.bfloat16)
    all_proposals = dict()
    softvote_scores = []

    with open(f"data/{filelist_path}", "r") as f:
        filelist = f.read().splitlines()

    with torch.inference_mode():
        for frame_idx, output in tracking_output.items():
            logger.info(f"Processing frame {frame_idx}/{len(tracking_output)}")
            frame_softvote_scores = []

            image = cv2.cvtColor(cv2.imread(str(frame_paths[frame_idx])), cv2.COLOR_BGR2RGB).astype(np.uint8)
            proposals = Proposals(image, output, 420, 0, frame_idx, bbox_extend=0.1, mask_rgb=False)

            if feature_type == "cls":
                proposals.features = F.normalize(feature_extractor(proposals.proposals.to(device, dtype=torch.bfloat16), feature_type="cls",layer=layer), dim=-1)
            elif feature_type == "ffa":
                proposal_features_raw = feature_extractor(proposals.proposals.to(device, dtype=torch.bfloat16), feature_type="patch", layer=layer)

                feats = []
                masks_downsampled = [cv2.resize(mask.float().cpu().numpy(), (30, 30), interpolation=cv2.INTER_AREA) > 0 for mask in proposals.proposals_masks]
                for feat, mask in zip(proposal_features_raw, masks_downsampled):
                    feat_ffa = feat[mask.flatten()]
                    feats.append(feat_ffa.mean(dim=0))

                proposals.features = F.normalize(torch.stack(feats), dim=-1)

            for prop_idx, feature in enumerate(proposals.features):
                scores = (retrieval_features @ feature).float()

                # get top 100 scores for coarse retrieval
                scores, I = torch.topk(scores, 100)

                s = torch.zeros(retrieval_features.shape[0])
                if topk == 0:
                    # take the first mesh and score
                    proposals.meshes.append(filelist[I[0].item()])
                    proposals.scores.append(scores[0].item())
                    s[I.cpu()] = scores.cpu()
                else:
                    # do per-view fine retrieval for top 100
                    scores = {}
                    for i, idx in enumerate(I.cpu().numpy()):
                        finegrained_features = torch.from_numpy(np.load(f"data/datasets/{retrieval}/{filelist[idx]}.npy")).to("cuda", dtype=torch.bfloat16)
                        finegrained_features = F.normalize(finegrained_features, dim=-1)
                        # take top-k scores
                        pred_scores = (finegrained_features @ feature).float()
                        topk_scores, topk_idx = torch.topk(pred_scores, topk)
                        scores[filelist[idx]] = topk_scores.cpu().numpy().mean().item()
                        s[idx] = topk_scores.cpu().numpy().mean().item()

                    # take mesh with highest score from top 100
                    mesh_id = max(scores, key=scores.get)
                    proposals.meshes.append(mesh_id)
                    proposals.scores.append(scores[mesh_id])
                frame_softvote_scores.append(s)

            del proposals.features
            del proposals.proposals
            proposals.features = None
            proposals.proposals = None
            
            all_proposals[frame_idx] = proposals
            softvote_scores.append(torch.stack(frame_softvote_scores))

    # Aggregate the per-frame scores (soft-voting)
    softvote_scores = torch.mean(torch.stack(softvote_scores), axis=0)
    scores, I = torch.topk(softvote_scores, 1, dim=1)
    scores = [scores[i].item() for i in range(len(proposals.scores))]
    meshes = [filelist[I[i].item()] for i in range(len(proposals.meshes))]

    logger.info(f"Retrieved meshes: {meshes}")
    for frame_idx, proposals in all_proposals.items():
        proposals.scores = scores
        proposals.meshes = meshes

    return all_proposals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--retrieval", type=str, default="objaverse_shards_ffa_22")
    parser.add_argument("--filelist", type=str, default="mesh_cache.txt")
    parser.add_argument("--box_thresh", type=float, default=0.2)
    parser.add_argument("--text_thresh", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--prompt", type=str, default="objects.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path("data/datasets/videos") / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])

    feature_type = "ffa" if "ffa" in args.retrieval else "cls"
    layer = int(args.retrieval.split("_")[-1])
    results = Path("data/results/videos").resolve() / args.video
    results.mkdir(parents=True, exist_ok=True)
    output_file = (results / f"props-ground-box-{args.box_thresh}-text-{args.text_thresh}-{feature_type}-{layer}-top-{args.topk}_{args.video}.json")

    # Predict initial bboxes
    image_path = frame_paths[-1 if args.reverse else 0]
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(np.uint8)
    image_h, image_w, _ = image.shape
    bboxes, scores, labels = get_init_bboxes(image, args.prompt, args.box_thresh, args.text_thresh, device=device)

    from scripts.vis_detections_video import vis_detections
    viz_path = results / f"viz_detections_{args.video}.png"
    vis_detections(image, bboxes, viz_path, xywh=False)
    logger.info(f"Saved visualization to {viz_path}")

    # Track with SAM2
    tracking_output = track_with_sam2(video_dir, bboxes, scores, reverse=args.reverse, device=device)

    # Retrieve meshes
    retrieval_features = np.load(f"data/{args.retrieval}.npy")
    retrieval_features = torch.from_numpy(retrieval_features).to("cuda", dtype=torch.bfloat16)
    retrieval_features = F.normalize(retrieval_features, dim=-1)

    all_proposals = retrieve_meshes(
        tracking_output,
        args.retrieval,
        retrieval_features,
        args.filelist,
        layer=layer,
        topk=args.topk,
        device=device,
    )

    # Convert to BOP format
    out_file = []
    for frame_idx, proposals in all_proposals.items():
        bop_dict = proposals.to_bop_dict()
        out_file.extend(bop_dict)

    # Save proposals to file
    with open(output_file, "w") as f:
        json.dump(out_file, f)
