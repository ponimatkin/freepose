import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import json
import argparse
from loguru import logger

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from src.pipeline.retrieval.dino import DINOv2FeatureExtractor
from src.dataloader.bop import BOPDataset
from src.pipeline.utils import Proposals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--retrieval", type=str, default='objaverse_shards_ffa_22')
    parser.add_argument("--filelist", type=str, default='mesh_cache.txt')
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--text_thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=0)
    args = parser.parse_args()

    dataset = BOPDataset(f'data/datasets/{args.dataset}/', args.split)
    
    feature_type = 'ffa' if 'ffa' in args.retrieval else 'cls'
    layer = int(args.retrieval.split('_')[-1])
    results = Path('data/results').resolve() / args.dataset
    output_file = results / f'props-ground-box-{args.box_thresh}-text-{args.text_thresh}-{feature_type}-{layer}-top-{args.topk}_{args.dataset}-{args.split}.json'
    results.mkdir(parents=True, exist_ok=True)

    retrieval_features = np.load(f'data/{args.retrieval}.npy')
    retrieval_features = torch.from_numpy(retrieval_features).to('cuda', dtype=torch.bfloat16)
    retrieval_features = F.normalize(retrieval_features, dim=-1)

    with open(f'data/{args.filelist}', 'r') as f:
        filelist = f.read().splitlines()

    feature_extractor = DINOv2FeatureExtractor().to('cuda', dtype=torch.bfloat16)
    
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    objectness_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    checkpoint = "./data/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device='cuda'))

    out_file = []
    for e_idx, entry in enumerate(dataset):
        logger.info(f"Processing {e_idx}/{len(dataset)}")
        predictor.set_image(entry['image'])

        inputs = processor(images=entry['image'], text='objects.', return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = objectness_model(**inputs)


        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_thresh,
            text_threshold=args.text_thresh,
            target_sizes=[entry['image'].shape[:2]]
        )[0]

        bboxes = results['boxes'].cpu().numpy()

        if bboxes.shape[0] == 0:
            continue

        output = {
            'boxes': [],
            'masks': [],
            'scores': [], 
        }

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
            )

        # remove masks for which sum is less than 100
        for score, mask, box in zip(scores, masks, bboxes):
            if sum(mask[0].flatten()) < 100:
                continue

            # Get the indices of non-zero elements
            y_indices, x_indices = np.nonzero(mask[0])

            # Get the min and max indices for each dimension
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)

            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                continue

            output['masks'].append(mask[0])
            output['boxes'].append(box)
            output['scores'].append(score)

        output['masks'] = torch.tensor(output['masks']).cuda()
        output['boxes'] = torch.tensor(output['boxes']).cuda()

        if len(output['masks']) == 0:
            continue

        proposals = Proposals(entry['image'], output, 420, entry['scene_id'], entry['frame_id'], bbox_extend=0.1, mask_rgb=True)

        if feature_type == 'cls':
            proposals.features = F.normalize(feature_extractor(proposals.proposals.to('cuda', dtype=torch.bfloat16), feature_type='cls', layer=layer), dim=-1)
        elif feature_type == 'ffa':
            proposal_features_raw = feature_extractor(proposals.proposals.to('cuda', dtype=torch.bfloat16), feature_type='patch', layer=layer)

            feats = []
            masks_downsampled = [cv2.resize(mask.float().cpu().numpy(), (30, 30), interpolation=cv2.INTER_AREA) > 0 for mask in proposals.proposals_masks]
            for feat, mask in zip(proposal_features_raw, masks_downsampled):
                feat_ffa = feat[mask.flatten()]
                feats.append(feat_ffa.mean(dim=0))
                
            proposals.features = F.normalize(torch.stack(feats), dim=-1)
        
        for prop_idx, feature in enumerate(proposals.features):
            scores = (retrieval_features@feature).float()

            # get top 100 scores for coarse retrieval
            scores, I = torch.topk(scores, 100)

            if args.topk == 0:
                # take the first mesh and score
                proposals.meshes.append(filelist[I[0].item()])
                proposals.scores.append(scores[0].item())
            else:
                # do per-view fine retrieval for top 100
                scores = {}
                for i, idx in enumerate(I.cpu().numpy()):
                    finegrained_features = torch.from_numpy(np.load(f'data/datasets/{args.retrieval}/{filelist[idx]}.npy')).to('cuda', dtype=torch.bfloat16)
                    finegrained_features = F.normalize(finegrained_features, dim=-1)
                    # take top-k scores
                    pred_scores = (finegrained_features@feature).float()
                    topk_scores, topk_idx = torch.topk(pred_scores, args.topk)
                    scores[filelist[idx]] = topk_scores.cpu().numpy().mean().item()

                # take mesh with highest score from top 100
                mesh_id = max(scores, key=scores.get)
                proposals.meshes.append(mesh_id)
                proposals.scores.append(scores[mesh_id])

        bop_dict = proposals.to_bop_dict()
        out_file.extend(bop_dict)

    with open(output_file, 'w') as f:
        json.dump(out_file, f)