import json
from argparse import ArgumentParser
from itertools import takewhile
from pathlib import Path

import numpy as np
from src.utils.bbox_utils import bbox_iou

DATA_PATH = Path("data").resolve()


def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_gt_boxes(vid, ann_id):
    d = np.load(DATA_PATH / 'video_gt' / f"{vid}_poses_id{ann_id}.npy", allow_pickle=True).item()
    bboxes = d["bboxes"]
    return bboxes


def main(args):
    gt_bboxes = load_gt_boxes(args.video, args.ann_id)
    proposals_path = DATA_PATH / "results" / "videos" / args.video / args.proposals
    proposals = read_json(proposals_path)
    
    N = len(list(takewhile(lambda x: x['image_id']==0, proposals)))
    object_proposals = [proposals[i::N] for i in range(N)]
    
    # measure the iou of the bboxes for each object:
    object_ious = []
    for i in range(N):
        object_bboxes = [x['bbox'] for x in object_proposals[i]]
        iou = np.mean([bbox_iou(a, b) for a,b in zip(gt_bboxes, object_bboxes)]).item()
        object_ious.append(iou)

    idx = np.argmax(object_ious)
    iou = object_ious[idx]

    if iou < 0.5:
        print("Warning: The best object ({idx}) has IoU={iou} < 0.5. Maybe the detection for the correct object is missing?")
    print(f"Best object: {idx} with IoU: {iou}")

    # save the best object to json
    best_object = object_proposals[idx]
    best_object_path = proposals_path.with_name(proposals_path.stem + "_best_object.json")
    with open(best_object_path, "w") as f:
        json.dump(best_object, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--proposals", type=str, required=True)
    parser.add_argument("--ann_id", type=int, default=1)
    args = parser.parse_args()

    main(args)