import functools
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
import trimesh
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.utils.bbox_utils import bbox_iou
from utils.video_evaluation import (
    get_average_depth_errors_dt,
    get_average_proj_errors_dt,
    get_average_rot_errors_dt,
)

DATA_PATH = Path("data").resolve()

@functools.lru_cache(maxsize=None)
def load_mesh(obj_id):
    mesh_path = DATA_PATH / 'mesh_cache' / str(obj_id) / f"{obj_id}.obj"
    mesh = trimesh.load_mesh(mesh_path)
    return mesh


def pd_row_to_se3(row, t_scale=1):
    return pin.SE3(
        np.fromstring(row["R"], dtype=float, sep=" ").reshape(3, 3),
        np.fromstring(row["t"], dtype=float, sep=" ") * t_scale,
    )


def sample_mesh_points(obj_id, scale, n_points=1000):
    mesh = load_mesh(obj_id).copy()
    mesh.apply_scale(scale)
    return mesh.sample(n_points)


def load_gt(vid, ann_id):
    d = np.load(DATA_PATH / 'video_gt' / f"{vid}_poses_id{ann_id}.npy", allow_pickle=True).item()
    sym_axis = d["sym_axis"] if "sym_axis" in d else None
    gt = d["poses"]
    gt = [pin.SE3(pose) for pose in gt]
    gt_scale = 0.15
    obj_id = d["mesh_id"]
    focal_length = d["focal_length"]
    bboxes = d["bboxes"]
    return gt, gt_scale, sym_axis, obj_id, focal_length, bboxes


def load_pred_csv(filepath, obj_id=None, bbox=None):
    df = pd.read_csv(filepath)
    if obj_id is not None:
        df = df[df["obj_id"] == obj_id]

    # select the object with the highest mean IoU over all frames
    if ((isinstance(bbox, np.ndarray) and bbox.ndim == 2) or
        (isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], np.ndarray) and bbox[0].ndim == 1)):
        N = np.sum(df['im_id']==0)
        object_ious = []
        for obj_idx in range(N):
            obj_bboxes = [np.array(list(map(int, x.split(' ')))) for x in df['bbox_visib'].values[obj_idx::N]]
            ious = [bbox_iou(a,b) for a,b in zip(obj_bboxes, bbox)]
            object_ious.append(np.mean(ious))
        object_index = np.argmax(object_ious) 
        max_iou = object_ious[object_index]
        if max_iou < 0.5:
            logger.warning(f"Warning: The best object has mean IoU = {max_iou:.4f} < 0.5. Maybe the detection for the correct object is missing?")
        df = df.iloc[object_index::N].reset_index(drop=True)
    elif bbox is None:
        pass
    else:
        raise ValueError(f"Unsupported bbox type: {bbox}")

    obj_id = df["obj_id"].values[0]

    N = np.sum(df['im_id']==0)
    assert N != 0, 'Could not find any relevant object!'
    assert N == 1, 'Found multiple objects!'

    pred_scales = df["scale"].values
    assert len(np.unique(pred_scales)) == 1, 'Found different scales for different frames!'
    pred_scale = pred_scales[0]

    pred = [pd_row_to_se3(d) for _, d in df.iterrows()]
    pts = sample_mesh_points(obj_id, pred_scale)

    bbox_visib0 = df["bbox_visib"].values[0]
    
    # fix infs/NaNs
    for i in range(len(pred)):
        if not np.isfinite(pred[i].translation).all():
            logger.warning(f"Warning: Found non-finite translation at index {i} ({filepath})")
            if i == 0:
                #find first finite translation:
                is_finite = np.isfinite(np.array([x.translation for x in pred]))
                idx = np.where(np.all(is_finite, axis=1))[0][0].item()
                pred[0].translation = pred[idx].translation
            else:    
                pred[i].translation = pred[i-1].translation

        if not np.isfinite(pred[i].rotation).all():
            logger.warning(f"Warning: Found non-finite rotation at index {i} ({filepath}")
            if i == 0:
                # find first finite rotation:
                is_finite = np.isfinite(np.array([x.rotation for x in pred]))
                idx = np.where(np.all(is_finite, axis=1))[0][0].item()
                pred[0].rotation = pred[idx].rotation
            else:
                pred[i].rotation = pred[i-1].rotation

    return pred, pred_scale, obj_id, bbox_visib0, pts


video_names = [
    'bowl1',
    'bowl2',
    'bowl3',
    'bowl4',
    'bowl5',
    'bowl6',
    'bowl7',
    'campbells1',
    'campbells2',
    'campbells3',
    'campbells4',
    'cups',
    'jug',
    'juice',
    'pour_268',
    'pour_805',
    'pour_2100',
    'pour_2257',
    'pour_2866',
    'pour_4168',
    'pour_4711',
    'pour_from_7369',
    'pour_from_8021',
    'pour_from_10591',
    'pour_in_1110',
    'pour_in_10109',
    'pour_into_1771',
    'pour_into_6685',
    'pour_onto_10437',
    'pour_into_8625',
    'pour_onto_8316',
    'spoons',
]


def main(args):
    ann_id = args.ann_id
    videos = args.videos
    
    results = {x: pd.DataFrame(np.nan, index=videos, columns=args.labels) for x in ['rot', 'proj', 'depth']}

    pbar = tqdm(videos, ncols=80)
    for video in pbar:
        pbar.set_description(f"video={video}")
        frame_path = list((DATA_PATH / 'datasets' / 'videos' / video).iterdir())[0]
        image_height,image_width = np.asarray(Image.open(frame_path)).shape[:2]

        gt, gt_scale, gt_sym_axis, gt_obj_id, gt_focal_length, gt_bboxes = load_gt(video, ann_id)

        poses = []
        scales = [] 
        for i, pattern in enumerate(args.patterns):
            try:
                pred_path = DATA_PATH / 'results' / 'videos' / video / pattern.format(video=video)
                pred_poses, pred_scale, pred_obj_id, bbox_visib, pred_mesh_pts = load_pred_csv(pred_path, bbox=gt_bboxes)
            except Exception as ex:
                logger.error(f"Error, failed to load video={video}, pattern={pattern}:\n{ex}")
                poses.append(None)
                scales.append(None)
                continue

            N = len(gt)
            assert len(pred_poses) == N

            poses.append(pred_poses)
            scales.append(pred_scale)
        
        dts = np.linspace(1, len(gt) / 2, num=10, dtype=int)
        print(dts)
        for i in range(len(args.labels)):
            method = args.labels[i]
            pose = poses[i]

            if pose is None:
                continue
            scale = scales[i]
            err =  get_average_rot_errors_dt(pose, gt, dts=dts, sym_axis=gt_sym_axis)
            if not np.isfinite(err):
                logger.error(f"Error [rot]! video={video}, method={method}: {err}")
            results['rot'].loc[video, method] = np.rad2deg(err)

            err = get_average_proj_errors_dt(pose, gt, scale, gt_scale, dts=dts, w=image_width, h=image_height)
            if not np.isfinite(err):
                logger.error(f"Error [proj]! video={video}, method={method}: {err}")
            results['proj'].loc[video, method] = err

            err = get_average_depth_errors_dt(pose, gt, scale, gt_scale, dts=dts)
            if not np.isfinite(err):
                logger.error(f"Error [depth]! video={video}, method={method}: {err}")
            results['depth'].loc[video, method] = err
            

    for k,v in results.items():
        v.to_csv(str(DATA_PATH / 'results' / 'videos' / f'results_{k}.csv'))
        print(f"{k} errors")
        print(np.round(v,2))
        print()
        print("-"*80)

    mean_results = pd.DataFrame(-1.0, index=args.labels, columns=['rot', 'proj', 'depth'])
    for method in args.labels:
        for metric in ['rot', 'proj', 'depth']:
            mean_results.loc[method, metric] = np.mean(results[metric][method])
            
    mean_results.rename(columns={'proj':'proj', 'depth':'depth'}, inplace=True)
    print("Mean results:")
    print(np.round(mean_results,2))
    mean_results.to_csv(str(DATA_PATH / 'results' / 'videos' / 'results_mean.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--videos", "-v", type=str, nargs='*', default=None, help="Video names")
    parser.add_argument("--labels", "-l", type=str, nargs='*', default=None, help="Method names")
    parser.add_argument("--patterns", "-p", type=str, nargs='*', default=None, help="Output .csv filename patterns with the {video} placeholder for each method")

    parser.add_argument("--ann_id", "-i", type=int, default=1, help="Ground-truth annotation ID")

    args = parser.parse_args()

    if args.labels is None and args.patterns is None:
        args.labels = [
            "MegaPose coarse",
            "MegaPose fine",
            "GigaPose",
            "FoundPose",
            "Ours coarse",
            "Ours fine",
        ]

        args.patterns = [
            "props-ground-box-0.2-text-0.2-ffa-22-top-25_{video}_gpt4_scaled_best_object_megapose_coarse.csv",
            "props-ground-box-0.2-text-0.2-ffa-22-top-25_{video}_gpt4_scaled_best_object_megapose_coarse_ref.csv",
            "gigapose_{video}_rescaled.csv",
            "foundpose_{video}_rescaled.csv",
            "props-ground-box-0.2-text-0.2-ffa-22-top-25_{video}_gpt4_scaled_best_object_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv",
            "{video}-tracked.csv",
        ]
    assert len(args.labels) == len(args.patterns)

    if args.videos is None:
        args.videos = video_names
        logger.info(f"Running evaluation on {len(args.videos)} videos")

    main(args)
