import argparse
import functools
import json
import os
from itertools import takewhile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import trimesh
from loguru import logger
from matplotlib import pyplot as plt
from sam2.utils.amg import rle_to_mask
from tqdm import tqdm

from src.dataloader.template import WebTemplateDataset
from src.pipeline.estimators.online_pose_estimator import DinoOnlinePoseEstimator
from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.pipeline.utils import Proposals

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.environ['PYOPENGL_PLATFORM'] = 'egl' 


def fix_mesh_texture(mesh):
    has_visual = hasattr(mesh, 'visual')
    has_material = has_visual and hasattr(mesh.visual, 'material')
    has_image = has_material and hasattr(mesh.visual.material, 'image')

    if has_image:
        if mesh.visual.material.image is not None:
            if mesh.visual.material.image.mode == 'LA':
                mesh.visual.material.image = mesh.visual.material.image.convert('RGBA')
            elif mesh.visual.material.image.mode == '1':
                mesh.visual.material.image = mesh.visual.material.image.convert('RGB')

    return mesh

def main(args):
    video_dir = (Path("data") / "datasets" / "videos" / args.video).resolve()
    frame_names = sorted([
        p for p in video_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg"]
    ])

    results_dir = (Path("data") / "results" / "videos" / args.video).resolve()
    proposals_path = results_dir / args.proposals

    pose_outputs = results_dir / args.proposals.replace('.json', f'_dinopose_layer_{args.layer}_bbext_{args.bbox_extend}_depth_{args.depth_method}.csv') #_cache_{args.cache_size}')
    
    templates = WebTemplateDataset('data/datasets/objaverse_shards', 'data/mesh_cache.csv', bbox_extend=args.bbox_extend)
    templates.get_template_by_name = functools.lru_cache(maxsize=args.template_cache_size)(
        templates.get_template_by_name)

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 0)
    cache_dir = Path('data')  / ('cache_' + str(SLURM_JOB_ID) + '_' + args.video)
    if args.no_rescore:
        model = DinoPoseEstimator(n_poses=600, cache_size=args.cache_size, save_all=args.save_all_cache, cache_dir=cache_dir).to(device, dtype=torch.bfloat16)
    else:
        model = DinoOnlinePoseEstimator(n_coarse_poses=600, n_fine_poses=20000, cache_size=args.cache_size, save_all=args.save_all_cache, cache_dir=cache_dir).to(device, dtype=torch.bfloat16)

    with open(proposals_path, 'r') as f:
        props = json.load(f)
    n_objects = len(list(takewhile(lambda x: x['image_id'] == 0, props)))
    n_frames = len(frame_names)
    assert n_objects * n_frames == len(props)
    
    # Merge all frames for each frame
    props = [props[i:i+n_objects] for i in range(0, len(props), n_objects)]

    
    if args.depth_method == 'const-0.05':
        scales = [0.05]*len(n_objects)
    elif args.depth_method == 'const-0.1':
        scales = [0.1]*len(n_objects)
    elif args.depth_method == 'const-0.15':
        scales = [0.15]*len(n_objects)
    elif args.depth_method == 'zoedepth':
        scales = [props[0][obj_idx]['scale'] for obj_idx in range(n_objects)]
        for i in range(n_objects):
            assert all(props[frame_idx][i]['scale'] == scales[i] for frame_idx in range(n_frames)), f"Object {i} has different scales for different frames!"
    else:
        raise NotImplementedError()
    
    # Load meshe for each object
    logger.info("Loading meshes")
    mesh_ids = []
    meshes = []
    for i in range(n_objects):
        mesh_id = props[0][i]['mesh']
        assert all(props[frame_idx][i]['mesh'] == mesh_id for frame_idx in range(n_frames)), f"Object {i} has different meshes for different frames!"

        mesh_path = Path('data').resolve() / 'mesh_cache' / mesh_id / f"{mesh_id}.obj"
        mesh = fix_mesh_texture(trimesh.load(str(mesh_path), force='mesh'))
        meshes.append(mesh)
        mesh_ids.append(mesh_id)
        del mesh

    results_dict = {
        "scene_id": [],
        "im_id": [],
        "obj_id": [],
        "score": [],
        "R": [],
        "t": [],
        'bbox_visib': [],
        'scale': [],
        "time": [],
    }

    img = cv2.cvtColor(cv2.imread(str(frame_names[0])), cv2.COLOR_BGR2RGB).astype(np.uint8)
    img_h, img_w, _ = img.shape
    f = np.sqrt(img_h**2 + img_w**2)
    K = np.array([[f, 0, img_w/2.0], [0, f, img_h/2.0], [0, 0, 1]]).astype(float)
    #np.savetxt(str(results_dir / "K.txt"), K)

    logger.info("Starting inference")
    prev_poses = [None for _ in range(n_objects)]
    all_scores = [[] for _ in range(n_objects)]
    for frame_idx, frame_name in enumerate(tqdm(frame_names, ncols=100)):
        scene_proposals = props[frame_idx]
        assert all(p['image_id'] == frame_idx for p in scene_proposals)

        img = cv2.cvtColor(cv2.imread(str(frame_name)), cv2.COLOR_BGR2RGB).astype(np.uint8)
        masks = [rle_to_mask(prop['segmentation']) for prop in scene_proposals]
        boxes = [np.array(prop['bbox']) for prop in scene_proposals]
        scores = [prop['score'] for prop in scene_proposals]


        masks = torch.from_numpy(np.stack(masks))
        boxes = torch.from_numpy(np.stack(boxes))
        # convert boxes from xywh to xyxy
        boxes[:, 2:] += boxes[:, :2]
        out = {'boxes': boxes, 'masks': masks}
        proposals = Proposals(img, out, 420, bbox_extend=args.bbox_extend)
        proposals.scores = scores
        proposals.meshes = meshes


        for obj_idx, (prop, prop_mask) in enumerate(zip(proposals.proposals, proposals.proposals_masks)):
            mesh = meshes[obj_idx]
            scale = scales[obj_idx]
            box = boxes[obj_idx]
            mesh_id = mesh_ids[obj_idx]
            mesh_entry = templates.get_template_by_name(mesh_id)

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                if args.no_rescore:
                    out = model(prop, mesh_entry, K, box,  scale,  layer=args.layer, batch_size=args.batch_size)
                else:
                    out = model(prop, prop_mask, mesh_entry, mesh, K, box, scale, prev_pose=prev_poses[obj_idx], neighborhood=15, layer=args.layer, batch_size=args.batch_size)
                    prev_poses[obj_idx] = out['TCO'][0]
                
            if args.no_rescore:
                all_scores[obj_idx].append(out['all_scores'])

            results_dict["scene_id"].append(0)
            results_dict["im_id"].append(int(frame_idx))
            results_dict["obj_id"].append(mesh_id)
            results_dict["score"].append(out['scores'][0])
            R = out['TCO'][0][:3, :3].flatten().tolist()
            t = out['TCO'][0][:3, 3].tolist()

            bbox = out['bbox'].cpu().numpy()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            results_dict["R"].append(" ".join([str(x) for x in R]))
            results_dict["t"].append(" ".join([str(x) for x in t]))
            results_dict["bbox_visib"].append(" ".join([str(x) for x in bbox]))
            results_dict["scale"].append(scales[obj_idx])
            results_dict["time"].append(-1)

    if args.no_rescore:
        all_scores = np.stack(all_scores)
        np.save(results_dir / 'all_scores.npy', all_scores)
        np.save(results_dir / 'all_poses.npy', model.mesh_poses)
    df = pd.DataFrame(results_dict)
    df.to_csv(pose_outputs, index=False, header=True)

    if args.viz:
        viz_dir = results_dir / 'viz_pose'
        logger.info(f"Vizualizing poses in directory {viz_dir}")
        viz_dir.mkdir(exist_ok=True, parents=True)

        cmap = matplotlib.colormaps['Spectral']
        vertices = dict()
        vertices_colors = dict()

        for idx, (mesh, mesh_id) in enumerate(zip(meshes, mesh_ids)):
            vs = trimesh.sample.sample_surface(mesh, 7500)[0] * scales[idx]
            vertices[str(mesh_id)] = vs

            a = vs.T[0]
            a = a - np.min(a)
            a = a/np.max(a)
            colors = cmap(a)
            vertices_colors[mesh_id] = colors

        for frame_idx, frame_name in enumerate(tqdm(frame_names, ncols=100)):
            img = cv2.cvtColor(cv2.imread(str(frame_name)), cv2.COLOR_BGR2RGB).astype(np.uint8)
            fig = plt.figure(frameon=False, figsize=(img.shape[1]//100, img.shape[0]//100))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.imshow(img)
            
            rows = df[df.im_id == frame_idx]
            for row in rows.iloc:
                R = np.fromstring(row.R, sep=' ').reshape(3, 3)
                t = np.fromstring(row.t, sep=' ').reshape(3, 1)
                mesh_id = row.obj_id

                vs = (R @ (vertices[mesh_id].T) + t)
                proj = K @ vs
                proj = proj[:2]/proj[2]

                ax.scatter(*proj, s=1, alpha=0.4, color=vertices_colors[mesh_id]) #c='cyan')

            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            plt.savefig(viz_dir / f"{str(frame_idx).zfill(6)}.jpg",  bbox_inches='tight', pad_inches=0)
            plt.close()
            
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--video', type=str)
    args.add_argument('--proposals', type=str)
    args.add_argument('--layer', type=int, default=22)
    args.add_argument('--depth_method', type=str, default='zoedepth')
    args.add_argument('--bbox_extend', type=float, default=0.05)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument("--template_cache_size", type=int, default=21)
    args.add_argument("--viz", action='store_true')
    args.add_argument("--no_rescore", action='store_true')

    args.add_argument('--cache_size', type=int, default=50)
    args.add_argument('--save_all_cache', action='store_true')
    
    args = args.parse_args()

    main(args)