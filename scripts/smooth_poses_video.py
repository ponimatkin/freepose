import itertools
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import trimesh
from loguru import logger
from PIL import Image, ImageDraw
from sam2.utils.amg import rle_to_mask
from tqdm import tqdm

from src.pipeline.estimators.tracking_refiner import TrackingRefiner
from src.pipeline.refiner_utils import smooth_transforms

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
    

def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def multiply_alpha(pil_image, alpha):
    return Image.fromarray(
        (  np.array(pil_image) * np.array([1.0, 1.0, 1.0, alpha])  ).astype(np.uint8)
    )

def create_outline(ren):
    kernel = np.ones((3, 3), np.uint8)
    mask_rgb = np.stack([np.uint8(ren[1] > 0)] * 3, 2) * 255
    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=2)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)
    canny = cv2.dilate(canny, kernel, iterations=2)
    outline = np.clip(np.stack([canny] * 3, 2).astype(np.float32) * np.array([[[0.21,0.49,0.74]]]), 0, 255.).astype(np.uint8)
    outline = np.concatenate([outline, canny[:, :, np.newaxis]], 2)
    return outline


def render(mesh, width, height, K, transform):
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.0001, zfar=9999)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[5.0, 5.0, 5.0])

    opencv2opengl = np.eye(4)
    opencv2opengl[1, 1] = -1
    opencv2opengl[2, 2] = -1

    scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), matrix=transform))
    scene.add_node(pyrender.Node(camera=camera, matrix=opencv2opengl))

    color_buff, depth_buff = renderer.render(scene)
    return color_buff, depth_buff


def predict_transforms_at_interval(frames, mesh, K, masks, track_interval, out_interval, init_index, init_transform, tracref):
    points2d, points3d = tracref.compute_2d3d_correspondences(
        mesh,
        Image.fromarray(frames[init_index]),
        K,
        init_transform,
        mask=masks[init_index]
    )

    query_points = np.pad(points2d, [(0,0), (1,0)], constant_values=init_index-track_interval[0])
    pred_tracks, pred_visibility = tracref._track_frames(frames[track_interval[0]:track_interval[1]], query_points)

    trackinfo = [init_index, out_interval, points3d, pred_tracks, pred_visibility]

    pred_transforms = predict_transforms_from_tracks(trackinfo, K)

    _from_ = out_interval[0]-track_interval[0]
    _to_ = out_interval[1]-track_interval[0]
    pred_transforms = pred_transforms[_from_:_to_]
    for i in range(2,5):
        trackinfo[i] = trackinfo[i][_from_:_to_]
    return pred_transforms, trackinfo



def predict_transforms(frames, transforms, mesh, K, masks):
    tracref = TrackingRefiner(dino_device="cuda")

    n_inliers, thr = tracref.n_inliers_per_pose(mesh, frames, K, transforms)    

    start_frame_idx = np.argmax(n_inliers)
    interval_length = 12
    #interval_padding = 0

    interval_boundaries = np.round(np.linspace(0, len(frames), len(frames)//interval_length)).astype(int)
    out_intervals = np.array(list(zip(interval_boundaries[:-1], interval_boundaries[1:])))
    
    track_intervals = out_intervals.copy()
    #track_intervals[:,0] -= interval_padding
    #track_intervals[:,1] += interval_padding
    track_intervals = np.clip(track_intervals, a_min=0, a_max=len(frames))

    start_interval_idx = np.where(np.logical_and(
        start_frame_idx >= out_intervals[:,0],
        start_frame_idx < out_intervals[:,1])
        )[0][0]
    
    computed_tracks = []
    pred_transforms = []

    # first interval
    interval_indices = [start_interval_idx]
    interval_directions = [0]
    # intervals after the first one
    interval_indices += list(range(start_interval_idx+1,track_intervals.shape[0]))
    interval_directions += [1 for _ in range(track_intervals.shape[0]-start_interval_idx-1)]
    # interval before the first one
    interval_indices += list(range(start_interval_idx))[::-1]
    interval_directions += [-1 for _ in range(start_interval_idx)]

    pred_transforms = [None for _ in range(len(interval_indices))]
    computed_tracks = [None for _ in range(len(interval_indices))]
    for i, direction in tqdm(list(zip(interval_indices, interval_directions))):
        i0,i1 = track_intervals[i]
        
        if direction == 0:
            init_transform = transforms[start_frame_idx]
            init_index = start_frame_idx
        elif direction == 1:
            init_transform = pred_transforms[i-1][-1] # last transform of the previous interval
            init_index = out_intervals[i][0]
        elif direction == -1:
            init_transform = pred_transforms[i+1][0] # first transform of the following interval
            init_index = out_intervals[i][1] - 1
        else:
            raise Exception("")

        pred_transforms_i, computed_tracks_i = predict_transforms_at_interval(
            frames=frames,
            mesh=mesh,
            K=K,
            masks=masks,
            track_interval=track_intervals[i],
            out_interval=out_intervals[i],
            init_index=init_index,
            init_transform=init_transform,
            tracref=tracref
        )

        computed_tracks[i] = computed_tracks_i
        pred_transforms[i] = pred_transforms_i

    return np.concatenate(pred_transforms, axis=0), computed_tracks


def predict_transforms_from_tracks(tracks, K):
    transforms = []
    for i in range(len(tracks[-1])):
        _, _, p3d, p2d, pvis = tracks
        p3d = p3d[pvis[i]]
        p2d = p2d[i][pvis[i]]

        if len(p2d) < 15:
            logger.warning("Warning: Small number of tracked points! Adding some of the invisible points to compute the pose.")
            _, _, p3d, p2d, pvis = tracks
            vis_mask = pvis[i].copy()
            n = 15 - np.sum(pvis[i])
            idxs = np.where(~vis_mask)[0]
            np.random.shuffle(idxs)
            vis_mask[idxs[:n]] = True

            p3d = p3d[vis_mask]
            p2d = p2d[i][vis_mask]


        _, rot, trans = cv2.solvePnP(p3d, p2d, K, np.array([]), flags=cv2.SOLVEPNP_EPNP)

        predicted_transform = np.eye(4)
        predicted_transform[:3, :3] = cv2.Rodrigues(rot)[0]
        predicted_transform[:3, 3] = trans.reshape(-1)
        transforms.append(predicted_transform)

    transforms = np.array(transforms)
    if len(transforms) == 0:
        raise Exception("Error: Got 0 poses!")
    return transforms


def create_viz_imgs(frames, transforms_all, meshes_all, K, computed_tracks_all, track_colors_all, t):
    h,w = frames[0].shape[:2]
    outline_img_all = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ren_img_all = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    track_img_all = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for transforms, mesh, computed_tracks, track_colors in zip(transforms_all, meshes_all, computed_tracks_all, track_colors_all):

        r = render(mesh, w, h, K, transforms[t])
        outline_img = Image.fromarray(create_outline(r))
        ren_img = Image.fromarray(np.concatenate([r[0], np.uint8(r[1] > 0)[:, :, np.newaxis] * 255], 2))
        track_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # process points from all active tracks
        for track_idx, tracks in enumerate(computed_tracks):
            query_index, frame_interval, _, p2d, pvis = tracks

            # skip tracks that are not active at the current frame
            if t < frame_interval[0] or t >= frame_interval[1]:
                continue

            n_points = p2d.shape[1]
            colors = track_colors[track_idx]
            t_rel = t - frame_interval[0]
            points_current = p2d[t_rel]
            if t_rel == 0:
                points_prev = points_current + 1
            else:
                points_prev = p2d[t_rel-1]

            for i in list(range(n_points))[::4]:
                coord_y = (int(points_prev[i, 0]), int(points_prev[i, 1]))
                coord_x = (int(points_current[i, 0]), int(points_current[i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    track_img = draw_line(track_img, coord_y, coord_x, colors[i], 3)

        ren_img_all.paste(ren_img, (0, 0), mask=ren_img)
        outline_img_all.paste(outline_img, (0, 0), mask=outline_img)
        track_img_all.paste(track_img, (0, 0), mask=track_img)

    return ren_img_all, outline_img_all, track_img_all


def visualize(frames, transforms_all,  meshes_all, K, computed_tracks_all, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    track_colors_all = []
    for computed_tracks in computed_tracks_all:
        track_colors = []
        cmap = matplotlib.colormaps["gist_rainbow"]
        for tracks_idx, tracks in enumerate(computed_tracks):
            _, _, _, p2d, _ = tracks
            n_points = p2d.shape[1]
            track_colors.append(
                (cmap(np.linspace(0, 1+1e-10, n_points))*255).astype(np.uint8)[::-1 if tracks_idx % 2 == 1 else 1]
            )
        track_colors_all.append(track_colors)
                        
    h, w, _ = frames[0].shape

    track_img_history = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for t in tqdm(range(len(frames))):
        frame_img = Image.fromarray(frames[t])

        ren_img, outline_img, track_img = create_viz_imgs(frames, transforms_all, meshes_all, K, computed_tracks_all, track_colors_all, t)
        
        track_img_history = multiply_alpha(track_img_history, 0.66)
        track_img_history.paste(track_img, (0, 0), mask=track_img)
        ren_img.putalpha(140)

        frame_img.paste(ren_img, (0, 0), mask=ren_img)
        frame_img.paste(outline_img, (0, 0), mask=outline_img)
        frame_img.paste(track_img_history, (0, 0), mask=track_img_history)

        frame_img.save(output_path / f"{t:06d}.png")


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
    data_dir = Path("data")
    frames_dir = data_dir / "datasets" / "videos" / args.video
    results_dir = data_dir / "results" / "videos" / args.video

    csv_path = results_dir / args.poses
    frame_paths = sorted(frames_dir.iterdir())
    logger.info(f"Processing video {args.video} with {len(frame_paths)} frames!")
    
    K_file = results_dir / "K.txt"
    if K_file.exists():
        K = np.loadtxt(results_dir / "K.txt")
    else:
        # load first frame, get its size and create K matrix
        frame = Image.open(frame_paths[0])
        w, h = frame.size
        f = np.sqrt(w**2 + h**2)
        K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]])
        

    df_all = pd.read_csv(csv_path)
    n_objects = len(list(itertools.takewhile(  lambda x: x == df_all.iloc[0]['im_id'], df_all['im_id']  )))

    if args.obj_idxs is None:
        args.obj_idxs = list(range(n_objects))

    with open(results_dir / args.proposals) as f:
        proposals_all = json.load(f)



    pred_transforms_all = []
    meshes_all = []
    computed_tracks_all = []
    out_dfs = []
    for obj_idx in args.obj_idxs:
        frames, transforms, scale = [], [], None
        assert obj_idx < n_objects
        df = df_all.iloc[list(range(len(df_all)))[obj_idx::n_objects]]
        assert len(frame_paths) == len(df)
        proposals = proposals_all[obj_idx::n_objects]
        assert len(frame_paths) == len(proposals)
        masks = [rle_to_mask(p["segmentation"]) for p in proposals]
        del proposals

        for idx in range(len(df)):
            if idx == 0:
                scale = df.iloc[idx]["scale"]
            assert scale == df.iloc[idx]["scale"]

            _RT_ = np.eye(4)
            _RT_[:3,:3] = np.array([float(x) for x in df.iloc[idx]["R"].split(" ")]).reshape(3, 3)
            _RT_[:3, 3] = np.array([float(x) for x in df.iloc[idx]["t"].split(" ")])
            transforms.append(_RT_)
            frames.append(np.array(Image.open(frame_paths[idx])))
        

        skip_n = 1
        frames, transforms = np.stack(frames[::skip_n]), np.stack(transforms[::skip_n])
        masks = masks[::skip_n]

        assert len(masks) == len(frames) == len(transforms)

        mesh_id = df.iloc[0]["obj_id"]
        mesh_path = data_dir/ "mesh_cache" / mesh_id / f"{mesh_id}.obj"
        mesh = trimesh.load_mesh(mesh_path)
        mesh = fix_mesh_texture(mesh)
        mesh.vertices = mesh.vertices * scale
        
        pred_transforms, computed_tracks = predict_transforms(frames, transforms, mesh, K, masks=masks)
        pred_transforms[:,:,3] = transforms[:,:,3] # use coarse translation
        pred_transforms = smooth_transforms(pred_transforms)

        df_out = df.iloc[list(range(len(df)))[::skip_n]]
        R = pred_transforms[:,:3,:3].reshape(-1,9)
        df_out["R"] = [' '.join(map(str,x)) for x in R]
        t = pred_transforms[:,:3,3]
        df_out["t"] = [' '.join(map(str,x)) for x in t]
        out_dfs.append(df_out)

        meshes_all.append(mesh) 
        pred_transforms_all.append(pred_transforms)
        computed_tracks_all.append(computed_tracks)

    for i,df in enumerate(out_dfs):
        df.index = df.index*n_objects + i
    df_all = pd.concat(out_dfs).sort_index()
    df_all.to_csv(results_dir/f"{args.video}-tracked.csv", index=False)
    
    if args.vis:
        visualize(frames, pred_transforms_all, meshes_all, K, computed_tracks_all, output_path=results_dir / "viz_tracked")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--obj-idxs", type=int, default=None, nargs="+")
    parser.add_argument("--poses", type=str, default=None)
    parser.add_argument("--proposals", type=str, default=None)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    if args.poses is None and args.proposals is None:
        args.poses   =   f"props-ground-box-0.2-text-0.2-ffa-22-top-25_{args.video}_gpt4_scaled_best_object_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv"
        args.proposals = f"props-ground-box-0.2-text-0.2-ffa-22-top-25_{args.video}_gpt4_scaled_best_object.json"

    main(args)
