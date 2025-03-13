import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

DATA_DIR = Path('data')

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

def load_mesh(path, scale=1.0):
    mesh = trimesh.load_mesh(path)
    mesh = fix_mesh_texture(mesh)
    mesh.apply_scale(scale)
    return mesh


def create_outline(mask, color=[0.21,0.49,0.74]):
    kernel = np.ones((3, 3), np.uint8)
    mask_rgb = np.stack([np.uint8(mask)] * 3, 2) * 255
    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=2)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)
    canny = cv2.dilate(canny, kernel, iterations=2)
    outline = np.clip(np.stack([canny] * 3, 2).astype(np.float32) * np.array([[color]]), 0, 255.).astype(np.uint8)
    outline = np.concatenate([outline, canny[:, :, np.newaxis]], 2)
    return outline


def main(args):
    pred_path = DATA_DIR / 'results' / 'videos' / args.video / args.predictions
    pred = pd.read_csv(pred_path)
    viz_dir = pred_path.parent / f'viz_{pred_path.stem}'
    viz_dir.mkdir(exist_ok=True, parents=True)

    N = (pred['im_id'] == 0).sum().item()
    objects_pred = [pred.iloc[i::N] for i in range(N)]
    mesh_ids = [x.iloc[0]['obj_id'] for x in objects_pred]
    scales = [x.iloc[0]['scale'].item() for x in objects_pred]
    meshes = []
    for mesh_id, scale in zip(mesh_ids, scales):
        mesh = load_mesh(DATA_DIR / 'mesh_cache' / mesh_id / f'{mesh_id}.obj', scale)
        meshes.append(mesh)

    video_path = DATA_DIR / 'datasets' / 'videos' / args.video
    frame_paths = sorted(video_path.iterdir())

    h,w,_ = np.array(Image.open(frame_paths[0])).shape

    f = np.sqrt(h**2 + w**2)
    cx = w/2
    cy = h/2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    for frame_idx, frame_path in tqdm(enumerate(frame_paths), ncols=100, total=len(frame_paths)):
        frame = Image.open(frame_path)

        # Extract poses
        Ts = []
        for object_idx, mesh in enumerate(meshes):
            pred = objects_pred[object_idx].iloc[frame_idx]

            R = np.array([float(x) for x in pred['R'].split(' ')]).reshape(3,3)
            t = np.array([float(x) for x in pred['t'].split(' ')])
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t
            Ts.append(T)
        
        # Sort object by distance to camera - render further objects first
        distances = [np.linalg.norm(T[:3,3]) for T in Ts]
        order = np.argsort(distances)[::-1]

        # Render and overlay
        for i in order:
            T = Ts[i]
            mesh = meshes[i]

            color, depth = render(mesh, w, h, K, T)
            mask = (depth>0)
            outline = Image.fromarray(create_outline(mask))
            ren_img = Image.fromarray(np.concatenate([color, np.uint8(mask)[:, :, np.newaxis] * 255], 2))

            frame.paste(ren_img, (0, 0), ren_img)
            frame.paste(outline, (0, 0), outline)
            
        frame.save(viz_dir / f'{frame_idx:06d}.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--predictions", '-p', type=str, required=True)

    args = parser.parse_args()
    main(args)
