import os
import numpy as np
from src.pipeline.retrieval.renderer import MeshRenderer

import webdataset as wds
from pathlib import Path
import trimesh
from loguru import logger
import argparse

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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--filelist", type=str, default='./data/mesh_cache.txt')
    args.add_argument("--shards_folder", type=str, default='objaverse_shards')
    args.add_argument("--offset", type=int, default=0)
    args = args.parse_args()

    shards_path = Path('./data/datasets').resolve() / args.shards_folder
    shards_path.mkdir(exist_ok=True)

    with open(args.filelist, 'r') as f:
        mesh_ids = f.read().splitlines()

    # Get SLURM task id
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) + args.offset
    start = job_id * 10
    end = (job_id + 1) * 10

    # Select meshes to render
    meshes = mesh_ids[start:end]

    # Create webdataset
    max_count = 600*10*2 # 800 images per object, 10 objects, 2 modalities (rgb, depth)
    shard_path = shards_path / "shard-%06d.tar"
    shard_writer = wds.ShardWriter(str(shard_path), maxcount=max_count, start_shard=job_id, maxsize=10e9)

    renderer = MeshRenderer(600)

    for idx, mesh_id in enumerate(meshes):
        logger.info(f'Rendering mesh {mesh_id} ({idx+1}/{len(meshes)})')
        mesh_path = Path('data/mesh_cache').resolve() / mesh_id / f'{mesh_id}.obj'
        mesh = trimesh.load(mesh_path)

        mesh = fix_mesh_texture(mesh)
        mesh.apply_scale(0.25)

        # Render mesh
        results = renderer.render(mesh, cull_faces=False)

        # Save results to webdataset
        for i, (rgb, depth, _) in enumerate(results):
            shard_writer.write({
                '__key__': f'{mesh_id.replace("_", "")}_{i}',
                'rgb.png': rgb.astype(np.uint8),
                'depth.png': (depth * 1000).astype(np.uint16),
            })

    # Close shard
    shard_writer.close()
