from pathlib import Path

import numpy as np
import trimesh
from loguru import logger
from PIL import Image
from tqdm import tqdm


def rescale_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh', process=False, maintain_order=True)

    vertices = mesh.vertices
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())

    scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin) / 2.0

    mesh.vertices[:, 0] -= (xmax + xmin) / 2.0
    mesh.vertices[:, 1] -= (ymax + ymin) / 2.0
    mesh.vertices[:, 2] -= (zmax + zmin) / 2.0
    mesh.vertices /= scale

    if mesh.visual.kind == 'texture' and mesh.visual.material is not None:
        material = mesh.visual.material

        if isinstance(material, trimesh.visual.material.SimpleMaterial) and material.image is not None:
            image = material.image

            if max(image.size) > 16384:
                scale = 16384 / max(image.size)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size)
                mesh.visual.material.image = np.array(image)

        # Handle PBRMaterial
        elif isinstance(material, trimesh.visual.material.PBRMaterial) and material.baseColorTexture is not None:
            image = material.baseColorTexture

            if max(image.size) > 16384:
                scale = 16384 / max(image.size)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size)
                mesh.visual.material.baseColorTexture = np.array(image)

    return mesh

if __name__ == "__main__":
    base_dataset_path = Path(__file__).parent.parent / "data" / "datasets" 
    filelist_path = Path(__file__).parent.parent / "data" / "mesh_cache.txt"

    target_path = base_dataset_path / 'models_normalized'
    target_path.mkdir(exist_ok=True, parents=True)

    with open(filelist_path, 'r') as f:
        files = f.readlines()
        files = [f.strip('\n') for f in files]

    for model in (base_dataset_path / 'objaverse_models').iterdir():
        if model.name.replace('.glb', '') not in files:
            continue
        
        try:
            output_path = target_path / model.name.replace('.glb', '')
            output_path.mkdir(exist_ok=True, parents=True)
            mesh = rescale_mesh(model)
            mesh.export((output_path / f'{model.name}.obj').as_posix())
        except Exception as e:
            logger.error(f"Failed to process {model.name}")
            continue
        

    for model in (base_dataset_path / 'google_scanned_objects' / 'models_normalized').iterdir():
        if model.name not in files:
            continue

        model_path = model / 'meshes' / 'model.obj'

        try:
            output_path = target_path / model.name
            output_path.mkdir(exist_ok=True, parents=True)
            mesh = rescale_mesh(model_path)
            mesh.export((output_path / f'{model.name}.obj').as_posix())
        except Exception as e:
            logger.error(f"Failed to process {model.name}")
            continue