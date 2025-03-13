import json
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class BOPDatasetBase(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        **kwargs,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """

        if 'tless' in path:
            split = f'{split}_primesense'
        
        if 'hb' in path:
            split = f'{split}_primesense'

        self.path = Path(path).resolve()
        self.split = split
        self.scenes_list = self.load_list_scene()

        if (self.path / f'{self.split}_metadata.json').exists():
            self.meta_data = pd.read_json(self.path / f'{self.split}_metadata.json')
        else:
            self.meta_data = self.generate_metadata()


    def load_list_scene(self) -> List:
        split_folder = self.path / self.split
        return sorted(split_folder.iterdir())

    # TODO: refactor this function
    def generate_metadata(self):
        meta_data = {
            "scene_id": [],
            "frame_id": [],
            "rgb_path": [],
            "mask_path": [],
            "mask_path_visib": [],
            "depth_path": [],
            "depth_pred_path": [],
            "intrinsic": [],
            "obj_id": []
        }
        
        for scene_path in self.scenes_list:
            logger.info(f'Working on scene {scene_path.name}')
            with (scene_path / 'scene_camera.json').open() as f:
                scene_camera = json.load(f)

            with (scene_path / 'scene_gt.json').open() as f:
                scene_gt = json.load(f)
            
            scene_id = scene_path.name

            if (scene_path / 'rgb').exists():
                rgb_paths = list(sorted(scene_path.glob("rgb/*.[pj][pn][g]")))
                depth_paths = list(sorted(scene_path.glob("depth/*.[pj][pn][g]")))
                depth_pred_paths = list(sorted(scene_path.glob("depth_pred/*.[pj][pn][g]")))
            else:
                rgb_paths = list(sorted(scene_path.glob("rgb/*.tif")))
                depth_paths = list(sorted(scene_path.glob("depth/*.tif")))
                depth_pred_paths = list(sorted(scene_path.glob("depth_pred/*.[pj][pn][g]")))
            
            for idx_frame in tqdm(range(len(rgb_paths)), total=len(rgb_paths)):
                # get rgb path
                rgb_path = rgb_paths[idx_frame]
                depth_path = depth_paths[idx_frame]
                depth_pred_path = depth_pred_paths[idx_frame]
                # get id frame
                id_frame = int(str(rgb_path).split("/")[-1].split(".")[0])

                masks_visib = list(sorted(scene_path.glob(f"mask_visib/{id_frame:06}_*.[pj][pn][g]")))
                masks = list(sorted(scene_path.glob(f"mask/{id_frame:06}_*.[pj][pn][g]")))
                obj_ids = [int(entry['obj_id']) for entry in scene_gt[f"{id_frame}"]]

                meta_data["scene_id"].append(scene_id)
                meta_data["frame_id"].append(id_frame)
                meta_data["rgb_path"].append(str(rgb_path))
                meta_data["depth_path"].append(str(depth_path))
                meta_data["depth_pred_path"].append(str(depth_pred_path))
                meta_data["mask_path"].append([str(x) for x in masks])
                meta_data["mask_path_visib"].append([str(x) for x in masks_visib])
                meta_data["intrinsic"].append(scene_camera[f"{id_frame}"]["cam_K"])
                meta_data["obj_id"].append(obj_ids)

        with open(self.path / f'{self.split}_metadata.json', 'w', encoding='utf-8') as f:
           json.dump(meta_data, f)

        meta_data = pd.DataFrame.from_dict(meta_data)
        return meta_data


    def __len__(self):
        return len(self.meta_data)
