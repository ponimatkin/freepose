import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.pipeline.utils import CropResizePad, mask_to_bbox


def collate_fn(batch):
    batch = [b for b in batch if b['templates'] is not None]
    if len(batch) == 0:
        return None

    templates = torch.cat([b['templates'] for b in batch])
    model_names = [b['model_name'] for b in batch]
    tar_files = [b['tar_file'] for b in batch]

    return {'templates': templates, 'model_name': model_names, 'tar_file': tar_files}

#TODO: Add speedup for loading templates
class WebTemplateDataset(Dataset):
    def __init__(
        self,
        wds_dir: str,
        filelist_path: str,
        resolution: int = 420,
        bbox_extend: int = 0,
        crop: bool = True
    ):
        self.wds_dir = Path(wds_dir).resolve()
        self.frame_index = pd.read_csv(Path(filelist_path).resolve(), dtype=str)['model_name']
        self.rgb_proposal_processor = CropResizePad(resolution, (420, 420), bbox_extend=bbox_extend)
        self.crop = crop

        # replace all underscores with empty string in frame index
        self.frame_index = self.frame_index.str.replace('_', '')

    def __len__(self):
        return len(self.frame_index)
    
    def get_template_by_name(self, model_name):
        # get idx corresponding to model_name
        idx = self.frame_index[self.frame_index == model_name].index[0]
        return self.__getitem__(idx)

    def __getitem__(self, idx: int):
        tar_file = idx // 10 # 10 meshes per tar file
        tar_path = self.wds_dir / f'shard-{tar_file:06d}.tar'
        tar_dict_path = self.wds_dir / f'shard-{tar_file:06d}.npy'
        tar = tarfile.open(tar_path.as_posix())

        if tar_dict_path.exists():
            tar_dict = np.load(tar_dict_path, allow_pickle=True).item()
        else:
            tar_dict = {m.name: m for m in tar.getmembers()}
            np.save(tar_dict_path, tar_dict, allow_pickle=True)
        model_name = self.frame_index[idx].replace('_', '')

        templates, depths, masks, bboxes = [], [], [], []
        for k in range(600):
            tar_file_rgb = tar.extractfile(tar_dict[f"{model_name}_{k}.rgb.png"])
            tar_file_depth = tar.extractfile(tar_dict[f"{model_name}_{k}.depth.png"])
            image = Image.open(io.BytesIO(tar_file_rgb.read()))
            depth = Image.open(io.BytesIO(tar_file_depth.read()))

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            depth = torch.from_numpy(np.array(depth) / 1000).float()
            mask = depth > 0

            if mask.sum() < 100:
                # if mask is too small, make it small square of 210x210
                mask[105:315, 105:315] = True
            bbox = mask_to_bbox(mask.numpy())

            templates.append(image)
            depths.append(depth)
            masks.append(mask)
            bboxes.append(bbox)

        tar.close()

        if len(templates) == 0:
            return {'templates': None, 'masks': None, 'depths': None, 'bboxes': None, 'model_name': model_name, 'tar_file': tar_path.name}

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        depths = torch.stack(depths)
        masks = torch.stack(masks)
        bboxes = torch.tensor(np.array(bboxes))
        if self.crop:
            templates = self.rgb_proposal_processor(templates, bboxes)
        intrinsic = torch.tensor([[600, 0, 210], [0, 600, 210], [0, 0, 1]]).reshape(3, 3)
        
        return {"templates": templates, "masks": masks, "depths": depths, "model_name": model_name, 'tar_file': tar_path.name,
                'intrinsic': intrinsic}

