import numpy as np
import torch
from PIL import Image

from src.dataloader.base_bop import BOPDatasetBase


class BOPDataset(BOPDatasetBase):
    def __init__(
        self,
        root_dir: str,
        split: str,
        use_visible_masks: bool = True,
        **kwargs,
    ):
        super().__init__(root_dir, split, **kwargs)
        self.use_visible_masks = use_visible_masks

    def __getitem__(self, idx):
        rgb_path = self.meta_data.iloc[idx]["rgb_path"]
        depth_path = self.meta_data.iloc[idx]["depth_path"]
        depth_path_pred = self.meta_data.iloc[idx]["depth_pred_path"]
        scene_id = self.meta_data.iloc[idx]["scene_id"]
        frame_id = self.meta_data.iloc[idx]["frame_id"]
        image = Image.open(rgb_path)
        image = np.asarray(image.convert('RGB')).copy()

        depth = Image.open(depth_path)
        depth = np.asarray(depth).copy()
        depth = (depth*0.1)/1000
        
        depth_pred = Image.open(depth_path_pred)
        depth_pred = np.asarray(depth_pred).copy()
        depth_pred = depth_pred/(2**16-1)

        if self.use_visible_masks:
            masks_path = self.meta_data.iloc[idx]["mask_path_visib"]
        else:
            masks_path = self.meta_data.iloc[idx]["mask_path"]

        masks = []
        boxes = []
        for mask in masks_path:
            mask = Image.open(mask)
            bbox = mask.getbbox()
            if bbox is None:
                continue
            bbox = torch.tensor(bbox).int()
            mask = torch.from_numpy(np.array(mask) / 255).float()
            masks.append(mask)
            boxes.append(bbox)

        masks = torch.stack(masks)
        boxes = torch.stack(boxes)
        intrinsic = np.asarray(self.meta_data.iloc[idx]["intrinsic"]).reshape(3, 3).astype(np.float32)
        return dict(
            image=image,
            depth=depth,
            depth_pred=depth_pred,
            scene_id=scene_id,
            frame_id=frame_id,
            masks=masks,
            boxes=boxes,
            intrinsic=intrinsic,
        )