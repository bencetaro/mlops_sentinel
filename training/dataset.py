from pathlib import Path
from typing import List, Optional, Tuple
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2


def sorted_alphanumeric(files: List[str]) -> List[str]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        img_size: Tuple[int, int] = (201, 201),
        transforms=None,
        training: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.training = training
        self.transforms = transforms
        self.img_size = img_size

        valid_exts_img = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        valid_exts_mask = {".png", ".tif", ".tiff"}

        image_files = [
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in valid_exts_img and not p.name.endswith(".aux.xml")
        ]

        if self.masks_dir:
            mask_files = [
                p for p in self.masks_dir.iterdir()
                if p.suffix.lower() in valid_exts_mask and not p.name.endswith(".aux.xml")
            ]

            image_dict = {p.stem: p for p in image_files}
            mask_dict = {p.stem: p for p in mask_files}

            common_stems = sorted_alphanumeric(list(set(image_dict.keys()) & set(mask_dict.keys())))

            self.image_paths = [str(image_dict[s]) for s in common_stems]
            self.mask_paths = [str(mask_dict[s]) for s in common_stems]

            print(f"✅ Using {len(self.image_paths)} matched image–mask pairs.")
        else:
            self.image_paths = sorted_alphanumeric([str(p) for p in image_files])
            self.mask_paths = None
            print(f"🟢 Inference mode: found {len(self.image_paths)} images.")

        assert len(self.image_paths) > 0, "No valid images found in the provided directory!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.moveaxis(img, -1, 0).astype("float32")
        img_t = torch.tensor(img)

        if self.mask_paths:
            mask_path = self.mask_paths[idx]
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask = (mask > 0).astype("float32")
            mask = cv2.resize(mask, self.img_size)
            mask_t = torch.tensor(mask).unsqueeze(0)

            if self.transforms:
                img_t = self.transforms(img_t)
            return img_t, mask_t
        else:
            if self.transforms:
                img_t = self.transforms(img_t)
            return img_t

