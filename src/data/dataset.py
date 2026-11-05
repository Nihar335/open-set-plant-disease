"""Manifest-backed image dataset and transforms (no images are copied)."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.utils.config import PROJECT_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    """Rows of a manifest -> (image tensor, class_id, index). Metadata stays in `self.df`."""

    def __init__(self, df: pd.DataFrame, transform=None, check_files: bool = True):
        self.df = df.reset_index(drop=True)
        self.paths = [PROJECT_ROOT / p for p in self.df["image_path"]]
        self.labels = self.df["class_id"].astype(int).tolist()
        self.transform = transform
        if check_files:
            missing = [str(p) for p in self.paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} manifest images missing, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], i


def build_transforms(tcfg: dict, acfg: dict, train: bool):
    size, resize = tcfg["image_size"], tcfg["resize"]
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:                                   # deterministic
        return T.Compose([T.Resize(resize), T.CenterCrop(size), *norm])
    ops = []
    if acfg.get("rotation"):
        ops.append(T.RandomRotation(acfg["rotation"], fill=tuple(acfg.get("rotation_fill", [0, 0, 0]))))
    ops.append(T.RandomResizedCrop(size, scale=tuple(acfg["random_resized_crop_scale"]),
                                   ratio=tuple(acfg["random_resized_crop_ratio"]), antialias=True))
    if acfg.get("horizontal_flip"):
        ops.append(T.RandomHorizontalFlip())
    if acfg.get("vertical_flip"):
        ops.append(T.RandomVerticalFlip())
    cj = acfg.get("color_jitter")
    if cj and any(cj.values()):
        ops.append(T.ColorJitter(**cj))
    return T.Compose([*ops, *norm])


def seed_worker(worker_id: int) -> None:
    import random

    import numpy as np

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
