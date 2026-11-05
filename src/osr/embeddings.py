"""Penultimate-feature extraction from the frozen classifier.

Representation = `model.extract_features(x)`: ResNet global-average-pooled layer4 output
(2048-d for ResNet-50), i.e. exactly the input of the 21-class linear head. Logits are
computed from the same features through `model.head`, so they can be checked against the
Stage-4 cached logits.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def extract(model, loader, device, fp32: bool = True, desc: str = "embed"):
    """Returns (embeddings float32 [N,D], logits float32 [N,C], dataset indices [N])."""
    model.eval()
    feats, logits, idx = [], [], []
    for x, _, i in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=(not fp32) and device.type == "cuda"):
            f = model.extract_features(x.to(device, non_blocking=True))
            z = model.head(f)
        feats.append(f.float().cpu()); logits.append(z.float().cpu()); idx.append(i)
    E, L, I = torch.cat(feats).numpy(), torch.cat(logits).numpy(), torch.cat(idx).numpy()
    if not (np.isfinite(E).all() and np.isfinite(L).all()):
        raise FloatingPointError("non-finite embeddings or logits")
    return E, L, I


def save_npz(path, embeddings, logits, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32), logits=logits.astype(np.float32),
                        image_path=np.array(meta["image_path"].tolist()))


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return z["embeddings"], z["logits"], list(z["image_path"])
