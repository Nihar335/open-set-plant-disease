"""Generic supervised training / evaluation loops (AMP, per-step schedulers)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def class_weights(train_labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    """w_c ∝ (1 / n_c) ** power, normalised to mean 1. Uses TRAIN counts only."""
    counts = np.bincount(train_labels, minlength=n_classes).astype(float)
    if (counts == 0).any():
        raise ValueError("a known class has no training images")
    w = (1.0 / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, amp=True, desc="train"):
    model.train()
    tot_loss, correct, n, t0 = 0.0, 0, 0, time.time()
    for x, y, _ in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": tot_loss / n, "accuracy": correct / n, "seconds": time.time() - t0}


@torch.no_grad()
def predict(model, loader, device, amp=True, desc="eval"):
    """Return logits (float32, CPU), labels, dataset indices."""
    model.eval()
    out_l, out_y, out_i = [], [], []
    for x, y, idx in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out_l.append(logits.float().cpu()); out_y.append(y); out_i.append(idx)
    return torch.cat(out_l), torch.cat(out_y), torch.cat(out_i)


def eval_loss(logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> float:
    w = criterion.weight.cpu() if getattr(criterion, "weight", None) is not None else None
    return float(nn.functional.cross_entropy(logits, labels, weight=w,
                                             label_smoothing=getattr(criterion, "label_smoothing", 0.0)))
