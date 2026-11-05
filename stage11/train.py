"""Train the Stage 11 multi-task model (disease classification + symptom detection).

Loss = CE(disease, label smoothing 0.1) + 0.5 * BCE(symptom logits, weak symptom profile of the disease).
Field photos (PlantDoc) are over-sampled 4x so the model does not only learn laboratory backgrounds.
Model selection uses validation only: mean of PlantVillage-val (2,500 sampled), PlantDoc-val and apple-field-val accuracy.

Run:  python -P stage11/train.py [--epochs 10]
Outputs: models/stage11/best.pt, stage11/results/train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF, IMG_SIZE, MEAN, STD, LeafNet  # noqa: E402

MODEL_DIR = ROOT / "models" / "stage11"
RESULTS = ROOT / "stage11" / "results"

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.35, 0.35, 0.3, 0.04),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])


class ImageSet(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths = [str(ROOT / p) for p in df.path]
        self.labels = df.label.to_numpy()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float((p == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples-per-epoch", type=int, default=44000)
    ap.add_argument("--workers", type=int, default=2)  # each Windows worker commits ~1.5 GB (torch DLLs)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    # channels_last was 5x slower on this RTX 4050 / torch 2.5.1 Windows setup (37 vs 207 img/s), so it is not used.
    torch.backends.cudnn.benchmark = True
    kb = load_kb()
    sym_targets = torch.tensor(kb.symptom_targets(), device=device)  # (C, S)

    idx = pd.read_csv(RESULTS / "image_index.csv")
    tr = idx[idx.split == "train"].reset_index(drop=True)
    counts = tr.label.value_counts()
    w = 1.0 / np.sqrt(tr.label.map(counts).to_numpy())
    w = w * np.where(tr.source == "plantdoc", 4.0, 1.0)  # scarce web/field photos of 13 crops
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), args.samples_per_epoch, replacement=True)

    # Keep the process count low: on Windows every DataLoader worker re-imports torch and commits ~1.5-2 GB of
    # virtual memory. 10 train + 8 val workers exhausted the commit limit and crashed other apps (2026-10-22).
    train_dl = DataLoader(ImageSet(tr, TRAIN_TF), batch_size=args.batch, sampler=sampler, drop_last=True,
                          num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_pv_df = idx[(idx.split == "val") & (idx.source == "plantvillage")].sample(2500, random_state=0)
    val_pv = DataLoader(ImageSet(val_pv_df, EVAL_TF), batch_size=64, num_workers=0)
    val_apple = DataLoader(ImageSet(idx[(idx.split == "val") & idx.source.isin(["appleleaf9", "kashmir_apple"])],
                                    EVAL_TF), batch_size=64, num_workers=0)
    val_pd = DataLoader(ImageSet(idx[(idx.split == "val") & (idx.source == "plantdoc")], EVAL_TF),
                        batch_size=64, num_workers=0)

    model = LeafNet(len(kb.model_classes), len(kb.symptom_ids)).to(device)
    head_params = list(model.disease_head.parameters()) + list(model.symptom_head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 3e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    steps_per_epoch = args.samples_per_epoch // args.batch
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch // 2
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log = open(RESULTS / "train_log.jsonl", "a", encoding="utf-8")
    best, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, run_acc, n = time.time(), 0.0, 0.0, 0
        for x, y in train_dl:
            frac = step / total
            scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * frac))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, sym = model(x)
                loss_d = F.cross_entropy(logits, y, label_smoothing=0.1)
                loss_s = F.binary_cross_entropy_with_logits(sym.float(), sym_targets[y])
                loss = loss_d + 0.5 * loss_s
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            run_loss += loss.item() * len(y)
            run_acc += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        acc_pv = evaluate(model, val_pv, device)
        acc_pd = evaluate(model, val_pd, device)
        acc_apple = evaluate(model, val_apple, device)
        score = (acc_pv + acc_pd + acc_apple) / 3
        rec = {"epoch": epoch, "train_loss": run_loss / n, "train_acc": run_acc / n, "val_pv_acc": acc_pv,
               "val_plantdoc_acc": acc_pd, "val_apple_field_acc": acc_apple, "val_score": score, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        ckpt = {"state_dict": model.state_dict(), "classes": [d["key"] for d in kb.model_classes],
                "symptoms": kb.symptom_ids, "epoch": epoch, "val": rec}
        torch.save(ckpt, MODEL_DIR / "last.pt")
        if score > best:
            best = score
            torch.save(ckpt, MODEL_DIR / "best.pt")
            print(f"  new best {score:.4f}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
