"""Stage 7 — supervised contrastive representation learning on the 21 KNOWN classes only.

    python scripts/train_contrastive.py                  # full run (config.yaml -> contrastive)
    python scripts/train_contrastive.py --max-batches 3  # smoke test (throw-away run name)

ImageNet-initialised ResNet-50 (NOT the Stage-3 checkpoint); layer3+layer4 trainable (same capacity as
Stage-3 phase B); MLP projection head 2048->2048->128 trained with SupCon (t = 0.07). Batches are
class-balanced (P classes x K images x 2 views). Checkpoint selection: nearest-centroid macro-F1 on the
KNOWN validation set (centroids from a fixed train subset). Unknown images are never loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ManifestDataset, build_transforms, seed_worker  # noqa: E402
from src.data.protocol import known_classes, load_manifest, verify_lock  # noqa: E402
from src.models.classifier import Classifier  # noqa: E402
from src.training.supcon import ClassBalancedBatchSampler, ProjectionHead, SupConLoss, TwoViewDataset  # noqa: E402
from src.training.trainer import warmup_cosine  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.reproducibility import environment, seed_everything  # noqa: E402


class ContrastiveNet(nn.Module):
    def __init__(self, n_classes, trainable, proj):
        super().__init__()
        self.enc = Classifier("resnet50", n_classes, pretrained=True)      # ImageNet init; its linear head is unused
        self.enc.freeze_backbone_except(trainable)
        for p in self.enc.head.parameters():
            p.requires_grad = False
        self.proj = ProjectionHead(self.enc.feature_dim, proj["hidden"], proj["out"])

    def train(self, mode=True):
        super().train(mode)
        self.enc.train(mode)                                               # keeps frozen stages' BN in eval
        return self

    def features(self, x):                                                 # EVALUATION representation (2048-d)
        return self.enc.extract_features(x)

    def forward(self, x):
        return self.proj(self.features(x))


@torch.no_grad()
def embed(model, ds, device, bs, nw):
    model.eval()
    out = []
    for x, _, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.features(x.to(device))
        out.append(F.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def val_metrics(model, ref_ds, ref_y, val_ds, val_y, device, bs, nw, n_classes):
    R, V = embed(model, ref_ds, device, bs, nw), embed(model, val_ds, device, bs, nw)
    C = np.stack([R[ref_y == c].mean(0) for c in range(n_classes)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    sim = V @ C.T
    pred = sim.argmax(1)
    own = sim[np.arange(len(V)), val_y]
    cc = C @ C.T
    return {"val_nc_accuracy": float(accuracy_score(val_y, pred)),
            "val_nearest_centroid_macro_f1": float(f1_score(val_y, pred, average="macro")),
            "val_mean_cosine_to_own_centroid": float(own.mean()),
            "centroid_mean_pairwise_cosine": float(cc[~np.eye(n_classes, dtype=bool)].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    c, b = cfg["contrastive"], cfg["baseline"]
    run = "smoke_contrastive" if a.max_batches else c["run_name"]
    verify_lock(cfg)
    classes = known_classes(cfg)
    mdir = resolve(cfg["paths"]["models"]) / run
    if (mdir / "best_model.pt").exists() and not (a.overwrite or a.max_batches):
        raise SystemExit(f"{mdir}/best_model.pt exists")
    mdir.mkdir(parents=True, exist_ok=True)
    seed_everything(c["seed"], deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rf = b["dataset"]["row_filter"]
    tr = load_manifest("data/splits/train.csv", rf, "train", classes)
    va = load_manifest("data/splits/val.csv", rf, "val", classes)
    forbidden = set(cfg["osr_protocol"]["unknown_near"]) | set(cfg["osr_protocol"]["unknown_val"])
    assert not (set(tr["class_name"]) | set(va["class_name"])) & forbidden and set(tr["role"]) == {"known"}
    aug = build_transforms(b["training"], b["augmentation"], train=True)          # documented Stage-3 augmentation
    ev = build_transforms(b["training"], b["augmentation"], train=False)
    ytr = tr["class_id"].to_numpy()
    nbatch = a.max_batches or c["batches_per_epoch"]
    sampler = ClassBalancedBatchSampler(ytr, list(range(21)), c["classes_per_batch"], c["samples_per_class"], nbatch, c["seed"])
    dl = DataLoader(TwoViewDataset(ManifestDataset(tr, None), aug, c["views"]), batch_sampler=sampler,
                    num_workers=c["num_workers"], pin_memory=True, persistent_workers=c["num_workers"] > 0,
                    worker_init_fn=seed_worker)
    rng = np.random.default_rng(c["seed"])
    ref_idx = np.concatenate([rng.choice(np.flatnonzero(ytr == k), c["val_reference_per_class"], replace=False) for k in range(21)])
    ref_ds = ManifestDataset(tr.iloc[ref_idx], ev)
    vsel = np.arange(len(va)) if not a.max_batches else rng.choice(len(va), 200, replace=False)
    val_ds = ManifestDataset(va.iloc[vsel], ev)
    val_y = va["class_id"].to_numpy()[vsel]

    model = ContrastiveNet(21, c["trainable"], c["projection"]).to(device)
    groups = [{"params": [p for p in model.enc.backbone.parameters() if p.requires_grad], "lr": c["lr_backbone"]},
              {"params": list(model.proj.parameters()), "lr": c["lr_head"]}]
    opt = torch.optim.AdamW(groups, weight_decay=c["weight_decay"])
    epochs = 1 if a.max_batches else c["epochs"]
    sched = warmup_cosine(opt, c["warmup_epochs"] * nbatch, epochs * nbatch)
    scaler = torch.amp.GradScaler(device.type, enabled=c["amp"] and device.type == "cuda")
    crit = SupConLoss(c["temperature"])
    env = {**environment(c["seed"], device), "batch_images": c["classes_per_batch"] * c["samples_per_class"],
           "views": c["views"], "effective_batch_views": c["classes_per_batch"] * c["samples_per_class"] * c["views"],
           "classes_per_batch": c["classes_per_batch"], "samples_per_class": c["samples_per_class"],
           "temperature": c["temperature"], "epochs_max": c["epochs"], "lr_backbone": c["lr_backbone"], "lr_head": c["lr_head"],
           "optimizer": "AdamW", "init": "torchvision ResNet50 IMAGENET1K_V2", "trainable": c["trainable"],
           "gradient_accumulation": "none (SupCon loss is batch-coupled; batch fits in memory)"}
    hist, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        sampler.set_epoch(ep)
        model.train()
        tot, n, te = 0.0, 0, time.time()
        for x, y, _ in tqdm(dl, desc=f"ep{ep} supcon", leave=False):
            x = x.flatten(0, 1).to(device, non_blocking=True)             # [2B,3,H,W] views of the same images
            yv = y.repeat_interleave(c["views"]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=c["amp"] and device.type == "cuda"):
                z = model(x)
            loss = crit(z.float(), yv)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        vm = val_metrics(model, ref_ds, ref_idx_y := ytr[ref_idx], val_ds, val_y, device, 64, c["num_workers"], 21)
        h = {"epoch": ep, "train_supcon_loss": tot / n, **vm, "seconds": round(time.time() - te, 1),
             "lr": [g["lr"] for g in opt.param_groups],
             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else None}
        hist.append(h)
        imp = vm["val_nearest_centroid_macro_f1"] > best + c["early_stopping"]["min_delta"]
        if imp:
            best, bad = vm["val_nearest_centroid_macro_f1"], 0
            best_state = {"epoch": ep, "state_dict": copy.deepcopy(model.state_dict()), "val": vm}
        else:
            bad += 1
        print(f"  ep{ep:02d} loss {h['train_supcon_loss']:.4f} | val NC acc {vm['val_nc_accuracy']:.4f} "
              f"macroF1 {vm['val_nearest_centroid_macro_f1']:.4f} own-cos {vm['val_mean_cosine_to_own_centroid']:.3f} "
              f"centroid-cos {vm['centroid_mean_pairwise_cosine']:.3f}{' *' if imp else ''} ({h['seconds']}s, {h['peak_gpu_gb']} GB)")
        if bad >= c["early_stopping"]["patience"]:
            print(f"  early stopping after {ep} epochs"); break
    minutes = (time.time() - t0) / 60
    torch.save({"state_dict": best_state["state_dict"], "best_epoch": best_state["epoch"], "val": best_state["val"],
                "class_names": classes, "evaluation_representation": c["evaluation_representation"],
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, mdir / "best_model.pt")
    (mdir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy(resolve("data/splits/PROTOCOL_LOCK.json"), mdir / "protocol_lock_used.json")
    mets = resolve(cfg["paths"]["metrics"])
    import hashlib
    env.update({"best_epoch": best_state["epoch"], "epochs_run": len(hist), "training_minutes": round(minutes, 1),
                "checkpoint": str((mdir / "best_model.pt").relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256((mdir / "best_model.pt").read_bytes()).hexdigest()})
    (mets / f"{run}_environment.json").write_text(json.dumps(env, indent=2))
    (mets / f"{run}_training_history.json").write_text(json.dumps({"history": hist, "best_epoch": best_state["epoch"],
                                                                   "best_val": best_state["val"], "minutes": minutes}, indent=2))
    from src.visualization.style import SERIES, apply_style, save
    import matplotlib.pyplot as plt
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ep_ = [h["epoch"] for h in hist]
    a1.plot(ep_, [h["train_supcon_loss"] for h in hist], "o-", color=SERIES[0], lw=2); a1.set_title("Training SupCon loss")
    a2.plot(ep_, [h["val_nearest_centroid_macro_f1"] for h in hist], "o-", color=SERIES[1], lw=2, label="val NC macro-F1")
    a2.plot(ep_, [h["val_nc_accuracy"] for h in hist], "s--", color=SERIES[2], lw=1.5, label="val NC accuracy")
    for ax in (a1, a2):
        ax.axvline(best_state["epoch"], color="#52514e", ls="--", lw=1); ax.set_xlabel("Epoch")
    a2.set_title("Known-validation representation quality"); a2.legend(fontsize=8)
    fig.suptitle(f"Supervised contrastive training (selected epoch {best_state['epoch']}, dashed)", fontweight="semibold")
    save(fig, resolve(cfg["paths"]["figures"]) / run / "training_curves.png")
    print(f"best epoch {best_state['epoch']}: {best_state['val']} | {minutes:.1f} min")


if __name__ == "__main__":
    main()
