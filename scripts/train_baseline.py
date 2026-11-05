"""Stage 3 — train the closed-set baseline on the locked manifests (known classes only).

    python scripts/train_baseline.py                    # uses config.yaml -> baseline
    python scripts/train_baseline.py --run-name X       # a new run without touching previous outputs
    python scripts/train_baseline.py --max-batches 5    # smoke test (writes to a throw-away run name)

Phase A: backbone frozen (incl. BatchNorm statistics), head trained.
Phase B: layer3 + layer4 + head fine-tuned with discriminative learning rates, warmup + cosine,
         early stopping on known-class validation macro-F1. The best checkpoint over A and B is kept.
Validation uses ONLY known-class rows; unknown classes are never loaded.
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
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ManifestDataset, build_transforms, seed_worker  # noqa: E402
from src.data.protocol import known_classes, load_manifest, verify_lock  # noqa: E402
from src.evaluation.classification import summary_metrics  # noqa: E402
from src.models.classifier import Classifier  # noqa: E402
from src.training.trainer import class_weights, eval_loss, predict, train_one_epoch, warmup_cosine  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.reproducibility import environment, seed_everything  # noqa: E402


def loaders(cfg, classes, max_batches=None):
    b = cfg["baseline"]
    d, t, a = b["dataset"], b["training"], b["augmentation"]
    tr = load_manifest(d["train_manifest"], d["row_filter"], "train", classes)
    va = load_manifest(d["val_manifest"], d["row_filter"], "val", classes)
    ds_tr = ManifestDataset(tr, build_transforms(t, a, train=True))
    ds_va = ManifestDataset(va, build_transforms(t, a, train=False))
    if max_batches:
        g = np.random.default_rng(0)
        ds_tr = Subset(ds_tr, g.choice(len(ds_tr), max_batches * t["batch_size"], replace=False).tolist())
        ds_va = Subset(ds_va, g.choice(len(ds_va), max_batches * t["batch_size"], replace=False).tolist())
    gen = torch.Generator().manual_seed(t["seed"])
    kw = dict(batch_size=t["batch_size"], num_workers=t["num_workers"], pin_memory=True,
              persistent_workers=t["num_workers"] > 0, worker_init_fn=seed_worker)
    sampler = None
    if t["class_balancing"] == "sampler":
        w = class_weights(tr["class_id"].to_numpy(), len(classes), 1.0).numpy()[tr["class_id"].to_numpy()]
        sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True, generator=gen)
    dl_tr = DataLoader(ds_tr, shuffle=sampler is None, sampler=sampler, generator=gen, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, shuffle=False, **kw)
    return tr, va, dl_tr, dl_va


def plot_history(hist, out_dir: Path, best: int):
    """`best` = the epoch actually saved (selection uses early_stopping.min_delta)."""
    from src.visualization.style import SERIES, apply_style, save
    import matplotlib.pyplot as plt
    apply_style()
    ep = [h["epoch"] for h in hist]
    for key, ylabel, fname in (("loss", "Weighted cross-entropy", "loss_curves.png"),
                               ("accuracy", "Accuracy", "accuracy_curves.png")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(ep, [h[f"train_{key}"] for h in hist], color=SERIES[0], lw=2, marker="o", ms=4, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], color=SERIES[1], lw=2, marker="o", ms=4, label="validation (known)")
        b = [h["epoch"] for h in hist if h["phase"] == "B"]
        if b:
            ax.axvspan(b[0] - 0.5, ep[-1] + 0.5, color="#e1e0d9", alpha=0.4, lw=0)
            ax.text(b[0] - 0.4, ax.get_ylim()[1], " phase B (fine-tune)", va="top", fontsize=8, color="#52514e")
        ax.axvline(best, color="#52514e", ls="--", lw=1)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"Baseline {key}: train vs validation (selected checkpoint: epoch {best}, dashed)")
        ax.legend(loc="best")
        save(fig, out_dir / fname)


def replot(cfg, run: str) -> None:
    h = json.loads((resolve(cfg["paths"]["metrics"]) / f"{run}_training_history.json").read_text())
    plot_history(h["history"], resolve(cfg["paths"]["figures"]) / run, h["best_epoch"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only redraw curves from saved history")
    ap.add_argument("--max-batches", type=int, default=None, help="smoke test: tiny subsets, 1 epoch per phase")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["baseline"]
    t = b["training"]
    run = args.run_name or (b["run_name"] if not args.max_batches else "smoke_baseline")
    if args.replot:
        return replot(cfg, run)

    lock = verify_lock(cfg)                                  # refuse to train on a modified protocol
    classes = known_classes(cfg)
    if len(classes) != b["model"]["num_classes"]:
        raise SystemExit(f"num_classes {b['model']['num_classes']} != {len(classes)} known classes")

    model_dir = resolve(cfg["paths"]["models"]) / run
    metrics_dir, fig_dir = resolve(cfg["paths"]["metrics"]), resolve(cfg["paths"]["figures"]) / run
    if (model_dir / "best_model.pt").exists() and not (args.overwrite or args.max_batches):
        raise SystemExit(f"{model_dir}/best_model.pt exists — use --run-name for a new run or --overwrite")
    for p in (model_dir, metrics_dir, fig_dir):
        p.mkdir(parents=True, exist_ok=True)

    seed_everything(t["seed"], t["deterministic"])
    device = torch.device("cuda" if (b["hardware"]["device"] == "auto" and torch.cuda.is_available())
                          or b["hardware"]["device"] == "cuda" else "cpu")
    env = environment(t["seed"], device)
    (metrics_dir / f"{run}_environment.json").write_text(json.dumps(env, indent=2))
    print(f"run={run} device={device} ({env.get('gpu', 'cpu')})")

    tr, va, dl_tr, dl_va = loaders(cfg, classes, args.max_batches)
    print(f"train {len(tr):,} images | val {len(va):,} images (known only) | classes {len(classes)}")

    model = Classifier(b["model"]["architecture"], len(classes), b["model"]["pretrained"]).to(device)
    w = class_weights(tr["class_id"].to_numpy(), len(classes), t["class_weight_power"]) \
        if t["class_balancing"] == "weighted_ce" else None
    criterion = nn.CrossEntropyLoss(weight=None if w is None else w.to(device), label_smoothing=t["label_smoothing"])
    scaler = torch.amp.GradScaler(device.type, enabled=t["amp"] and device.type == "cuda")

    hist, best, best_state, bad, epoch = [], -1.0, None, 0, 0
    t_start = time.time()

    def run_epoch(phase, optimizer, scheduler):
        nonlocal best, best_state, bad, epoch
        epoch += 1
        trm = train_one_epoch(model, dl_tr, criterion, optimizer, scaler, device, scheduler, t["amp"],
                              desc=f"ep{epoch} {phase} train")
        logits, y, _ = predict(model, dl_va, device, t["amp"], desc=f"ep{epoch} val")
        vm = summary_metrics(y.numpy(), logits.argmax(1).numpy(), len(classes))
        h = {"epoch": epoch, "phase": phase, "train_loss": trm["loss"], "train_accuracy": trm["accuracy"],
             "val_loss": eval_loss(logits, y, criterion), "val_accuracy": vm["accuracy"],
             "val_macro_f1": vm["macro_f1"], "lr": [g["lr"] for g in optimizer.param_groups],
             "train_seconds": round(trm["seconds"], 1)}
        hist.append(h)
        improved = h["val_macro_f1"] > best + t["early_stopping"]["min_delta"]
        if improved:
            best, bad = h["val_macro_f1"], 0
            best_state = {"epoch": epoch, "phase": phase, "state_dict": copy.deepcopy(model.state_dict()),
                          "val_metrics": vm, "val_loss": h["val_loss"]}
        else:
            bad += 1
        print(f"  ep{epoch:02d} [{phase}] train loss {h['train_loss']:.4f} acc {h['train_accuracy']:.4f} | "
              f"val loss {h['val_loss']:.4f} acc {h['val_accuracy']:.4f} macroF1 {h['val_macro_f1']:.4f}"
              f"{' *' if improved else ''} ({h['train_seconds']}s)")

    # ---------------- Phase A: head only
    pa = t["phase_a"]
    model.freeze_backbone_except([])
    opt = torch.optim.AdamW(model.param_groups(pa["learning_rate"], 0.0), weight_decay=t["weight_decay"])
    for _ in range(1 if args.max_batches else pa["epochs"]):
        run_epoch("A", opt, None)

    # ---------------- Phase B: fine-tune later stages
    pb = t["phase_b"]
    model.freeze_backbone_except(pb["unfreeze"])
    opt = torch.optim.AdamW(model.param_groups(pb["learning_rate"], pb["backbone_learning_rate"]),
                            weight_decay=t["weight_decay"])
    n_ep = 1 if args.max_batches else pb["max_epochs"]
    sched = warmup_cosine(opt, pb["warmup_epochs"] * len(dl_tr), n_ep * len(dl_tr))
    bad, stopped_early = 0, False
    for _ in range(n_ep):
        run_epoch("B", opt, sched)
        if bad >= t["early_stopping"]["patience"]:
            stopped_early = True
            print(f"  early stopping: no val macro-F1 improvement for {bad} epochs")
            break
    train_minutes = (time.time() - t_start) / 60

    # ---------------- save
    ckpt = {"state_dict": best_state["state_dict"], "architecture": b["model"]["architecture"],
            "num_classes": len(classes), "class_names": classes, "best_epoch": best_state["epoch"],
            "best_phase": best_state["phase"], "val_metrics": best_state["val_metrics"],
            "protocol_lock_created": lock["created_utc"], "run_name": run,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    torch.save(ckpt, model_dir / "best_model.pt")
    (model_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy(resolve("data/splits/PROTOCOL_LOCK.json"), model_dir / "protocol_lock_used.json")
    summary = {"run_name": run, "epochs_run": epoch, "phase_a_epochs": sum(h["phase"] == "A" for h in hist),
               "phase_b_epochs": sum(h["phase"] == "B" for h in hist), "stopped_early": stopped_early,
               "best_epoch": best_state["epoch"], "best_phase": best_state["phase"],
               "best_val": best_state["val_metrics"], "best_val_loss": best_state["val_loss"],
               "training_minutes": round(train_minutes, 2),
               "class_weights": None if w is None else dict(zip(classes, map(float, w))),
               "train_images": len(tr), "val_images": len(va), "history": hist}
    (metrics_dir / f"{run}_training_history.json").write_text(json.dumps(summary, indent=2))
    plot_history(hist, fig_dir, best_state["epoch"])
    print(f"\nbest epoch {best_state['epoch']} ({best_state['phase']}): val macro-F1 "
          f"{best_state['val_metrics']['macro_f1']:.4f}, acc {best_state['val_metrics']['accuracy']:.4f} | "
          f"{train_minutes:.1f} min -> {model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
