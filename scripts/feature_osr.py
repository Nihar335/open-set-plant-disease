"""Stage 6 — feature-space analysis and Mahalanobis OSR on the FROZEN Stage-3 model.

    python scripts/feature_osr.py extract    # penultimate embeddings for train/val/test (+ consistency checks)
    python scripts/feature_osr.py select     # Mahalanobis fit (TRAIN known only) + validation threshold (frozen)
    python scripts/feature_osr.py evaluate   # protocol check, test scoring once, feature-space analyses, figures
    python scripts/feature_osr.py report     # results/reports/STAGE_6_FEATURE_OSR.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_msp import eval_pair, load_frozen, manifest, sha256  # noqa: E402
from src.data.dataset import ManifestDataset  # noqa: E402
from src.data.protocol import known_classes, verify_lock  # noqa: E402
from src.osr.embeddings import extract, load_npz, save_npz  # noqa: E402
from src.osr.scores import energy_score, msp_scores  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

SPLITS = ("train", "val", "test")
META_COLS = ["image_path", "dataset", "crop", "disease", "class_name", "class_id", "role", "known_unknown", "split",
             "pv_equivalent", "strict_test", "eval_subset", "experiments"]


def emb_dir(cfg) -> Path:
    return resolve(cfg["feature_osr"]["embeddings_dir"])


def load_split(cfg, split):
    E, L, paths = load_npz(emb_dir(cfg) / f"{split}_embeddings.npz")
    meta = pd.read_csv(emb_dir(cfg) / f"{split}_meta.csv", low_memory=False, keep_default_na=False)
    if paths != meta["image_path"].tolist():
        raise SystemExit(f"STOP: {split} embeddings and metadata are misaligned")
    meta["class_id"] = meta["class_id"].astype(int)
    meta["strict_test"] = meta["strict_test"].astype(str).str.lower().eq("true")
    return meta, E.astype(np.float64), L.astype(np.float64)


# ----------------------------------------------------------------------------- extract
def cmd_extract(cfg):
    fo = cfg["feature_osr"]
    model, classes, tf, device, s3, ck = load_frozen(cfg)          # verifies lock + checkpoint hash + 21 outputs
    out = emb_dir(cfg)
    checks = {"embedding_dim": int(model.feature_dim), "splits": {}}
    for split in SPLITS:
        df = manifest(f"data/splits/{split}.csv", split)
        dl = DataLoader(ManifestDataset(df, tf), batch_size=fo["extract_batch_size"], shuffle=False,
                        num_workers=s3["training"]["num_workers"], pin_memory=True)
        E, L, I = extract(model, dl, device, fp32=fo["inference_precision"] == "fp32", desc=f"embed {split}")
        assert (I == np.arange(len(df))).all() and E.shape == (len(df), model.feature_dim)
        ms = msp_scores(L)
        meta = df[[c for c in META_COLS if c in df.columns]].copy()
        meta["pred_class_id"] = ms["pred"]
        meta["pred_class"] = [classes[i] for i in ms["pred"]]
        meta["msp"] = ms["msp"]
        meta["energy"] = energy_score(L, 1.0)
        save_npz(out / f"{split}_embeddings.npz", E, L, meta)
        meta.to_csv(out / f"{split}_meta.csv", index=False)
        W, b = (model.head.weight.detach().cpu().numpy().astype(np.float64),
                model.head.bias.detach().cpu().numpy().astype(np.float64))
        c = {"n": int(len(df)), "finite": bool(np.isfinite(E).all()),
             "head_consistency_max_abs": float(np.abs(E.astype(np.float64) @ W.T + b - L).max()),
             "embedding_norm_median": float(np.median(np.linalg.norm(E, axis=1)))}
        cache = resolve(cfg["paths"]["results"]) / "predictions" / "cache" / f"msp_logits_{split}.npz"
        if cache.exists():
            z = np.load(cache)
            assert list(z["paths"]) == df["image_path"].tolist()
            c["vs_stage4_logits_max_abs"] = float(np.abs(z["logits"] - L).max())
            c["vs_stage4_argmax_mismatches"] = int((z["logits"].argmax(1) != L.argmax(1)).sum())
        checks["splits"][split] = c
        print(f"  {split}: {E.shape} {c}")
    # determinism: re-extract the first batch of the test split
    df = manifest("data/splits/test.csv", "test")
    dl = DataLoader(Subset(ManifestDataset(df, tf), list(range(fo["extract_batch_size"]))),
                    batch_size=fo["extract_batch_size"], shuffle=False, num_workers=0)
    E2, _, _ = extract(model, dl, device, fp32=True, desc="determinism")
    E1, _, _ = load_npz(out / "test_embeddings.npz")
    checks["determinism_rerun_max_abs_diff"] = float(np.abs(E1[: len(E2)] - E2).max())
    checks["checkpoint_sha256"] = cfg["osr_msp"]["checkpoint_sha256"]
    checks["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "extraction_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({k: v for k, v in checks.items() if k != "splits"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "select", "evaluate", "report"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.step == "extract":
        cmd_extract(cfg)
    elif a.step == "select":
        from scripts.feature_osr_steps import cmd_select
        cmd_select(cfg)
    elif a.step == "evaluate":
        from scripts.feature_osr_steps import cmd_evaluate
        cmd_evaluate(cfg)
    else:
        from scripts.feature_osr_report import write_report
        write_report(cfg)


if __name__ == "__main__":
    main()
