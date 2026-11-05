"""Cache the v2 model's validation logits / symptom probabilities for the audit (validation only, no test).

Run:  python -P stage11/audit/collect_val.py [--ckpt models/stage11/best.pt] [--out stage11_local/audit/v2_val.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.evaluate import collect  # noqa: E402
from stage11.kb import load_kb  # noqa: E402
from stage11.model import LeafNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/stage11/best.pt")
    ap.add_argument("--index", default="stage11/results/image_index.csv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="stage11_local/audit/v2_val.npz")
    a = ap.parse_args()
    kb = load_kb()
    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    model = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda().eval()
    idx = pd.read_csv(ROOT / a.index)
    df = idx[idx.split == a.split].reset_index(drop=True)
    lg, sy = collect(model, df, torch.device("cuda"))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=lg, sym=sy, label=df.label.to_numpy(), source=df.source.to_numpy().astype(str),
             path=df.path.to_numpy().astype(str))
    print(len(df), "images ->", out, "classes", len(kb.model_classes))


if __name__ == "__main__":
    main()
