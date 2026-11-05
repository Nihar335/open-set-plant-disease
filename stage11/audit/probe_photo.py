"""Probe one photo: all-class image probabilities and in-crop mass under the centre crop and the whole-photo view.

Run:  python -P stage11/audit/probe_photo.py <photo> <crop> [--ckpt models/stage11/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.decision import softmax  # noqa: E402
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF, WHOLE_TF, LeafNet  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("crop")
    ap.add_argument("--ckpt", default="models/stage11/best.pt")
    a = ap.parse_args()
    kb = load_kb()
    T = json.loads((ROOT / "models/stage11/calibration.json").read_text())["temperature"]
    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    model = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img = ImageOps.exif_transpose(Image.open(a.photo)).convert("RGB")
    crops = np.array([d["crop"] for d in kb.model_classes])
    keys = [d["key"] for d in kb.model_classes]
    for name, tf in (("centre", EVAL_TF), ("whole", WHOLE_TF)):
        p = softmax(model(tf(img)[None])[0][0].numpy() / T)
        m = crops == a.crop
        pc = p * m / p[m].sum()
        top = np.argsort(-p)[:4]
        print(f"{name:6s} in-crop mass {p[m].sum():.3f} | all classes: "
              + ", ".join(f"{keys[i]} {p[i]:.2f}" for i in top)
              + " | within crop: " + ", ".join(f"{keys[i]} {pc[i]:.2f}" for i in np.argsort(-pc)[:3]))


if __name__ == "__main__":
    main()
