"""Shortcut probe (validation photos only): does the model still pick the right class when the lesions are removed?

Views of each photo:
  original     : app view
  blurred      : Gaussian blur radius 12 on the 384 px photo (spots and pustules disappear, colour / layout stay)
  centre_grey  : the central 60% of the photo painted grey (the leaf is mostly hidden; borders / background stay)
If accuracy among the crop's classes stays far above chance without visible lesions, the class is being predicted
from capture style (camera, background, lighting) that is specific to one source dataset.

Run:  python -P stage11/audit/shortcut_probe.py [--ckpt models/stage11/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF, LeafNet  # noqa: E402


def views(img):
    blur = img.filter(ImageFilter.GaussianBlur(12 * max(img.size) / 384))
    grey = img.copy()
    w, h = img.size
    ImageDraw.Draw(grey).rectangle([0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h], fill=(128, 128, 128))
    return {"original": img, "blurred": blur, "centre_grey": grey}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/stage11/best.pt")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--out", default="stage11/audit/shortcut_probe_v2.json")
    a = ap.parse_args()
    kb = load_kb()
    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    model = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda().eval()
    crops = np.array([d["crop"] for d in kb.model_classes])
    idx = pd.read_csv(ROOT / "stage11/results/image_index.csv")
    val = idx[idx.split == "val"]
    res = {}
    for src_name, srcs in {"apple_field": ["appleleaf9", "kashmir_apple"], "lab": ["plantvillage"],
                           "field": ["plantdoc"]}.items():
        df = val[val.source.isin(srcs)]
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), a.per_class), random_state=0))
        if src_name == "lab":  # keep the probe comparable: only the crops that also have field data
            df = df[df.class_key.str.startswith(("apple_", "tomato_", "potato_", "corn_", "grape_"))]
        hits = {k: [] for k in ("original", "blurred", "centre_grey")}
        for p, y in zip(df.path, df.label):
            img = Image.open(ROOT / p).convert("RGB")
            vs = views(img)
            x = torch.stack([EVAL_TF(vs[k]) for k in hits]).cuda()
            lg = model(x)[0].float().cpu().numpy()
            mask = crops == crops[y]
            for k, l in zip(hits, lg):
                l = np.where(mask, l, -np.inf)
                hits[k].append(int(l.argmax() == y))
        n_cls = df.groupby(df.label.map(lambda i: crops[i])).label.nunique().mean()
        res[src_name] = {"n": len(df), "mean_classes_per_crop": round(float(n_cls), 2),
                         **{k: round(float(np.mean(v)), 3) for k, v in hits.items()}}
        print(src_name, res[src_name], flush=True)
    (ROOT / a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
