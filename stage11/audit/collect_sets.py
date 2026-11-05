"""Cache logits / symptom probabilities for the audit sets under two views of each photo.

  centre : Resize(256) + CenterCrop(224)  (current app view; cuts the edges of a phone photo)
  whole  : whole photo letter-boxed to 224 x 224 (nothing cut off)

Run:  python -P stage11/audit/collect_sets.py --which ood|known --split calib|eval [--ckpt ...] [--tag v2]
Outputs stage11_local/audit/<tag>_<which>_<split>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.audit.sets import known_sets, ood_sets  # noqa: E402
from stage11.kb import load_kb  # noqa: E402
from stage11.model import EVAL_TF, WHOLE_TF, LeafNet  # noqa: E402


class Views(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        return EVAL_TF(img), WHOLE_TF(img)


@torch.no_grad()
def run(model, paths):
    dl = DataLoader(Views(paths), batch_size=64, num_workers=2)
    out = {k: [] for k in ("lc", "sc", "lw", "sw")}
    for xc, xw in dl:
        for x, a, b in ((xc, "lc", "sc"), (xw, "lw", "sw")):
            with torch.autocast("cuda", dtype=torch.float16):
                lg, sy = model(x.cuda())
            out[a].append(lg.float().cpu())
            out[b].append(torch.sigmoid(sy.float()).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ood", "known"], required=True)
    ap.add_argument("--split", choices=["calib", "eval"], required=True)
    ap.add_argument("--ckpt", default="models/stage11/best.pt")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    kb = load_kb()
    df = (ood_sets() if a.which == "ood" else known_sets(kb))
    df = df[df.split == a.split].reset_index(drop=True)
    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    model = LeafNet(len(ckpt["classes"]), len(ckpt["symptoms"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda().eval()
    res = run(model, df.path.tolist())
    out = ROOT / f"stage11_local/audit/{a.tag}_{a.which}_{a.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res, label=df.label.to_numpy(), source=df.source.to_numpy().astype(str),
             crop=df.crop.to_numpy().astype(str), path=df.path.to_numpy().astype(str))
    print(len(df), "->", out)


if __name__ == "__main__":
    main()
