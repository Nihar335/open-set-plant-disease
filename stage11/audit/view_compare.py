"""Compare the v2 fuse rule with the centre view vs the mean of centre + whole-photo views (calibration split only).

m0 is re-derived for each view with the documented v2 method (5th percentile of in-crop mass on field val photos).
Run:  python -P stage11/audit/view_compare.py [--tag v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD, fuse, softmax  # noqa: E402
from stage11.kb import load_kb  # noqa: E402


def as_logits(z, view, T):
    if view == "centre":
        return z["lc"], z["sc"]
    p = 0.5 * (softmax(z["lc"] / T) + softmax(z["lw"] / T))
    return T * np.log(p + 1e-12), 0.5 * (z["sc"] + z["sw"])  # softmax(l / T) == p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    T, prof = cal["temperature"], kb.symptom_targets()
    crops = np.array([d["crop"] for d in kb.model_classes])
    K = dict(np.load(ROOT / f"stage11_local/audit/{a.tag}_known_calib.npz"))
    O = dict(np.load(ROOT / f"stage11_local/audit/{a.tag}_ood_calib.npz"))
    y, src = K["label"], K["source"]
    mask = crops[None, :] == crops[y][:, None]
    fieldish = np.isin(src, ["field", "apple_field"])
    for view in ("centre", "mean"):
        lk, sk = as_logits(K, view, T)
        lo, so = as_logits(O, view, T)
        m0 = float(np.percentile(fuse(lk[fieldish], sk[fieldish], mask[fieldish], prof, T=T)["in_mass"], 5))
        out = fuse(lk, sk, mask, prof, T=T, m0=m0)
        pred, named = out["conf"].argmax(1), out["conf"].max(1) >= THRESHOLD
        line = [f"{view:6s} m0={m0:.3f}"]
        for s in ("lab", "field", "apple_field"):
            m = src == s
            line.append(f"{s} top1 {np.mean(pred[m] == y[m]):.1%} cov {named[m].mean():.1%} "
                        f"acc {np.mean(pred[m & named] == y[m & named]):.1%}")
        fa = np.stack([fuse(lo, so, np.broadcast_to(crops == c, lo.shape), prof, T=T, m0=m0)["conf"].max(1)
                       >= THRESHOLD for c in sorted(set(crops))], 1)
        for s in np.unique(O["source"]):
            line.append(f"{s} FA {fa[O['source'] == s].mean():.1%}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
