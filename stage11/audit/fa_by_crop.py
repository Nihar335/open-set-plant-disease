"""False naming per claimed crop under the CURRENT v2 rule (calibration split of the OOD sets + wrong-crop claims).

Run:  python -P stage11/audit/fa_by_crop.py [--tag v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD, fuse  # noqa: E402
from stage11.kb import load_kb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    crops = np.array([d["crop"] for d in kb.model_classes])
    keys = [d["key"] for d in kb.model_classes]
    O = dict(np.load(ROOT / f"stage11_local/audit/{a.tag}_ood_calib.npz"))
    K = dict(np.load(ROOT / f"stage11_local/audit/{a.tag}_known_calib.npz"))
    field = K["source"] == "field"
    print(f"{'claimed crop':12s} {'#classes':>8s} {'OOD named':>9s} {'field photo of another crop named':>34s}  most common named")
    for c in sorted(set(crops)):
        o = fuse(O["lc"], O["sc"], np.broadcast_to(crops == c, O["lc"].shape), kb.symptom_targets(),
                 T=cal["temperature"], beta=cal["beta"], m0=cal["m0"])
        n = o["conf"].max(1) >= THRESHOLD
        other = field & (crops[K["label"]] != c)
        k = fuse(K["lc"][other], K["sc"][other], np.broadcast_to(crops == c, K["lc"][other].shape),
                 kb.symptom_targets(), T=cal["temperature"], beta=cal["beta"], m0=cal["m0"])
        nk = k["conf"].max(1) >= THRESHOLD
        top = Counter(keys[i] for i in o["conf"][n].argmax(1)).most_common(2)
        print(f"{c:12s} {int((crops == c).sum()):8d} {n.mean():9.1%} {nk.mean():34.1%}  {dict(top)}")


if __name__ == "__main__":
    main()
