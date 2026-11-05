"""Audit of the v2 decision rule on validation photos (no test data).

Run:  python -P stage11/audit/analyse_val.py [stage11_local/audit/v2_val.npz]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD, fuse  # noqa: E402
from stage11.kb import load_kb  # noqa: E402


def main(path):
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    z = np.load(ROOT / path, allow_pickle=False)
    lg, sy, y, src = z["logits"], z["sym"], z["label"], z["source"]
    crops = np.array([d["crop"] for d in kb.model_classes])
    keys = [d["key"] for d in kb.model_classes]
    mask = crops[None, :] == crops[y][:, None]
    out = fuse(lg, sy, mask, kb.symptom_targets(), T=cal["temperature"], beta=cal["beta"], m0=cal["m0"])
    conf = out["conf"]
    pred, top = conf.argmax(1), conf.max(1)
    named = top >= THRESHOLD
    print(f"m0 = {cal['m0']}  (fitted as 5th pct of in-crop mass over ALL field val photos)")
    for s in np.unique(src):
        m = src == s
        im = out["in_mass"][m]
        print(f"{s:14s} n={m.sum():5d} in_mass p5={np.percentile(im, 5):.3f} median={np.median(im):.3f} "
              f"penalised={np.mean(out['fit'][m] < 1):.1%} top1={np.mean(pred[m] == y[m]):.1%} "
              f"named={named[m].mean():.1%} named_acc={np.mean(pred[m & named] == y[m & named]):.1%} "
              f"top1_unpenalised_rule={np.mean(out['fused'][m].argmax(1) == y[m]):.1%}")
    # what a per-source-balanced m0 would be
    per = {s: float(np.percentile(out["in_mass"][(src == s)], 5)) for s in ("plantdoc", "appleleaf9", "kashmir_apple")}
    print("per-source p5 of in-crop mass:", {k: round(v, 3) for k, v in per.items()})
    # PlantDoc: is the penalty helping or hurting? accuracy of penalised photos under each ranking
    m = (src == "plantdoc") & (out["fit"] < 1)
    print(f"PlantDoc penalised photos n={m.sum()}: top1 with blended rank {np.mean(pred[m] == y[m]):.1%}, "
          f"top1 image-only crop-restricted {np.mean(out['p_crop'][m].argmax(1) == y[m]):.1%}, "
          f"top1 symptom-rank only {np.mean(out['symptom_rank'][m].argmax(1) == y[m]):.1%}")
    # where the out-of-crop mass goes for penalised PlantDoc photos
    other = (out["p_full"] * ~mask)
    oc = crops[other.argmax(1)]
    pairs = {}
    for i in np.nonzero(m)[0]:
        k = f"{crops[y[i]]}->{oc[i]}"
        pairs[k] = pairs.get(k, 0) + 1
    print("penalised PlantDoc: true crop -> crop taking the out-of-crop mass:", dict(sorted(pairs.items(), key=lambda t: -t[1])))
    # apple black rot
    br = keys.index("apple_black_rot")
    for s in np.unique(src):
        m = (src == s) & (y == br)
        if m.any():
            print(f"apple_black_rot {s}: n={m.sum()} top1={np.mean(pred[m] == br):.1%} named={named[m].mean():.1%} "
                  f"in_mass median={np.median(out['in_mass'][m]):.3f}")
    # how often other-crop look-alikes take mass from apple photos
    am = (crops[y] == "apple") & (src == "plantdoc")
    print("PlantDoc apple val photos: mean out-of-crop mass by class:",
          {keys[j]: round(float(out["p_full"][am, j].mean()), 3) for j in np.argsort(-out["p_full"][am].mean(0))[:6]})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage11_local/audit/v2_val.npz")
