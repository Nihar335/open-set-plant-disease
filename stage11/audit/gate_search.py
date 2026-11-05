"""Explore decision gates on the CALIBRATION split only (PlantVillage/PlantDoc/apple val, DTD + D2-dev calib halves).

For every photo the user's crop is simulated:
  known photos  : the true crop (correct claim) and every other model crop (wrong claim)
  OOD photos    : every model crop (the user names a supported crop for a texture or an unsupported plant)
A photo is "named" when the rule outputs a disease at >= 80%.

Run:  python -P stage11/audit/gate_search.py [--tag v2]
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


def load(tag, which, split):
    return dict(np.load(ROOT / f"stage11_local/audit/{tag}_{which}_{split}.npz"))


def probs(z, view, T):
    if view == "centre":
        return softmax(z["lc"] / T)
    if view == "whole":
        return softmax(z["lw"] / T)
    return 0.5 * (softmax(z["lc"] / T) + softmax(z["lw"] / T))


def logits_view(z, view):
    return z["lc"] if view == "centre" else z["lw"] if view == "whole" else 0.5 * (z["lc"] + z["lw"])


def rule(p_full, energy, crop_mask, m_min, e_min):
    """New rule: within-crop probability >= 80%, AND in-crop mass >= m_min, AND energy >= e_min."""
    in_mass = (p_full * crop_mask).sum(1)
    pc = p_full * crop_mask / (in_mass[:, None] + 1e-12)
    return pc.argmax(1), (pc.max(1) >= THRESHOLD) & (in_mass >= m_min) & (energy >= e_min), in_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    T = cal["temperature"]
    crops = np.array([d["crop"] for d in kb.model_classes])
    ucrops = sorted(set(crops))
    K, O = load(a.tag, "known", "calib"), load(a.tag, "ood", "calib")
    y, src = K["label"], K["source"]
    true_mask = crops[None, :] == crops[y][:, None]

    # ---- current v2 rule (centre view, m0 penalty), correct crop claim
    out = fuse(K["lc"], K["sc"], true_mask, kb.symptom_targets(), T=T, beta=cal["beta"], m0=cal["m0"])
    named0 = out["conf"].max(1) >= THRESHOLD
    pred0 = out["conf"].argmax(1)
    print("CURRENT v2 rule (correct crop):")
    for s in ("lab", "field", "apple_field"):
        m = src == s
        print(f"  {s:12s} coverage {named0[m].mean():.1%}  named acc {np.mean(pred0[m & named0] == y[m & named0]):.1%}")
    fa_ood = []
    for c in ucrops:
        cm = np.broadcast_to(crops == c, O["lc"].shape)
        o = fuse(O["lc"], O["sc"], cm, kb.symptom_targets(), T=T, beta=cal["beta"], m0=cal["m0"])
        fa_ood.append(o["conf"].max(1) >= THRESHOLD)
    fa_ood = np.stack(fa_ood, 1)
    for s in np.unique(O["source"]):
        print(f"  OOD {s:18s} named for a claimed crop: {fa_ood[O['source'] == s].mean():.1%}")

    print("\nCANDIDATE rule: p_crop >= 0.8 & in_mass >= m & energy >= e   (calibration split)")
    print(f"{'view':7s} {'m':>5s} {'e_pct':>5s} | {'field cov':>9s} {'field acc':>9s} | {'lab cov':>7s} {'apple cov':>9s} | "
          f"{'wrongcrop FA':>12s} | {'dtd FA':>6s} {'plant FA':>8s}")
    rows = []
    for view in ("centre", "whole", "mean"):
        pk, po = probs(K, view, T), probs(O, view, T)
        ek = np.log(np.exp(logits_view(K, view) / T).sum(1)) * T
        eo = np.log(np.exp(logits_view(O, view) / T).sum(1)) * T
        field = src == "field"
        for e_pct in (0, 1, 2, 5):
            e_min = np.percentile(ek[field], e_pct) if e_pct else -np.inf
            for m_min in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9416):
                pred, named, _ = rule(pk, ek, true_mask, m_min, e_min)
                wrong = []
                for c in ucrops:
                    other = crops[y] != c
                    _, n2, _ = rule(pk[other], ek[other], np.broadcast_to(crops == c, pk[other].shape), m_min, e_min)
                    wrong.append(n2.mean())
                fo = np.stack([rule(po, eo, np.broadcast_to(crops == c, po.shape), m_min, e_min)[1] for c in ucrops], 1)
                r = dict(view=view, m=m_min, e_pct=e_pct,
                         field_cov=named[field].mean(), field_acc=np.mean(pred[field & named] == y[field & named]),
                         lab_cov=named[src == "lab"].mean(), apple_cov=named[src == "apple_field"].mean(),
                         wrong=float(np.mean(wrong)), dtd=fo[O["source"] == "dtd_texture"].mean(),
                         plant=fo[O["source"] == "unsupported_plant"].mean())
                rows.append(r)
                print(f"{view:7s} {m_min:5.2f} {e_pct:5d} | {r['field_cov']:9.1%} {r['field_acc']:9.1%} | "
                      f"{r['lab_cov']:7.1%} {r['apple_cov']:9.1%} | {r['wrong']:12.2%} | {r['dtd']:6.2%} {r['plant']:8.2%}")
    (ROOT / f"stage11_local/audit/{a.tag}_gate_search.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
