"""Validation-only check of the current naming threshold (decision.THRESHOLD, 60% since 2026-10-23) vs the earlier 80%.

Uses the cached centre-view outputs of the VALIDATION split only (stage11_local/audit/v2_known_calib.npz and
v2_ood_calib.npz, made by stage11/audit/collect_sets.py --split calib from models/stage11/best.pt). No test photo
is loaded or scored. Caveat: the same photos were used to fit T / beta / m0 and the v3 gate, so these numbers are
in-sample and likely optimistic; they do not replace a held-out test.

Rule = decision.may_name (the app's v3 rule): score >= threshold AND crop diagnosable AND no crop conflict.
The 80% column is the same rule with the threshold at 0.80 (v3 at 0.60 AND score >= 0.80).

Run:  python -P stage11/evaluate_val_threshold.py  -> stage11/results/val_threshold_eval.json + VAL_THRESHOLD_EVALUATION.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD  # noqa: E402
from stage11.evaluate_multicrop import Rules  # noqa: E402
from stage11.kb import load_kb  # noqa: E402

RESULTS = ROOT / "stage11" / "results"
AUDIT = ROOT / "stage11_local" / "audit"
OLD = 0.80


def block(y, r, sel, named):
    """Named rate, accuracy of named, and top-3 hit rate of the 'Unknown' photos for one subset."""
    pred, top3 = r["pred"][sel], r["top3"][sel]
    y, n = y[sel], named[sel]
    return {"n": int(sel.sum()), "named": float(n.mean()), "n_named": int(n.sum()),
            "named_acc": float((pred[n] == y[n]).mean()) if n.any() else None,
            "unknown_top3_has_true": float((top3[~n] == y[~n, None]).any(1).mean()) if (~n).any() else None}


def main():
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    R = Rules(kb, cal)
    crops = R.crops
    model_crops = sorted(set(crops))
    K = dict(np.load(AUDIT / "v2_known_calib.npz"))
    O = dict(np.load(AUDIT / "v2_ood_calib.npz"))
    y, src = K["label"], K["source"]
    r = R.apply(K["lc"], K["sc"], crops[y])
    rules = {"current": r["v3"], "old_80": r["v3"] & (r["top"] >= OLD)}

    out = {"created": datetime.now().isoformat(timespec="seconds"), "split": "validation only (audit 'calib')",
           "checkpoint": "models/stage11/best.pt (v2, epoch 10)", "calibration": cal,
           "threshold": THRESHOLD, "old_threshold": OLD, "by_source": {}, "wrong_crop_claims": {}, "ood": {}}
    for s in ("lab", "field", "apple_field"):
        sel = src == s
        out["by_source"][s] = {
            "n": int(sel.sum()),
            "top1": float((r["pred"][sel] == y[sel]).mean()),
            "top3": float((r["top3"][sel] == y[sel, None]).any(1).mean()),
            **{k: block(y, r, sel, v) for k, v in rules.items()}}

    # false naming: photo of one model crop, user names another model crop
    for s in ("lab", "field"):
        cnt, tot = {"current": 0, "old_80": 0}, 0
        for c in model_crops:
            sel = (src == s) & ~np.isin(crops[y], kb.crop_keys(c))
            if not sel.any():
                continue
            rr = R.apply(K["lc"][sel], K["sc"][sel], np.array([c] * sel.sum()))
            cnt["current"] += int(rr["v3"].sum())
            cnt["old_80"] += int((rr["v3"] & (rr["top"] >= OLD)).sum())
            tot += int(sel.sum())
        out["wrong_crop_claims"][s] = {"pairs": tot, **{k: v / tot for k, v in cnt.items()}}

    # false naming: textures / unsupported plants, claimed as every model crop
    for s in np.unique(O["source"]):
        sel = O["source"] == s
        cur, old = [], []
        for c in model_crops:
            rr = R.apply(O["lc"][sel], O["sc"][sel], np.array([c] * sel.sum()))
            cur.append(rr["v3"].mean())
            old.append((rr["v3"] & (rr["top"] >= OLD)).mean())
        out["ood"][str(s)] = {"n_photos": int(sel.sum()), "current": float(np.mean(cur)), "old_80": float(np.mean(old))}

    (RESULTS / "val_threshold_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(RESULTS / "test_runs.log", "a", encoding="utf-8") as f:
        f.write(f"NOTE {out['created']} evaluate_val_threshold.py: v3 rule at threshold {THRESHOLD} vs {OLD} on the "
                f"VALIDATION split only (cached v2 outputs); no test photo loaded or scored\n")
    write_md(out)
    print(json.dumps(out, indent=1))


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def write_md(o):
    t, t0 = pct(o["threshold"]), pct(o["old_threshold"])
    L = [f"# Naming threshold {t} vs {t0}: validation-only check", "",
         f"Generated {o['created']} by `stage11/evaluate_val_threshold.py` on the **validation split only** "
         "(cached outputs of the unchanged v2 checkpoint). No test photo was loaded or scored.", "",
         "**Caveat:** these validation photos were also used to fit T / beta / m0 and the crop gate, so the numbers are "
         "in-sample and likely optimistic. The score is a heuristic, not a probability of being correct.", "",
         f"Rule = the app's `may_name`: score ≥ threshold AND crop diagnosable AND no crop conflict.", "",
         "## Crop stated correctly", "",
         f"| | Lab (PlantVillage val) | Field (PlantDoc val) | Apple field val |", "|---|---|---|---|"]
    b = o["by_source"]
    S = ("lab", "field", "apple_field")
    rows = [("Photos", lambda x: str(x["n"])), ("Top-1 within stated crop", lambda x: pct(x["top1"])),
            ("Top-3 within stated crop", lambda x: pct(x["top3"]))]
    for k, lbl in (("current", t), ("old_80", t0)):
        rows += [(f"Named at {lbl}", lambda x, k=k: pct(x[k]["named"])),
                 (f"Accuracy of named at {lbl}", lambda x, k=k: pct(x[k]["named_acc"])),
                 (f"'Unknown' with true disease in top 3 at {lbl}", lambda x, k=k: pct(x[k]["unknown_top3_has_true"]))]
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(b[s]) for s in S) + " |")
    L += ["", "## False naming (wrong crop named, textures, unsupported plants)", "", f"| Case | {t} | {t0} |", "|---|---|---|"]
    for s, v in o["wrong_crop_claims"].items():
        L.append(f"| {s} val photo, user names a different model crop ({v['pairs']} pairs) | {pct(v['current'])} | "
                 f"{pct(v['old_80'])} |")
    for s, v in o["ood"].items():
        L.append(f"| {s} calib half ({v['n_photos']} photos × every model crop) | {pct(v['current'])} | {pct(v['old_80'])} |")
    L += ["", "Notes:", "- PlantDoc val is small (229 photos); field numbers have wide uncertainty.",
          "- Apple field val shares sources with apple training data, so it overstates phone-photo performance."]
    (RESULTS / "VAL_THRESHOLD_EVALUATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
