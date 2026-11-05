"""Multi-crop evaluation of the Stage 11 decision rule (v2 checkpoint), old rule vs new rule. Scores the EVAL split once.

Rules compared on the same cached outputs (stage11/audit/collect_sets.py, centre view = the app's view):
  v2_rule  : conf >= 80% (decision.fuse; m0 photo-fit) — the rule before the 2026-10-23 audit, weather excluded
  v3_rule  : decision.may_name = v2 rule AND crop "diagnosable" (KB.model_coverage) AND no crop conflict (crop_check)
Nothing is fitted here: T / beta / m0 come from models/stage11/calibration.json (validation only).

Sets (stage11/audit/sets.py): lab = PlantVillage test, field = PlantDoc test (held-out web photos),
apple_field = AppleLeaf9 + Kashmir test (same sources as training), dtd_texture and unsupported_plant eval halves.
Simulated claims: known photos with the true crop and with every other model crop; OOD photos with every model crop.

Run:  python -P stage11/audit/collect_sets.py --which known --split eval
      python -P stage11/audit/collect_sets.py --which ood --split eval
      python -P stage11/evaluate_multicrop.py      -> stage11/results/multicrop_eval.json + MULTICROP_EVALUATION.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.decision import THRESHOLD, crop_check, fuse, may_name  # noqa: E402
from stage11.kb import load_kb  # noqa: E402

RESULTS = ROOT / "stage11" / "results"
AUDIT = ROOT / "stage11_local" / "audit"


def ece(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(out)


class Rules:
    def __init__(self, kb, cal):
        self.kb, self.cal = kb, cal
        self.crops = np.array([d["crop"] for d in kb.model_classes])
        self.prof = kb.symptom_targets()
        self.level = {c: kb.model_coverage(c)["level"] for c in set(self.crops)}

    def apply(self, logits, sym, claimed):
        """claimed: array of model-crop names (one per photo). Returns dict of per-photo arrays."""
        mask = np.stack([np.isin(self.crops, self.kb.crop_keys(c)) for c in claimed])
        out = fuse(logits, sym, mask, self.prof, T=self.cal["temperature"], beta=self.cal["beta"], m0=self.cal["m0"])
        conf = np.where(mask, out["conf"], -1)
        top = conf.max(1)
        pred = conf.argmax(1)
        conflict = np.array([crop_check(out["p_full"][i], self.crops, mask[i])["conflict"] for i in range(len(top))])
        v2 = top >= THRESHOLD
        v3 = np.array([may_name(t, self.level[c], k) for t, c, k in zip(top, claimed, conflict)])
        top3 = np.argsort(-conf, 1)[:, :3]
        return {"pred": pred, "top": top, "top3": top3, "v2": v2, "v3": v3, "conflict": conflict,
                "p_crop": out["p_crop"], "fit": out["fit"]}


def known_block(y, r, sel):
    y, pred, top3 = y[sel], r["pred"][sel], r["top3"][sel]
    res = {"n": int(sel.sum()), "top1": float((pred == y).mean()) if sel.any() else None,
           "top3": float((top3 == y[:, None]).any(1).mean()) if sel.any() else None}
    pc = r["p_crop"][sel]
    if sel.any():
        res["ece_p_crop"] = ece(pc.max(1), pc.argmax(1) == y)
        res["false_crop_conflict"] = float(r["conflict"][sel].mean())
    for rule in ("v2", "v3"):
        n = r[rule][sel]
        res[rule] = {"named": float(n.mean()) if sel.any() else None,
                     "named_acc": float((pred[n] == y[n]).mean()) if n.any() else None,
                     "known_rejected": float(1 - n.mean()) if sel.any() else None,
                     "unknown_top3_has_true": float((top3[~n] == y[~n, None]).any(1).mean()) if (~n).any() else None}
    return res


def main():
    kb = load_kb()
    cal = json.loads((ROOT / "models/stage11/calibration.json").read_text())
    R = Rules(kb, cal)
    crops = R.crops
    model_crops = sorted(set(crops))
    K = dict(np.load(AUDIT / "v2_known_eval.npz"))
    O = dict(np.load(AUDIT / "v2_ood_eval.npz"))
    y, src = K["label"], K["source"]
    true_crop = crops[y]
    keys = [d["key"] for d in kb.model_classes]
    r = R.apply(K["lc"], K["sc"], true_crop)

    out = {"created": datetime.now().isoformat(timespec="seconds"), "checkpoint": "models/stage11/best.pt (v2, epoch 10)",
           "calibration": cal, "threshold": THRESHOLD, "by_source": {}, "per_crop": {}, "per_disease_field": {},
           "wrong_crop_claims": {}, "ood": {}}
    for s in ("lab", "field", "apple_field"):
        out["by_source"][s] = known_block(y, r, src == s)
    for c in model_crops:
        out["per_crop"][c] = {"coverage_level": R.level[c]}
        for s in ("lab", "field", "apple_field"):
            sel = (src == s) & (true_crop == c)
            if sel.any():
                out["per_crop"][c][s] = known_block(y, r, sel)
    for s in ("field", "apple_field"):
        for k in sorted(set(y[src == s])):
            out["per_disease_field"][f"{s}:{keys[k]}"] = known_block(y, r, (src == s) & (y == k))

    # wrong crop claims: held-out field photos, each other model crop
    for s in ("lab", "field"):
        v2n = v3n = tot = 0
        for c in model_crops:
            sel = (src == s) & ~np.isin(true_crop, kb.crop_keys(c))
            if not sel.any():
                continue
            rr = R.apply(K["lc"][sel], K["sc"][sel], np.array([c] * sel.sum()))
            v2n += rr["v2"].sum()
            v3n += rr["v3"].sum()
            tot += sel.sum()
        out["wrong_crop_claims"][s] = {"pairs": int(tot), "v2_named": float(v2n / tot), "v3_named": float(v3n / tot)}

    for s in np.unique(O["source"]):
        sel = O["source"] == s
        per = {}
        for c in model_crops:
            rr = R.apply(O["lc"][sel], O["sc"][sel], np.array([c] * sel.sum()))
            per[c] = {"v2_named": float(rr["v2"].mean()), "v3_named": float(rr["v3"].mean())}
        out["ood"][s] = {"n_photos": int(sel.sum()), "per_claimed_crop": per,
                         "v2_named": float(np.mean([v["v2_named"] for v in per.values()])),
                         "v3_named": float(np.mean([v["v3_named"] for v in per.values()]))}

    (RESULTS / "multicrop_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(RESULTS / "test_runs.log", "a", encoding="utf-8") as f:
        f.write(f"{out['created']} evaluate_multicrop.py v3 decision rule (coverage gate + crop check, weather not "
                f"scored) on v2 checkpoint epoch 10; T/beta/m0 unchanged from validation; test split + DTD/D2-dev "
                f"eval halves scored once\n")
    write_md(out)
    print(json.dumps(out["by_source"], indent=1))


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def write_md(o):
    L = ["# Stage 11 multi-crop evaluation (v2 checkpoint; old vs new decision rule)", "",
         f"Generated {o['created']} by `stage11/evaluate_multicrop.py`. Held-out photos scored once; nothing fitted on "
         f"them. Threshold {pct(o['threshold'])} on a heuristic score (not a probability of being correct). "
         "Weather is not part of the score.", "",
         "- **v2 rule**: score ≥ 80% (the rule before this audit).",
         "- **v3 rule**: v2 rule AND the crop is *diagnosable* (healthy and disease classes trained) AND the photo does "
         "not look more like another crop. This is the rule the app now uses.", "",
         "## By image domain (crop stated correctly)", "",
         "| | Lab (PlantVillage) | Field (held-out PlantDoc) | Apple field (same sources as training) |",
         "|---|---|---|---|"]
    b = o["by_source"]
    rows = [("Photos", lambda x: str(x["n"])), ("Top-1 within stated crop", lambda x: pct(x["top1"])),
            ("Top-3 within stated crop", lambda x: pct(x["top3"])),
            ("ECE of within-crop probability", lambda x: pct(x.get("ece_p_crop"))),
            ("False crop-conflict flags", lambda x: pct(x.get("false_crop_conflict"))),
            ("v2: named (≥80%)", lambda x: pct(x["v2"]["named"])), ("v2: accuracy of named", lambda x: pct(x["v2"]["named_acc"])),
            ("v3: named (≥80%)", lambda x: pct(x["v3"]["named"])), ("v3: accuracy of named", lambda x: pct(x["v3"]["named_acc"])),
            ("v3: known disease rejected ('Unknown')", lambda x: pct(x["v3"]["known_rejected"])),
            ("v3: 'Unknown' with true disease in top 3", lambda x: pct(x["v3"]["unknown_top3_has_true"]))]
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(b[s]) for s in ("lab", "field", "apple_field")) + " |")
    L += ["", "## Unknown photos, wrong crop names and unsupported plants (false naming)", "",
          "| Case | v2 named | v3 named |", "|---|---|---|"]
    for s, v in o["wrong_crop_claims"].items():
        L.append(f"| {s} photo, user names a different model crop ({v['pairs']} pairs) | {pct(v['v2_named'])} | "
                 f"{pct(v['v3_named'])} |")
    for s, v in o["ood"].items():
        L.append(f"| {s} ({v['n_photos']} photos × every model crop) | {pct(v['v2_named'])} | {pct(v['v3_named'])} |")
    L += ["", "### False naming per claimed crop (unsupported plants / textures)", "",
          "| Claimed crop | coverage | unsupported plant v2 → v3 | texture v2 → v3 |", "|---|---|---|---|"]
    for c in o["ood"]["unsupported_plant"]["per_claimed_crop"]:
        up, tx = o["ood"]["unsupported_plant"]["per_claimed_crop"][c], o["ood"]["dtd_texture"]["per_claimed_crop"][c]
        L.append(f"| {c} | {o['per_crop'][c]['coverage_level']} | {pct(up['v2_named'])} → {pct(up['v3_named'])} | "
                 f"{pct(tx['v2_named'])} → {pct(tx['v3_named'])} |")
    L += ["", "## Per crop (crop stated correctly)", "",
          "| Crop | coverage | lab n | lab top-1 | lab v3 named | field n | field top-1 | field v3 named | field v3 named acc | "
          "apple-field n | apple-field top-1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in o["per_crop"].items():
        la, fi, ap = v.get("lab"), v.get("field"), v.get("apple_field")
        L.append(f"| {c} | {v['coverage_level']} | {la['n'] if la else 0} | {pct(la['top1']) if la else '–'} | "
                 f"{pct(la['v3']['named']) if la else '–'} | {fi['n'] if fi else 0} | {pct(fi['top1']) if fi else '–'} | "
                 f"{pct(fi['v3']['named']) if fi else '–'} | {pct(fi['v3']['named_acc']) if fi else '–'} | "
                 f"{ap['n'] if ap else 0} | {pct(ap['top1']) if ap else '–'} |")
    L += ["", "## Per disease, field photos", "", "| Source:class | n | top-1 | top-3 | v3 named | v3 named acc |",
          "|---|---|---|---|---|---|"]
    for k, v in o["per_disease_field"].items():
        L.append(f"| {k} | {v['n']} | {pct(v['top1'])} | {pct(v['top3'])} | {pct(v['v3']['named'])} | "
                 f"{pct(v['v3']['named_acc'])} |")
    L += ["", "Notes:", "- PlantDoc field test has 3–12 photos per class: per-disease field numbers are descriptive only.",
          "- Apple field test shares sources (orchards, cameras) with apple training data, so it overstates phone-photo "
          "performance. No independent apple black-rot field photos exist in the evaluation.",
          "- Crops with only one trained class (blueberry, raspberry, soybean, citrus, squash) are never named under v3: "
          "their lab coverage drops by design.",
          "- Unsupported plants: Stage 10 D2 dev photos (iNaturalist, CC BY-NC, internal evaluation only), citrus removed. "
          "Textures: DTD images excluded from Stages 1–9."]
    (RESULTS / "MULTICROP_EVALUATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
