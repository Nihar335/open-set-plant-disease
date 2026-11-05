"""Stage 7 — POST-HOC head-matched control (added after the Stage-7 test results were seen).

Question: is the gain of "Energy/MSP on a linear head over contrastive features" due to the contrastive
REPRESENTATION, or to the HEAD (an L2-regularised logistic regression on frozen, L2-normalised features
instead of the end-to-end Stage-3 head)?  Control: the identical head (same C, same features
normalisation, same TRAIN-only fit) on the ORIGINAL Stage-3 features.

Nothing here changes any Stage-7 threshold, checkpoint or score.  The control's own threshold is chosen
from KNOWN VALIDATION only (95% acceptance) and frozen to disk before the test set is scored.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.contrastive_osr import Fits, load_rep  # noqa: E402
from scripts.osr_msp import eval_pair, sha256  # noqa: E402
from src.data.protocol import verify_lock  # noqa: E402
from src.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from src.evaluation.osr_metrics import ranking_metrics  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

KEYS = ["msp_linear_head_u", "energy_linear_head"]
THR = "contrastive_head_control_thresholds.json"
OUT = "contrastive_head_control.json"


def frame(meta, s, key):
    d = meta.assign(label=meta["class_id"].to_numpy(), pred=s["linear_pred"], u=s[key])
    if key == "msp_linear_head_u":
        d["msp"] = s["msp_linear_head"]
    return d


def sets(d):
    K = d[d["role"] == "known"]
    kf = d[d["role"] == "known_field"]
    return {"B_hard_same_crop": (K, d[d["role"] == "unknown_near"]),
            "A1_unseen_crops": (K, d[d["role"] == "unknown_far_crop"]),
            "A2_non_plant_textures": (K, d[d["role"] == "unknown_far_texture"]),
            "B_strict_known_sensitivity": (K[K["strict_test"]], d[d["role"] == "unknown_near"]),
            "C_plantdoc_all_unknown": (kf, d[d["role"].isin(["unknown_near_field", "unknown_far_field"])]),
            "C_plantdoc_near_unknown": (kf, d[d["role"] == "unknown_near_field"]),
            "C_plantdoc_far_unknown": (kf, d[d["role"] == "unknown_far_field"])}


def main():
    cfg = load_config()
    verify_lock(cfg)
    co = cfg["contrastive"]
    mdir = resolve(cfg["paths"]["metrics"])
    thr_path, out_path = mdir / THR, mdir / OUT
    stage7_thr = json.loads((mdir / "contrastive_thresholds.json").read_text())
    f_o = Fits(cfg, "original", linear_head=True)

    # ---- 1. validation-only threshold, frozen to disk
    if not thr_path.exists():
        va, Ev = load_rep(cfg, "original", "val")
        s = f_o.scores(Ev)
        k = (va["role"] == "known").to_numpy(); uv = (va["role"] == "unknown_val").to_numpy()
        y = va["class_id"].to_numpy()[k]
        rec = {"note": "POST-HOC control added after Stage-7 test results were seen; threshold from known validation only",
               "head": f"LogisticRegression(C={co['linear_head']['C']}) on L2-normalised ORIGINAL Stage-3 features, TRAIN only",
               "thresholds": {}, "validation_unknown_reported_only": {},
               "known_validation_linear_head_accuracy": float((s["linear_pred"][k] == y).mean()),
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test_used": False}
        for key in KEYS:
            t = float(np.quantile(s[key][k], co["target_known_acceptance"], method="higher"))
            rec["thresholds"][f"original::{key}"] = t
            rec["validation_unknown_reported_only"][f"original::{key}"] = ranking_metrics(s[key][k], s[key][uv])["auroc"]
        thr_path.write_text(json.dumps(rec, indent=2))
        print("thresholds frozen:", json.dumps(rec["thresholds"]))
    rec = json.loads(thr_path.read_text())
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — control already evaluated once.")

    # ---- 2. score test once
    with open(resolve("results/test_runs.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage7 POST-HOC head-matched control evaluate "
                 f"thresholds={json.dumps(rec['thresholds'])} test_sha256={sha256(resolve('data/splits/test.csv'))}\n")
    te_o, Eo = load_rep(cfg, "original", "test")
    te_c, Ec = load_rep(cfg, "contrastive", "test")
    assert (te_o["image_path"].to_numpy() == te_c["image_path"].to_numpy()).all()
    f_c = Fits(cfg, "contrastive", linear_head=True)
    so, sc = f_o.scores(Eo), f_c.scores(Ec)
    known = (te_o["role"] == "known").to_numpy()
    y = te_o["class_id"].to_numpy()
    res = {"note": rec["note"], "head": rec["head"], "thresholds": rec["thresholds"],
           "known_test_accuracy": {"original_linear_head": float((so["linear_pred"][known] == y[known]).mean()),
                                   "contrastive_linear_head": float((sc["linear_pred"][known] == y[known]).mean())},
           "results": {}, "bootstrap": {}, "per_disease": []}
    T7 = stage7_thr["thresholds"]
    for key in KEYS:
        do, dc = frame(te_o, so, key), frame(te_c, sc, key)
        res["results"][f"original_linear_head::{key}"] = {k: eval_pair(a, b, rec["thresholds"][f"original::{key}"], 21)
                                                          for k, (a, b) in sets(do).items()}
        (ko, uo), (kc, uc) = sets(do)["B_hard_same_crop"], sets(dc)["B_hard_same_crop"]
        res["bootstrap"][f"{key}: original-linear-head vs contrastive-linear-head (B)"] = paired_bootstrap(
            ko["u"].to_numpy(), uo["u"].to_numpy(), rec["thresholds"][f"original::{key}"],
            kc["u"].to_numpy(), uc["u"].to_numpy(), T7[f"contrastive::{key}"], co["bootstrap_resamples"], co["seed"])
        for cl in cfg["osr_protocol"]["unknown_near"]:
            mo, mc = uo["class_name"] == cl, uc["class_name"] == cl
            res["per_disease"].append({
                "score": key, "unknown_class": cl, "images": int(mo.sum()),
                "original_linear_head_auroc": ranking_metrics(ko["u"].to_numpy(), uo["u"][mo].to_numpy())["auroc"],
                "contrastive_linear_head_auroc": ranking_metrics(kc["u"].to_numpy(), uc["u"][mc].to_numpy())["auroc"],
                "original_linear_head_detection": float((uo["u"][mo] > rec["thresholds"][f"original::{key}"]).mean()),
                "contrastive_linear_head_detection": float((uc["u"][mc] > T7[f"contrastive::{key}"]).mean())})
    out_path.write_text(json.dumps(res, indent=2, default=float))
    for k, v in res["results"].items():
        b = v["B_hard_same_crop"]
        print(f"{k:45s} B AUROC {b['auroc']:.4f} det {b['unknown_recall']:.4f} F1 {b['unknown_f1']:.4f} FPR95 {b['fpr_at_95tpr']:.4f}"
              f" | A1 {v['A1_unseen_crops']['auroc']:.4f} A2 {v['A2_non_plant_textures']['auroc']:.4f} C {v['C_plantdoc_all_unknown']['auroc']:.4f}")
    for k, v in res["bootstrap"].items():
        a = v["auroc"]
        print(f"{k}: {a['A']:.4f} -> {a['B']:.4f} diff {a['diff_B_minus_A']:+.4f} CI {a['diff_ci95']}")
    for r in res["per_disease"]:
        print(r)
    print(res["known_test_accuracy"])


if __name__ == "__main__":
    main()
