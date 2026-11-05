"""Stage 4 — Maximum Softmax Probability (MSP) open-set baseline on the FROZEN Stage-3 model.

    python scripts/osr_msp.py select      # validation only: choose + freeze the threshold, calibration
    python scripts/osr_msp.py evaluate    # protocol check, then score every locked test set ONCE
    python scripts/osr_msp.py report      # results/reports/STAGE_4_MSP_OSR.md from saved metrics

MSP(x) = max_c softmax(logits(x))_c.  Unknown-ness score u = 1 - MSP (float64, no cancellation).
Decision: accept as KNOWN (predict argmax class) iff MSP >= tau, else UNKNOWN.
No training, no new head, no unknown class in the classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ManifestDataset, build_transforms  # noqa: E402
from src.data.protocol import known_classes, verify_lock  # noqa: E402
from src.evaluation.calibration import reliability  # noqa: E402
from src.evaluation.osr_metrics import (oscr, open_set_macro_f1, per_unknown_class, ranking_metrics,  # noqa: E402
                                        threshold_metrics)
from src.models.classifier import Classifier  # noqa: E402
from src.osr.scores import msp_scores  # noqa: E402
from src.training.trainer import predict  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

THR_FILE = "osr_msp_threshold.json"
TEST_LOG = "results/test_runs.log"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def short(c: str) -> str:
    return (c.replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("Cherry_(including_sour)", "Cherry")
             .replace("___", " · ").replace("Cercospora_leaf_spot Gray_leaf_spot", "Gray leaf spot").replace("_", " ")
             .replace("Spider mites Two-spotted spider mite", "Spider mites").replace("Tomato Yellow Leaf Curl Virus", "YLCV")
             .replace("Leaf blight (Isariopsis Leaf Spot)", "Leaf blight").replace("Haunglongbing (Citrus greening)", "HLB"))


# ----------------------------------------------------------------------------- frozen model
def load_frozen(cfg):
    o = cfg["osr_msp"]
    verify_lock(cfg)
    classes = known_classes(cfg)
    ck_path = resolve(o["checkpoint"])
    if sha256(ck_path) != o["checkpoint_sha256"]:
        raise SystemExit("STOP: baseline checkpoint hash differs from the recorded Stage-3 checkpoint")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if ck["class_names"] != classes or ck["num_classes"] != 21:
        raise SystemExit("STOP: checkpoint classes differ from the locked class mapping")
    s3 = yaml.safe_load(resolve(o["stage3_config"]).read_text(encoding="utf-8"))["baseline"]
    model = Classifier(ck["architecture"], ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tf = build_transforms(s3["training"], s3["augmentation"], train=False)       # exact Stage-3 eval transform
    with torch.no_grad():
        out = model(torch.zeros(1, 3, s3["training"]["image_size"], s3["training"]["image_size"], device=device))
    assert out.shape == (1, 21), f"model must output 21 logits, got {tuple(out.shape)}"
    return model, classes, tf, device, s3, ck


def manifest(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(resolve(path), low_memory=False, keep_default_na=False)
    df = df[(df["split"] == split) & ~df["excluded"].astype(str).str.lower().eq("true")].reset_index(drop=True)
    df["class_id"] = df["class_id"].astype(int)
    df["strict_test"] = df["strict_test"].astype(str).str.lower().eq("true")
    return df


def infer(model, df, tf, device, s3, fp32: bool, cache: Path) -> np.ndarray:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == df["image_path"].tolist():
            print(f"  [cached] logits {cache.name}")
            return z["logits"]
    dl = DataLoader(ManifestDataset(df, tf), batch_size=s3["training"]["batch_size"], shuffle=False,
                    num_workers=s3["training"]["num_workers"], pin_memory=True)
    logits, _, idx = predict(model, dl, device, amp=not fp32, desc=f"logits {cache.stem}")
    assert (idx.numpy() == np.arange(len(df))).all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits.numpy(), paths=np.array(df["image_path"].tolist()))
    return logits.numpy()


# ----------------------------------------------------------------------------- step 1: select (validation only)
def cmd_select(cfg):
    o = cfg["osr_msp"]
    model, classes, tf, device, s3, ck = load_frozen(cfg)
    out = resolve(cfg["paths"]["metrics"]) / THR_FILE
    if out.exists():
        raise SystemExit(f"{out} exists — the threshold is frozen. Delete it deliberately to reselect.")
    val = manifest(s3["dataset"]["val_manifest"], "val")          # known val + validation-unknown
    cache = resolve(cfg["paths"]["results"]) / "predictions" / "cache" / "msp_logits_val.npz"
    sc = msp_scores(infer(model, val, tf, device, s3, o["inference_precision"] == "fp32", cache))
    known = (val["role"] == "known").to_numpy()
    uval = (val["role"] == "unknown_val").to_numpy()
    uk, msp_k = sc["u"][known], sc["msp"][known]
    correct = sc["pred"][known] == val["class_id"].to_numpy()[known]

    def val_point(t_u):
        return {"threshold_msp": float(1 - t_u), "threshold_u": float(t_u),
                "known_val_acceptance": float((uk <= t_u).mean()),
                "known_val_accuracy_among_accepted": float(correct[uk <= t_u].mean()) if (uk <= t_u).any() else None,
                "misclassified_val_rejected": float((uk[~correct] > t_u).mean()) if (~correct).any() else None,
                "correct_val_rejected": float((uk[correct] > t_u).mean()),
                "validation_unknown_rejected": float((sc["u"][uval] > t_u).mean())}

    # Strategy A — fixed grid (characterisation only)
    strat_a = [val_point(1 - m) for m in o["fixed_grid"]]
    # Strategy B — validation percentile: accept target share of KNOWN validation images
    t_b = float(np.quantile(uk, o["target_known_acceptance"], method="higher"))
    # Strategy C — error-aware: maximise (share of misclassified rejected) - (share of correct rejected)
    cand = np.unique(uk)
    j = np.array([(uk[~correct] > t).mean() - (uk[correct] > t).mean() for t in cand])
    t_c = float(cand[int(np.argmax(j))])
    strategies = {"A_fixed_grid": strat_a,
                  "B_validation_percentile": {**val_point(t_b), "target_known_acceptance": o["target_known_acceptance"]},
                  "C_error_aware": {**val_point(t_c), "youden_j": float(j.max()),
                                    "n_misclassified_val": int((~correct).sum())}}
    chosen = {"A": None, "B": t_b, "C": t_c}[o["selected_strategy"]]
    if chosen is None:
        raise SystemExit("strategy A is characterisation only")
    calib = reliability(msp_k, correct, o["ece_bins"])
    q = [0.0, 0.01, 0.05, 0.10, 0.25, 0.5]
    rec = {
        "selected_strategy": o["selected_strategy"], "threshold_u": chosen, "threshold_msp": 1 - chosen,
        "selected_on": "known validation images only (role=known, split=val)",
        "rule": f"accept as known iff MSP >= {1 - chosen!r} (u = 1-MSP <= {chosen!r})",
        "strategies": strategies,
        "validation": {"n_known": int(known.sum()), "n_validation_unknown": int(uval.sum()),
                       "known_val_accuracy": float(correct.mean()), "n_misclassified": int((~correct).sum()),
                       "msp_quantiles_correct": {str(x): float(np.quantile(msp_k[correct], x)) for x in q},
                       "msp_quantiles_incorrect": ({str(x): float(np.quantile(msp_k[~correct], x)) for x in q}
                                                   if (~correct).any() else None),
                       "u_quantiles_known": {str(x): float(np.quantile(uk, x)) for x in [0.5, 0.9, 0.95, 0.99]}},
        "calibration_known_validation": calib,
        "validation_unknown_note": ("validation-unknown images are only REPORTED at the chosen threshold "
                                    "(Stage-2 protocol allows them for method settings); they did not select it"),
        "checkpoint_sha256": o["checkpoint_sha256"],
        "val_manifest_sha256": sha256(resolve(s3["dataset"]["val_manifest"])),
        "test_used": False, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(rec, indent=2))
    pd.DataFrame({"image_path": val["image_path"], "role": val["role"], "class_name": val["class_name"],
                  "class_id": val["class_id"], "pred": sc["pred"], "msp": sc["msp"], "u": sc["u"]}).to_csv(
        resolve(cfg["paths"]["results"]) / "predictions" / "osr_msp_validation_scores.csv", index=False)

    from src.visualization import osr_plots as OP
    figs = resolve(cfg["paths"]["figures"]) / "osr"
    OP.validation_distribution(sc["u"][known], correct, sc["u"][uval], rec, figs / "msp_validation_distribution.png")
    OP.reliability_diagram(calib, msp_k, figs / "msp_reliability_validation.png")
    print(json.dumps({k: rec[k] for k in ("selected_strategy", "threshold_msp", "threshold_u", "rule")}, indent=2))
    for name in ("B_validation_percentile", "C_error_aware"):
        print(name, {k: v for k, v in strategies[name].items()})
    print(f"known-val ECE {calib['ece']:.5f}, mean conf {calib['mean_confidence']:.5f}, acc {calib['accuracy']:.5f}")


# ----------------------------------------------------------------------------- step 2: protocol check
def protocol_check(cfg, thr: dict) -> dict:
    o, sp = cfg["osr_msp"], cfg["paths"]
    tr = manifest("data/splits/train.csv", "train")
    te = manifest("data/splits/test.csv", "test")
    va = manifest("data/splits/val.csv", "val")
    unknown_names = set(cfg["osr_protocol"]["unknown_near"]) | set(cfg["osr_protocol"]["unknown_val"]) | \
        set(te.loc[te["known_unknown"] == "unknown", "class_name"])
    log = resolve(TEST_LOG)
    prior = [l for l in log.read_text().splitlines() if "stage4" in l] if log.exists() else []
    checks = {
        "Known classes in training": tr["class_name"].nunique(),
        "Unknown classes in training": len(set(tr["class_name"]) & unknown_names) + int((tr["known_unknown"] != "known").sum()),
        "Threshold selected from validation": "YES" if thr.get("selected_on", "").startswith("known validation") else "NO",
        # the threshold must have been frozen before the FIRST time any test image was scored in stage 4
        "Final test used for threshold selection": "NO" if (not thr["test_used"] and all(
            l.split()[0] > thr["created_utc"] for l in prior)) else "YES",
        "Unknown test images used during training": "NO" if not (set(te.loc[te["known_unknown"] == "unknown", "image_path"])
                                                                 & set(tr["image_path"])) else "YES",
        "Unknown test images used during threshold selection": "NO" if not (set(te["image_path"]) & set(va["image_path"]))
        and thr["val_manifest_sha256"] == sha256(resolve("data/splits/val.csv")) else "YES",
        "Baseline checkpoint unchanged": "YES" if sha256(resolve(o["checkpoint"])) == o["checkpoint_sha256"]
        == thr["checkpoint_sha256"] else "NO",
    }
    expected = {"Known classes in training": 21, "Unknown classes in training": 0,
                "Threshold selected from validation": "YES", "Final test used for threshold selection": "NO",
                "Unknown test images used during training": "NO",
                "Unknown test images used during threshold selection": "NO", "Baseline checkpoint unchanged": "YES"}
    print("\nPROTOCOL CHECK")
    for k, v in checks.items():
        print(f"  {k}: {v}" + ("" if v == expected[k] else "   <-- FAIL"))
    if checks != expected:
        raise SystemExit("STOP: protocol check failed")
    return checks


# ----------------------------------------------------------------------------- step 3: evaluate (test, once)
def eval_pair(kdf, udf, t, n_classes):
    uk, uu = kdf["u"].to_numpy(), udf["u"].to_numpy()
    pk, yk = kdf["pred"].to_numpy(), kdf["label"].to_numpy()
    res = {**ranking_metrics(uk, uu), "oscr": oscr(uk, pk == yk, uu),
           **threshold_metrics(uk, pk, yk, uu, t, n_classes)}
    res["open_set_macro_f1"] = open_set_macro_f1(pk, uk <= t, yk, udf["pred"].to_numpy(), uu <= t, n_classes)
    res["per_unknown_class"] = per_unknown_class(uk, udf, t)
    return res


def cmd_evaluate(cfg):
    o = cfg["osr_msp"]
    thr = json.loads((resolve(cfg["paths"]["metrics"]) / THR_FILE).read_text())
    checks = protocol_check(cfg, thr)
    model, classes, tf, device, s3, ck = load_frozen(cfg)
    t = thr["threshold_u"]
    log = resolve(TEST_LOG)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage4 msp evaluate "
                f"threshold_msp={thr['threshold_msp']!r} test_sha256={sha256(resolve('data/splits/test.csv'))}\n")

    te = manifest(s3["dataset"]["test_manifest"], "test")
    cache = resolve(cfg["paths"]["results"]) / "predictions" / "cache" / "msp_logits_test.npz"
    sc = msp_scores(infer(model, te, tf, device, s3, o["inference_precision"] == "fp32", cache))
    lab_name = np.where(te["dataset"] == "plantdoc", te["pv_equivalent"], te["class_name"])
    df = te.assign(u=sc["u"], msp=sc["msp"], pred=sc["pred"], label=te["class_id"].to_numpy(),
                   pred_class=[classes[i] for i in sc["pred"]], label_name=lab_name,
                   top3=[";".join(classes[i] for i in r) for r in sc["top3"]],
                   top3_probs=[";".join(f"{p:.6f}" for p in r) for r in sc["top3_probs"]])

    # checkpoint/preprocessing reproduction check against Stage-3 saved predictions (known PV test)
    s3p = pd.read_csv(resolve(cfg["paths"]["results"]) / "predictions" / "baseline_test_predictions.csv")
    kpv = df[(df["role"] == "known")].merge(s3p[["image_path", "pred_class_id", "max_softmax_prob"]], on="image_path")
    repro = {"n": int(len(kpv)), "argmax_mismatches_vs_stage3": int((kpv["pred"] != kpv["pred_class_id"]).sum()),
             "max_abs_msp_diff_vs_stage3_amp": float((kpv["msp"] - kpv["max_softmax_prob"]).abs().max())}
    print(f"  reproduction vs Stage 3: {repro}")

    K = df[df["role"] == "known"]
    sets = {
        "B_hard_same_crop": (K, df[df["role"] == "unknown_near"]),
        "A1_unseen_crops": (K, df[df["role"] == "unknown_far_crop"]),
        "A1_unseen_crops_balanced300": (K, df[(df["role"] == "unknown_far_crop") & (df["eval_subset"] == "far_balanced")]),
        "A2_non_plant_textures": (K, df[df["role"] == "unknown_far_texture"]),
        "B_strict_known_sensitivity": (K[K["strict_test"]], df[df["role"] == "unknown_near"]),
        "C_plantdoc_all_unknown": (df[df["role"] == "known_field"], df[df["role"].isin(["unknown_near_field", "unknown_far_field"])]),
        "C_plantdoc_near_unknown": (df[df["role"] == "known_field"], df[df["role"] == "unknown_near_field"]),
        "C_plantdoc_far_unknown": (df[df["role"] == "known_field"], df[df["role"] == "unknown_far_field"]),
    }
    results = {name: eval_pair(k, u, t, len(classes)) for name, (k, u) in sets.items()}

    # closed-set failure demonstration (no rejection): what does the classifier say for unknowns?
    near_cfg = cfg["osr_protocol"]["unknown_near"]
    summ = []
    for c, d in df[df["known_unknown"] == "unknown"].groupby("class_name"):
        vc = d["pred_class"].value_counts()
        crop = d["crop"].iat[0]
        summ.append({"unknown_class": c, "role": d["role"].iat[0], "dataset": d["dataset"].iat[0], "images": len(d),
                     "most_predicted_known": vc.index[0], "share_most_predicted": vc.iat[0] / len(d),
                     "share_predicted_same_crop": float(d["pred_class"].str.split("___").str[0].eq(crop).mean())
                     if d["dataset"].iat[0] == "plantvillage" else None,
                     "share_predicted_lookalike": float(d["pred_class"].isin(near_cfg[c]["resembles"]).mean())
                     if c in near_cfg else None,
                     "share_predicted_healthy": float(d["pred_class"].str.endswith("healthy").mean()),
                     "mean_msp": float(d["msp"].mean()), "median_msp": float(d["msp"].median()),
                     "share_msp_ge_0.90": float((d["msp"] >= 0.90).mean()),
                     "share_msp_ge_0.99": float((d["msp"] >= 0.99).mean()),
                     "share_msp_ge_0.999": float((d["msp"] >= 0.999).mean())})
    summ = pd.DataFrame(summ)
    rng = np.random.default_rng(o["seed"])
    demo = []
    for c in near_cfg:
        d = df[df["class_name"] == c]
        for i in rng.choice(len(d), o["examples_per_unknown_class"], replace=False):
            r = d.iloc[i]
            demo.append({"image_path": r.image_path, "true_unknown_disease": c, "top1_predicted": r.pred_class,
                         "confidence_msp": r.msp, "top3": r.top3, "top3_probs": r.top3_probs,
                         "msp_decision": "KNOWN" if r.u <= t else "UNKNOWN"})
    demo = pd.DataFrame(demo)

    # per-known-class rejection (PlantVillage known test)
    pk = []
    for cid, d in K.groupby("label"):
        acc = d["u"] <= t
        corr = d["pred"] == d["label"]
        pk.append({"class_name": classes[cid], "images": len(d), "closed_set_accuracy": float(corr.mean()),
                   "accepted": int(acc.sum()), "rejected": int((~acc).sum()), "false_rejection_rate": float((~acc).mean()),
                   "accuracy_among_accepted": float(corr[acc].mean()) if acc.any() else None,
                   "rejected_that_were_misclassified": int((~acc & ~corr).sum()),
                   "median_msp": float(d["msp"].median())})
    pk = pd.DataFrame(pk)

    # failure analysis: accepted (false-known) vs rejected hard unknowns
    inv = pd.read_csv(resolve(cfg["paths"]["processed"]) / "inventory.csv", low_memory=False, keep_default_na=False,
                      usecols=["rel_path", "border_mean", "border_std", "mean_r", "mean_g", "mean_b"])
    B = df[df["role"] == "unknown_near"].merge(inv, left_on="image_path", right_on="rel_path", how="left")
    B["accepted"] = B["u"] <= t
    for col in ("border_mean", "border_std", "mean_r", "mean_g", "mean_b"):
        B[col] = pd.to_numeric(B[col], errors="coerce")
    B["brightness"] = B[["mean_r", "mean_g", "mean_b"]].mean(1)
    fa = {"accepted_unknowns": int(B["accepted"].sum()), "rejected_unknowns": int((~B["accepted"]).sum()),
          "accepted_predicted_same_crop": float(B.loc[B["accepted"], "pred_class"].str.split("___").str[0]
                                                .eq(B.loc[B["accepted"], "crop"]).mean()),
          "accepted_predicted_designated_lookalike": float(np.mean([p in near_cfg[c]["resembles"] for p, c in
                                                                    zip(B.loc[B["accepted"], "pred_class"],
                                                                        B.loc[B["accepted"], "class_name"])])),
          "accepted_predicted_healthy": float(B.loc[B["accepted"], "pred_class"].str.endswith("healthy").mean()),
          "accepted_top_predictions": B.loc[B["accepted"]].groupby(["class_name", "pred_class"]).size()
          .sort_values(ascending=False).head(12).reset_index().rename(columns={0: "count"}).to_dict(orient="records"),
          "image_statistics_mean": {g: {c: float(B.loc[B["accepted"] == (g == "accepted"), c].mean())
                                        for c in ("brightness", "border_mean", "border_std")}
                                    for g in ("accepted", "rejected")}}
    kr = K[K["u"] > t]
    fa["false_rejections_known"] = {"n": int(len(kr)), "of_which_misclassified": int((kr["pred"] != kr["label"]).sum()),
                                    "by_class": kr["class_name"].value_counts().to_dict()}

    # qualitative examples (reproducible rule, hard unknowns)
    Bd = df[df["role"] == "unknown_near"].reset_index(drop=True)
    acc_b = Bd["u"] <= t
    picks = [("highest-confidence failure (accepted unknown)", Bd.loc[acc_b, "msp"].idxmax() if acc_b.any() else None),
             ("lowest-confidence unknown", Bd["msp"].idxmin()),
             ("median-confidence unknown", (Bd["msp"] - Bd["msp"].median()).abs().idxmin()),
             ("random successful rejection (seed)", rng.choice(np.flatnonzero(~acc_b)) if (~acc_b).any() else None)]
    picks += [(f"random from {short(c)} (seed)", int(rng.choice(np.flatnonzero(Bd["class_name"] == c)))) for c in near_cfg]
    ex = pd.DataFrame([{"rule": r, "image_path": Bd.at[i, "image_path"], "true_unknown_disease": Bd.at[i, "class_name"],
                        "classifier_prediction": Bd.at[i, "pred_class"], "classifier_confidence_msp": Bd.at[i, "msp"],
                        "msp_decision": "KNOWN" if Bd.at[i, "u"] <= t else "UNKNOWN",
                        "final_result": "correct rejection" if Bd.at[i, "u"] > t else "FAILURE: accepted as known"}
                       for r, i in picks if i is not None])

    # predictions file (every scored test image)
    is_known = df["known_unknown"] == "known"
    pred_ku = np.where(df["u"] <= t, "known", "unknown")
    preds = pd.DataFrame({
        "image_path": df["image_path"], "dataset": df["dataset"], "role": df["role"], "experiments": df["experiments"],
        "true_class": df["label_name"].where(is_known, df["class_name"]), "true_known_unknown": df["known_unknown"],
        "predicted_class": np.where(pred_ku == "known", df["pred_class"], "UNKNOWN"),
        "argmax_class": df["pred_class"], "predicted_known_unknown": pred_ku,
        "msp": df["msp"].map(lambda v: f"{v:.17g}"), "u_one_minus_msp": df["u"].map(lambda v: f"{v:.6e}"),
        "threshold": thr["threshold_msp"],
        "correct_classification": np.where(is_known, (df["pred"] == df["label"]).map(str), ""),
        "correct_osr_decision": (pred_ku == df["known_unknown"]),
        "top3": df["top3"], "top3_probs": df["top3_probs"], "strict_test": df["strict_test"]})
    pdir = resolve(cfg["paths"]["results"]) / "predictions"
    preds.to_csv(pdir / "osr_msp_predictions.csv", index=False)

    tdir = resolve(cfg["paths"]["tables"])
    write_table(summ, tdir / "osr_closed_set_failure_summary")
    write_table(demo, tdir / "osr_closed_set_failure_examples")
    write_table(pk, tdir / "osr_msp_per_known_class")
    puc = pd.concat([pd.DataFrame(results[e]["per_unknown_class"]).assign(experiment=e)
                     for e in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures",
                               "C_plantdoc_all_unknown")], ignore_index=True)
    write_table(puc, tdir / "osr_msp_per_unknown_class")
    write_table(ex, tdir / "osr_msp_unknown_examples")

    out = {"method": "MSP", "threshold": thr, "protocol_check": checks, "reproduction_check": repro,
           "checkpoint": o["checkpoint"], "checkpoint_sha256": o["checkpoint_sha256"], "seed": o["seed"],
           "inference_precision": o["inference_precision"],
           "evaluation_sets": {k: {"known": int(len(v[0])), "unknown": int(len(v[1])),
                                   "known_roles": sorted(v[0]["role"].unique()),
                                   "unknown_roles": sorted(v[1]["role"].unique())} for k, v in sets.items()},
           "test_manifest_sha256": sha256(resolve("data/splits/test.csv")),
           "closed_set_system": {"known_accuracy_pv": float((K["pred"] == K["label"]).mean()),
                                 "known_accuracy_plantdoc_field": float((sets["C_plantdoc_all_unknown"][0]["pred"]
                                                                         == sets["C_plantdoc_all_unknown"][0]["label"]).mean()),
                                 "share_unknowns_assigned_a_known_class": 1.0},
           "results": results, "closed_set_failure_summary": summ.to_dict(orient="records"),
           "per_known_class": pk.to_dict(orient="records"), "failure_analysis": fa,
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (resolve(cfg["paths"]["metrics"]) / "osr_msp_results.json").write_text(json.dumps(out, indent=2, default=float))

    from src.visualization import osr_plots as OP
    figs = resolve(cfg["paths"]["figures"]) / "osr"
    OP.known_vs_unknown(df, t, figs / "msp_known_vs_unknown.png")
    OP.unknown_by_class(df, t, figs / "msp_unknown_by_class.png")
    OP.roc_pr(df, t, figs / "msp_roc_curve.png", figs / "msp_pr_curve.png", results)
    val_scores = pd.read_csv(pdir / "osr_msp_validation_scores.csv")
    OP.threshold_analysis(val_scores, df, t, figs / "msp_threshold_analysis.png")
    OP.examples_figure(ex, figs / "msp_unknown_examples.png")
    OP.plantdoc_figure(df, t, figs / "plantdoc_msp_known_vs_unknown.png", results)
    for name in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown"):
        r = results[name]
        print(f"{name:28s} AUROC {r['auroc']:.4f} AUPR-Out {r['aupr_out']:.4f} FPR95 {r['fpr_at_95tpr']:.4f} "
              f"OSCR {r['oscr']:.4f} | accept {r['known_acceptance_rate']:.4f} unkRec {r['unknown_recall']:.4f} "
              f"unkPrec {r['unknown_precision']:.4f} unkF1 {r['unknown_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "evaluate", "report"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.step == "select":
        cmd_select(cfg)
    elif a.step == "evaluate":
        cmd_evaluate(cfg)
    else:
        from scripts.osr_report import write_report  # noqa: E402
        write_report(cfg)


if __name__ == "__main__":
    main()
