"""Stage 5 — Energy-based OSR on the FROZEN Stage-3 model, using the SAME logits as Stage-4 MSP.

    python scripts/osr_energy.py select      # validation only: temperatures, frozen energy threshold
    python scripts/osr_energy.py evaluate    # protocol check, then score the locked test sets once
    python scripts/osr_energy.py report      # results/reports/STAGE_5_ENERGY_OSR.md

E(x) = -T * logsumexp(f(x)/T). Lower = more known. Accept as KNOWN iff E <= tau, else UNKNOWN.
No retraining, no new head; logits come from Stage 4's cache (verified against the manifests).
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_msp import eval_pair, manifest, sha256  # noqa: E402
from src.data.protocol import known_classes, verify_lock  # noqa: E402
from src.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from src.evaluation.osr_metrics import ranking_metrics  # noqa: E402
from src.osr.scores import energy_score, msp_scores  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.reproducibility import environment  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

THR_FILE = "osr_energy_threshold.json"
TEST_LOG = "results/test_runs.log"


def frozen_logits(cfg, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Locked manifest rows + the Stage-4 cached logits, verified to be the same images in the same order."""
    e = cfg["osr_energy"]
    verify_lock(cfg)
    if sha256(resolve(cfg["osr_msp"]["checkpoint"])) != cfg["osr_msp"]["checkpoint_sha256"]:
        raise SystemExit("STOP: baseline checkpoint changed")
    df = manifest(f"data/splits/{split}.csv", split)
    z = np.load(resolve(e[f"logits_{split}"]), allow_pickle=False)
    if list(z["paths"]) != df["image_path"].tolist() or z["logits"].shape != (len(df), 21):
        raise SystemExit(f"STOP: cached {split} logits do not match the locked manifest")
    if not np.isfinite(z["logits"]).all():
        raise SystemExit("STOP: non-finite logits")
    return df, z["logits"].astype(np.float64)


# ----------------------------------------------------------------------------- select (validation only)
def cmd_select(cfg):
    e = cfg["osr_energy"]
    out = resolve(cfg["paths"]["metrics"]) / THR_FILE
    if out.exists():
        raise SystemExit(f"{out} exists — the energy threshold is frozen.")
    val, L = frozen_logits(cfg, "val")
    known = (val["role"] == "known").to_numpy()
    uval = (val["role"] == "unknown_val").to_numpy()
    ms = msp_scores(L)
    correct = ms["pred"][known] == val["class_id"].to_numpy()[known]
    per_t = {}
    for T in e["temperature_grid"]:
        en = energy_score(L, T)
        ek = en[known]
        tau = float(np.quantile(ek, e["target_known_acceptance"], method="higher"))
        q = lambda a: {str(x): float(np.quantile(a, x)) for x in (0.05, 0.25, 0.5, 0.75, 0.95)}  # noqa: E731
        per_t[str(T)] = {
            "temperature": T, "threshold_energy": tau,
            "known_val_acceptance": float((ek <= tau).mean()),
            "misclassified_val_rejected": float((ek[~correct] > tau).mean()),
            "correct_val_rejected": float((ek[correct] > tau).mean()),
            "energy_quantiles_correct": q(ek[correct]), "energy_quantiles_incorrect": q(ek[~correct]),
            "auroc_misclassified_vs_correct_known_val": float(ranking_metrics(ek[correct], ek[~correct])["auroc"]),
            "auroc_known_val_vs_validation_unknown": float(ranking_metrics(ek, en[uval])["auroc"]),
            "validation_unknown_rejected": float((en[uval] > tau).mean())}
    msp_val = ranking_metrics(ms["u"][known], ms["u"][uval])["auroc"]
    t_star = max(e["temperature_grid"], key=lambda T: per_t[str(T)]["auroc_known_val_vs_validation_unknown"])
    T1 = e["temperature_primary"]
    rec = {
        "primary": {"temperature": T1, "threshold_energy": per_t[str(T1)]["threshold_energy"],
                    "selection": "T fixed a priori (Liu et al. 2020 default; no unknown data); threshold = energy "
                                 "accepting 95% of KNOWN validation images (same rule as Stage-4 MSP)"},
        "secondary": {"temperature": t_star, "threshold_energy": per_t[str(t_star)]["threshold_energy"],
                      "selection": "T = argmax AUROC(known validation vs validation-unknown) over the grid "
                                   "(Stage-2 protocol permits validation-unknowns for such settings); same threshold rule",
                      "differs_from_primary": t_star != T1},
        "rule": "accept as known iff E(x) <= threshold_energy, E = -T*logsumexp(logits/T)",
        "per_temperature_validation": per_t,
        "msp_validation_reference": {"auroc_known_val_vs_validation_unknown": float(msp_val),
                                     "threshold_msp": json.loads(resolve(e["stage4_threshold"]).read_text())["threshold_msp"]},
        "validation": {"n_known": int(known.sum()), "n_validation_unknown": int(uval.sum()),
                       "n_misclassified": int((~correct).sum())},
        "logits_val_sha256": sha256(resolve(e["logits_val"])), "logits_test_sha256": sha256(resolve(e["logits_test"])),
        "checkpoint_sha256": cfg["osr_msp"]["checkpoint_sha256"],
        "val_manifest_sha256": sha256(resolve("data/splits/val.csv")), "test_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(rec, indent=2))
    (resolve(cfg["paths"]["metrics"]) / "energy_validation.json").write_text(json.dumps(
        {"per_temperature": per_t, "primary_temperature": T1, "secondary_temperature": t_star,
         "msp_reference_auroc_known_val_vs_validation_unknown": float(msp_val),
         "note": "validation data only; validation-unknown AUROC used only to choose the SECONDARY temperature"}, indent=2))
    from src.visualization import energy_plots as EP
    figs = resolve(cfg["paths"]["figures"]) / "osr"
    EP.validation_distribution(energy_score(L, T1)[known], correct, energy_score(L, T1)[uval],
                               rec["primary"]["threshold_energy"], T1, figs / "energy_validation_distribution.png")
    EP.temperature_curve(per_t, msp_val, figs / "energy_temperature_validation.png")
    for T, r in per_t.items():
        print(f"T={T:>5}: tau={r['threshold_energy']:9.4f} acc={r['known_val_acceptance']:.4f} "
              f"misclass_rej={r['misclassified_val_rejected']:.4f} AUROC(kval vs val-unk)={r['auroc_known_val_vs_validation_unknown']:.4f}")
    print(f"MSP reference AUROC(kval vs val-unk) = {msp_val:.4f}; primary T={T1}, secondary T*={t_star}")


# ----------------------------------------------------------------------------- protocol check
def protocol_check(cfg, thr) -> dict:
    tr, te, va = (manifest(f"data/splits/{s}.csv", s) for s in ("train", "test", "val"))
    locked_near = list(cfg["osr_protocol"]["unknown_near"])
    import yaml
    cm = yaml.safe_load(resolve("data/splits/class_mapping.yaml").read_text(encoding="utf-8"))
    mapped_near = [c["class_name"] for c in cm["unknown_classes"]["unknown_near (Experiment B, test only)"]]
    log = resolve(TEST_LOG)
    prior = [l for l in log.read_text().splitlines() if "stage5" in l] if log.exists() else []
    unk = set(te.loc[te["known_unknown"] == "unknown", "class_name"]) | set(cfg["osr_protocol"]["unknown_val"])
    checks = {
        "Known classes in training": tr["class_name"].nunique(),
        "Unknown classes in training": len(set(tr["class_name"]) & unk),
        "Hard-unknown classes equal locked class_mapping": "YES" if mapped_near == locked_near else "NO",
        "Threshold and temperature selected from validation": "YES" if not thr["test_used"] else "NO",
        "Final test used for selection": "NO" if all(l.split()[0] > thr["created_utc"] for l in prior) else "YES",
        "Unknown test images in training or validation": "NO" if not (set(te["image_path"]) & (set(tr["image_path"]) | set(va["image_path"]))) else "YES",
        "Baseline checkpoint unchanged": "YES" if sha256(resolve(cfg["osr_msp"]["checkpoint"])) == thr["checkpoint_sha256"] else "NO",
        "Logits identical to Stage-4 MSP logits": "YES" if sha256(resolve(cfg["osr_energy"]["logits_test"])) == thr["logits_test_sha256"] else "NO",
    }
    expected = {"Known classes in training": 21, "Unknown classes in training": 0,
                "Hard-unknown classes equal locked class_mapping": "YES",
                "Threshold and temperature selected from validation": "YES", "Final test used for selection": "NO",
                "Unknown test images in training or validation": "NO", "Baseline checkpoint unchanged": "YES",
                "Logits identical to Stage-4 MSP logits": "YES"}
    print("\nPROTOCOL CHECK")
    for k, v in checks.items():
        print(f"  {k}: {v}" + ("" if v == expected[k] else "   <-- FAIL"))
    if checks != expected:
        raise SystemExit("STOP: protocol check failed")
    return checks


# ----------------------------------------------------------------------------- evaluate (test, once)
def failure_stats(B: pd.DataFrame, accepted: np.ndarray, near_cfg: dict) -> dict:
    a = B[accepted]
    return {"accepted_unknowns": int(accepted.sum()), "rejected_unknowns": int((~accepted).sum()),
            "accepted_predicted_same_crop": float(a["pred_class"].str.split("___").str[0].eq(a["crop"]).mean()) if len(a) else None,
            "accepted_predicted_designated_lookalike": float(np.mean([p in near_cfg[c]["resembles"] for p, c in
                                                                      zip(a["pred_class"], a["class_name"])])) if len(a) else None,
            "accepted_predicted_healthy": float(a["pred_class"].str.endswith("healthy").mean()) if len(a) else None,
            "accepted_by_class": a["class_name"].value_counts().to_dict(),
            "accepted_top_predictions": a.groupby(["class_name", "pred_class"]).size().sort_values(ascending=False)
            .head(10).reset_index().rename(columns={0: "count"}).to_dict(orient="records"),
            "image_statistics_mean": {g: {c: float(B.loc[accepted == (g == "accepted"), c].mean())
                                          for c in ("brightness", "border_mean", "border_std")}
                                      for g in ("accepted", "rejected")}}


def cmd_evaluate(cfg):
    e = cfg["osr_energy"]
    thr = json.loads((resolve(cfg["paths"]["metrics"]) / THR_FILE).read_text())
    checks = protocol_check(cfg, thr)
    classes = known_classes(cfg)
    log = resolve(TEST_LOG)
    with open(log, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage5 energy evaluate "
                f"T={thr['primary']['temperature']} tau={thr['primary']['threshold_energy']!r} "
                f"test_sha256={sha256(resolve('data/splits/test.csv'))}\n")
    te, L = frozen_logits(cfg, "test")
    ms = msp_scores(L)
    lab_name = np.where(te["dataset"] == "plantdoc", te["pv_equivalent"], te["class_name"])
    base = te.assign(msp=ms["msp"], u_msp=ms["u"], pred=ms["pred"], label=te["class_id"].to_numpy(),
                     pred_class=[classes[i] for i in ms["pred"]], label_name=lab_name)
    s4 = json.loads(resolve(e["stage4_results"]).read_text())
    s4thr = json.loads(resolve(e["stage4_threshold"]).read_text())
    t_msp = s4thr["threshold_u"]

    def sets(df):
        K = df[df["role"] == "known"]
        return {
            "B_hard_same_crop": (K, df[df["role"] == "unknown_near"]),
            "A1_unseen_crops": (K, df[df["role"] == "unknown_far_crop"]),
            "A1_unseen_crops_balanced300": (K, df[(df["role"] == "unknown_far_crop") & (df["eval_subset"] == "far_balanced")]),
            "A2_non_plant_textures": (K, df[df["role"] == "unknown_far_texture"]),
            "B_strict_known_sensitivity": (K[K["strict_test"]], df[df["role"] == "unknown_near"]),
            "C_plantdoc_all_unknown": (df[df["role"] == "known_field"], df[df["role"].isin(["unknown_near_field", "unknown_far_field"])]),
            "C_plantdoc_near_unknown": (df[df["role"] == "known_field"], df[df["role"] == "unknown_near_field"]),
            "C_plantdoc_far_unknown": (df[df["role"] == "known_field"], df[df["role"] == "unknown_far_field"]),
        }

    # 1) MSP re-check: same logits must reproduce Stage 4 exactly
    msp_res = {k: eval_pair(kd, ud, t_msp, len(classes)) for k, (kd, ud) in sets(base.assign(u=base["u_msp"])).items()}
    repro = {k: {m: abs(msp_res[k][m] - s4["results"][k][m]) for m in ("auroc", "aupr_out", "fpr_at_95tpr", "unknown_recall")}
             for k in msp_res}
    worst = max(v for d in repro.values() for v in d.values())
    print(f"  Stage-4 MSP reproduced from shared logits: max |diff| = {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("STOP: MSP from cached logits does not reproduce Stage 4")

    # 2) Energy, primary (and secondary if different)
    variants = {"primary": thr["primary"]}
    if thr["secondary"]["differs_from_primary"]:
        variants["secondary"] = thr["secondary"]
    energy_res, dfs = {}, {}
    for name, v in variants.items():
        en = energy_score(L, v["temperature"])
        dfs[name] = base.assign(u=en, energy=en)
        energy_res[name] = {k: eval_pair(kd, ud, v["threshold_energy"], len(classes)) for k, (kd, ud) in sets(dfs[name]).items()}
    df = dfs["primary"]
    tau = thr["primary"]["threshold_energy"]

    # 3) paired bootstrap MSP (A) vs Energy-primary (B) on identical images
    near_cfg = cfg["osr_protocol"]["unknown_near"]
    boot = {}
    S = sets(df)
    for k in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown",
              "C_plantdoc_near_unknown", "C_plantdoc_far_unknown"):
        kd, ud = S[k]
        boot[k] = paired_bootstrap(kd["u_msp"].to_numpy(), ud["u_msp"].to_numpy(), t_msp,
                                   kd["energy"].to_numpy(), ud["energy"].to_numpy(), tau, e["bootstrap_resamples"], e["seed"])
        print(f"  bootstrap {k} done")
    K, U = S["B_hard_same_crop"]
    for c in near_cfg:
        ud = U[U["class_name"] == c]
        boot[f"B_class::{c}"] = paired_bootstrap(K["u_msp"].to_numpy(), ud["u_msp"].to_numpy(), t_msp,
                                                 K["energy"].to_numpy(), ud["energy"].to_numpy(), tau,
                                                 e["bootstrap_resamples"], e["seed"])

    # 4) per-unknown-class comparison (B, A1, C)
    rows = []
    for exp in ("B_hard_same_crop", "A1_unseen_crops", "C_plantdoc_all_unknown"):
        m = {r["class_name"]: r for r in msp_res[exp]["per_unknown_class"]}
        for r in energy_res["primary"][exp]["per_unknown_class"]:
            mm = m[r["class_name"]]
            ud = sets(df)[exp][1]
            ud = ud[ud["class_name"] == r["class_name"]]
            rows.append({"experiment": exp, "unknown_class": r["class_name"], "crop": r["crop"], "images": r["images"],
                         "msp_auroc": mm["auroc_vs_known"], "energy_auroc": r["auroc_vs_known"],
                         "msp_detection": mm["unknown_recall"], "energy_detection": r["unknown_recall"],
                         "msp_unknown_f1": mm["unknown_f1"], "energy_unknown_f1": r["unknown_f1"],
                         "median_msp": mm["median_msp"], "median_energy": float(ud["energy"].median())})
    per_cls = pd.DataFrame(rows)

    # 5) Grape Esca deep dive (+ Corn NLB): vs all known, and vs its look-alike known class only
    deep = {}
    for c in ("Grape___Esca_(Black_Measles)", "Corn_(maize)___Northern_Leaf_Blight"):
        ud = U[U["class_name"] == c]
        look = K[K["class_name"].isin(near_cfg[c]["resembles"])]
        crop_known = K[K["crop"] == ud["crop"].iat[0]]
        deep[c] = {
            "images": int(len(ud)), "resembles": near_cfg[c]["resembles"],
            "predicted_as": ud["pred_class"].value_counts().head(3).to_dict(),
            "msp": {"median": float(ud["msp"].median()), "share_ge_0.999": float((ud["msp"] >= 0.999).mean()),
                    "detection": float((ud["u_msp"] > t_msp).mean()),
                    "auroc_vs_all_known": ranking_metrics(K["u_msp"].to_numpy(), ud["u_msp"].to_numpy())["auroc"],
                    "auroc_vs_lookalike_known": ranking_metrics(look["u_msp"].to_numpy(), ud["u_msp"].to_numpy())["auroc"],
                    "auroc_vs_same_crop_known": ranking_metrics(crop_known["u_msp"].to_numpy(), ud["u_msp"].to_numpy())["auroc"]},
            "energy": {"median": float(ud["energy"].median()), "detection": float((ud["energy"] > tau).mean()),
                       "auroc_vs_all_known": ranking_metrics(K["energy"].to_numpy(), ud["energy"].to_numpy())["auroc"],
                       "auroc_vs_lookalike_known": ranking_metrics(look["energy"].to_numpy(), ud["energy"].to_numpy())["auroc"],
                       "auroc_vs_same_crop_known": ranking_metrics(crop_known["energy"].to_numpy(), ud["energy"].to_numpy())["auroc"]},
            "lookalike_known_test_images": int(len(look)),
            "median_energy_lookalike_known": float(look["energy"].median()),
            "median_energy_all_known": float(K["energy"].median()),
            "bootstrap": boot[f"B_class::{c}"]}

    # 6) failure analysis for Energy-accepted hard unknowns (+ overlap with MSP failures)
    inv = pd.read_csv(resolve(cfg["paths"]["processed"]) / "inventory.csv", low_memory=False, keep_default_na=False,
                      usecols=["rel_path", "border_mean", "border_std", "mean_r", "mean_g", "mean_b"])
    Bd = U.merge(inv, left_on="image_path", right_on="rel_path", how="left")
    for col in ("border_mean", "border_std", "mean_r", "mean_g", "mean_b"):
        Bd[col] = pd.to_numeric(Bd[col], errors="coerce")
    Bd["brightness"] = Bd[["mean_r", "mean_g", "mean_b"]].mean(1)
    acc_e, acc_m = (Bd["energy"] <= tau).to_numpy(), (Bd["u_msp"] <= t_msp).to_numpy()
    fa = failure_stats(Bd, acc_e, near_cfg)
    fa["overlap_with_msp_failures"] = {"both_accept": int((acc_e & acc_m).sum()), "energy_only_accept": int((acc_e & ~acc_m).sum()),
                                       "msp_only_accept": int((~acc_e & acc_m).sum()), "both_reject": int((~acc_e & ~acc_m).sum())}
    rho = spearmanr(df.loc[df["role"].isin(["known", "unknown_near"]), "u_msp"],
                    df.loc[df["role"].isin(["known", "unknown_near"]), "energy"]).correlation
    fa["spearman_msp_u_vs_energy_B_pool"] = float(rho)
    worst_e = Bd[acc_e].nsmallest(8, "energy")[["image_path", "class_name", "pred_class", "msp", "energy"]]
    fa["most_confident_energy_failures"] = worst_e.to_dict(orient="records")

    # 7) predictions file
    is_known = df["known_unknown"] == "known"
    pku = np.where(df["energy"] <= tau, "known", "unknown")
    preds = pd.DataFrame({
        "image_path": df["image_path"], "dataset": df["dataset"], "role": df["role"], "experiments": df["experiments"],
        "true_class": df["label_name"].where(is_known, df["class_name"]), "true_known_unknown": df["known_unknown"],
        "predicted_class": np.where(pku == "known", df["pred_class"], "UNKNOWN"), "argmax_class": df["pred_class"],
        "msp": df["msp"].map(lambda v: f"{v:.17g}"), "energy": df["energy"].map(lambda v: f"{v:.17g}"),
        "temperature": thr["primary"]["temperature"], "energy_threshold": tau,
        "predicted_known_unknown": pku, "correct_osr_decision": pku == df["known_unknown"],
        "msp_decision": np.where(df["u_msp"] <= t_msp, "known", "unknown"),
        "correct_classification": np.where(is_known, (df["pred"] == df["label"]).map(str), ""),
        "strict_test": df["strict_test"]})
    pdir = resolve(cfg["paths"]["results"]) / "predictions"
    preds.to_csv(pdir / "osr_energy_predictions.csv", index=False)

    tdir = resolve(cfg["paths"]["tables"])
    write_table(per_cls.round(6), tdir / "osr_energy_per_unknown_class")
    pk = []
    for cid, d in S["B_hard_same_crop"][0].groupby("label"):
        pk.append({"class_name": classes[cid], "images": len(d), "closed_set_accuracy": float((d["pred"] == d["label"]).mean()),
                   "msp_rejected": int((d["u_msp"] > t_msp).sum()), "energy_rejected": int((d["energy"] > tau).sum())})
    write_table(pd.DataFrame(pk), tdir / "osr_energy_per_known_class")

    env = {**environment(e["seed"], torch.device("cuda" if torch.cuda.is_available() else "cpu")),
           "checkpoint": cfg["osr_msp"]["checkpoint"], "checkpoint_sha256": cfg["osr_msp"]["checkpoint_sha256"],
           "protocol_lock_sha256": sha256(resolve("data/splits/PROTOCOL_LOCK.json")),
           "protocol_lock_created_utc": json.loads(resolve("data/splits/PROTOCOL_LOCK.json").read_text())["created_utc"],
           "logits_test_sha256": thr["logits_test_sha256"], "temperature": thr["primary"]["temperature"],
           "energy_threshold": tau, "note_inference": "no new inference: Stage-4 fp32 logits reused"}
    (resolve(cfg["paths"]["metrics"]) / "energy_environment.json").write_text(json.dumps(env, indent=2))

    out = {"method": "Energy", "threshold": thr, "protocol_check": checks, "stage4_msp_reproduction_max_abs_diff": worst,
           "msp_threshold_msp": s4thr["threshold_msp"], "msp_results": msp_res, "energy_results": energy_res,
           "bootstrap_msp_vs_energy_primary": boot, "per_unknown_class": per_cls.to_dict(orient="records"),
           "deep_dive": deep, "failure_analysis_energy": fa,
           "closed_set_system": s4["closed_set_system"], "environment": env,
           "protocol_identifier": {"lock_created_utc": env["protocol_lock_created_utc"], "lock_sha256": env["protocol_lock_sha256"]},
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (resolve(cfg["paths"]["metrics"]) / "osr_energy_results.json").write_text(json.dumps(out, indent=2, default=float))

    from src.visualization import energy_plots as EP
    figs = resolve(cfg["paths"]["figures"]) / "osr"
    EP.known_vs_unknown(df, tau, figs / "energy_known_vs_unknown.png")
    EP.msp_vs_energy(df, msp_res, energy_res["primary"], per_cls, boot, figs / "msp_vs_energy_comparison.png")
    EP.deep_dive(df, near_cfg, t_msp, tau, deep, figs / "grape_esca_msp_vs_energy.png")
    EP.plantdoc(df, tau, energy_res["primary"], msp_res, figs / "plantdoc_energy_vs_msp.png")
    for k in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown"):
        r, m = energy_res["primary"][k], msp_res[k]
        print(f"{k:24s} ENERGY AUROC {r['auroc']:.4f} AUPR {r['aupr_out']:.4f} FPR95 {r['fpr_at_95tpr']:.4f} "
              f"det {r['unknown_recall']:.4f} F1 {r['unknown_f1']:.4f} acc {r['known_acceptance_rate']:.4f} | "
              f"MSP AUROC {m['auroc']:.4f} det {m['unknown_recall']:.4f}")
    print(per_cls[per_cls["experiment"] == "B_hard_same_crop"][["unknown_class", "msp_auroc", "energy_auroc",
                                                               "msp_detection", "energy_detection"]].to_string(index=False))


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
        from scripts.osr_energy_report import write_report
        write_report(cfg)


if __name__ == "__main__":
    main()
