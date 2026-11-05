"""Stage 8 — class-agnostic OSR score fusion and final benchmark.

  python -P scripts/osr_fusion.py select     # KNOWN validation only -> results/metrics/fusion_threshold.json (frozen)
  python -P scripts/osr_fusion.py evaluate   # protocol check, ONE test scoring, all analyses
  python -P scripts/osr_fusion_report.py     # report from saved JSON

Nothing from Stages 2-7 is retrained or overwritten: the six raw signals are recomputed from the frozen
Stage-3 logits cache, the Stage-6 embeddings and the Stage-7 contrastive embeddings with the same TRAIN-only
fits, and checked against the saved thresholds/results of those stages.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.contrastive_osr import Fits, load_rep  # noqa: E402
from scripts.osr_msp import eval_pair, sha256  # noqa: E402
from src.data.protocol import verify_lock  # noqa: E402
from src.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from src.evaluation.osr_metrics import ranking_metrics  # noqa: E402
from src.osr.fusion import (ECDFNormalizer, cluster_weights, complete_linkage_clusters, fuse_max, fuse_mean,  # noqa: E402
                            fuse_weighted, known_quantile_threshold, orient, representatives, simplex_grid,
                            spearman_matrix)
from src.osr.mahalanobis import TiedMahalanobis  # noqa: E402
from src.osr.scores import energy_score, msp_scores  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

THR_FILE, NORM_FILE, RES_FILE = "fusion_threshold.json", "fusion_normalisation.npz", "fusion_results.json"
TEST_LOG = "results/test_runs.log"
FUSIONS = ["fusion_mean", "fusion_max", "fusion_cluster_weighted", "fusion_valunk_weighted"]
NEAR_LOOKALIKE = {"Grape___Esca_(Black_Measles)": "Grape___Black_rot",
                  "Corn_(maize)___Northern_Leaf_Blight": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"}


def metrics_dir(cfg):
    return resolve(cfg["paths"]["metrics"])


def prior_files_hash(cfg) -> dict:
    """sha256 of every pre-Stage-8 metric file (must stay unchanged)."""
    return {p.name: sha256(p) for p in sorted(metrics_dir(cfg).glob("*")) if p.is_file() and not p.name.startswith("fusion")}


# ----------------------------------------------------------------------------- signals
class SignalModel:
    """TRAIN-only fits that produce the six raw signals (identical to Stages 4-7)."""

    def __init__(self, cfg):
        tr, E = load_rep(cfg, "original", "train")
        if set(tr["role"]) != {"known"}:
            raise SystemExit("STOP: training rows must all be known")
        self.maha = TiedMahalanobis(cfg["feature_osr"]["eig_floor_rel"]).fit(E, tr["class_id"].to_numpy(), 21)
        self.cfits = Fits(cfg, "contrastive", linear_head=True)
        self.train_unknown_count = int((tr["role"] != "known").sum())

    def signals(self, cfg, split):
        meta, Eo = load_rep(cfg, "original", split)
        mc, Ec = load_rep(cfg, "contrastive", split)
        assert meta["image_path"].tolist() == mc["image_path"].tolist()
        z = np.load(resolve(cfg["osr_energy"][f"logits_{split}"]))
        assert list(z["paths"]) == meta["image_path"].tolist(), "logits cache not aligned with embeddings"
        L = z["logits"].astype(np.float64)
        ms = msp_scores(L)
        cs = self.cfits.scores(Ec)
        raw = {"orig_msp": ms["msp"], "orig_energy": energy_score(L, 1.0), "orig_mahalanobis": self.maha.score(Eo)[0],
               "contrastive_prototype": cs["prototype_cosine"], "contrastive_mahalanobis": cs["mahalanobis_normalised"],
               "contrastive_energy": cs["energy_linear_head"]}
        extra = {"u_msp": ms["u"], "stage3_pred": ms["pred"], "contrastive_pred": cs["linear_pred"]}
        return meta, raw, extra


def oriented(cfg, raw):
    return {k: orient(raw[k], v[2]) for k, v in cfg["osr_fusion"]["candidate_signals"].items()}


def fuse_all(P, design):
    reps = design["representatives"]
    return {"fusion_mean": fuse_mean(P, reps), "fusion_max": fuse_max(P, reps),
            "fusion_cluster_weighted": fuse_weighted(P, design["cluster_weights"]),
            "fusion_valunk_weighted": fuse_weighted(P, design["valunk_weights"])}


# ----------------------------------------------------------------------------- select
def cmd_select(cfg):
    verify_lock(cfg)
    fo = cfg["osr_fusion"]
    out = metrics_dir(cfg) / THR_FILE
    if out.exists():
        raise SystemExit(f"{out} exists — Stage-8 fusion design is frozen.")
    sm = SignalModel(cfg)
    va, raw, extra = sm.signals(cfg, "val")
    K = (va["role"] == "known").to_numpy()
    UV = (va["role"] == "unknown_val").to_numpy()
    assert set(va.loc[K, "split"]) == {"val"} and K.sum() > 0
    q = fo["target_known_acceptance"]

    # reproduction of earlier frozen thresholds from the recomputed known-validation signals
    M = metrics_dir(cfg)
    prev = {"stage4_msp_u": (json.loads((M / "osr_msp_threshold.json").read_text())["threshold_u"], extra["u_msp"]),
            "stage5_energy": (json.loads((M / "osr_energy_threshold.json").read_text())["primary"]["threshold_energy"], raw["orig_energy"]),
            "stage6_mahalanobis": (json.loads((M / "mahalanobis_threshold.json").read_text())["threshold_mahalanobis"], raw["orig_mahalanobis"]),
            "stage7_contrastive_energy": (json.loads((M / "contrastive_thresholds.json").read_text())["thresholds"]["contrastive::energy_linear_head"], raw["contrastive_energy"])}
    repro = {}
    for k, (saved, s) in prev.items():
        t = known_quantile_threshold(s[K], q)
        repro[k] = {"saved": saved, "recomputed": t, "abs_diff": abs(t - saved)}
        if abs(t - saved) > 1e-6 * max(1.0, abs(saved)):
            raise SystemExit(f"STOP: recomputed {k} threshold {t} != saved {saved}")
    print("reproduced earlier thresholds:", {k: f"{v['abs_diff']:.1e}" for k, v in repro.items()})

    O = oriented(cfg, raw)
    norms = {k: ECDFNormalizer().fit(O[k][K]) for k in O}
    P = {k: norms[k].transform(O[k]) for k in O}

    names, R = spearman_matrix({k: O[k][K] for k in O})
    clusters = complete_linkage_clusters(names, R, fo["redundancy"]["threshold"])
    rep_map = representatives(names, R, clusters)
    reps = [rep_map[tuple(c)] for c in clusters]
    cw = cluster_weights(clusters)

    # D (secondary): protocol-permitted validation-unknown weights over the representatives
    best, best_w = -1.0, None
    for w in simplex_grid(len(reps), fo["oracle"]["grid_step"]):
        f = np.sum([wi * P[r] for wi, r in zip(w, reps)], axis=0)
        a = ranking_metrics(f[K], f[UV])["auroc"]
        if a > best + 1e-12:
            best, best_w = a, w
    vw = {r: float(x) for r, x in zip(reps, best_w) if x > 0}
    vw = {r: x / sum(vw.values()) for r, x in vw.items()}

    design = {"representatives": reps, "cluster_weights": cw, "valunk_weights": vw}
    Fz = fuse_all(P, design)
    thresholds = {f: known_quantile_threshold(Fz[f][K], q) for f in FUSIONS}
    for k in O:
        thresholds[f"single::{k}"] = known_quantile_threshold(P[k][K], q)

    np.savez(resolve(cfg["paths"]["metrics"]) / NORM_FILE, **{k: n.ref_ for k, n in norms.items()})
    rec = {
        "rule": "accept as known iff fused score <= threshold; threshold accepts 95% of KNOWN validation images",
        "normalisation": {"method": fo["normalisation"], "fitted_on": f"known validation images only (role=known, split=val, n={int(K.sum())})",
                          "reference_file": f"results/metrics/{NORM_FILE}", "reference_sha256": sha256(metrics_dir(cfg) / NORM_FILE),
                          "directions": {k: v[2] for k, v in fo["candidate_signals"].items()}},
        "known_validation_spearman": {"names": names, "matrix": R.tolist()},
        "redundancy": {**fo["redundancy"], "clusters": clusters, "representatives": reps},
        "design": design, "primary": fo["primary"], "thresholds": thresholds,
        "data_used": {"fusion_A_B_C": "known validation only (no unknown image, no unknown label)",
                      "fusion_D": "known validation + validation-unknown classes (Tomato mosaic, Cherry x2) — protocol-permitted, secondary",
                      "unknown_test_labels": "not used", "test_images": "not loaded"},
        "fusion_D_validation_unknown_auroc_at_selected_weights": best,
        "validation_unknown_auroc_reported_only": {f: ranking_metrics(Fz[f][K], Fz[f][UV])["auroc"] for f in FUSIONS},
        "validation_unknown_auroc_single_signals_reported_only": {k: ranking_metrics(P[k][K], P[k][UV])["auroc"] for k in P},
        "reproduction_of_earlier_thresholds": repro,
        "checkpoint_sha256": {"stage3": sha256(resolve(cfg["osr_msp"]["checkpoint"])),
                              "contrastive": sha256(resolve(cfg["paths"]["models"]) / cfg["contrastive"]["run_name"] / "best_model.pt")},
        "prior_metric_files_sha256": prior_files_hash(cfg),
        "train_unknown_count": sm.train_unknown_count,
        "test_used": False, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(rec, indent=2))
    print("clusters:", clusters, "| representatives:", reps)
    print("Spearman (known val):\n", pd.DataFrame(R, index=names, columns=names).round(3).to_string())
    print("D weights:", vw, f"(val-unknown AUROC {best:.4f})")
    for f in FUSIONS:
        print(f"  {f:26s} tau {thresholds[f]:.5f}  val-unknown AUROC (reported) {rec['validation_unknown_auroc_reported_only'][f]:.4f}")


# ----------------------------------------------------------------------------- evaluate helpers
def protocol_check(cfg, thr, sm):
    verify_lock(cfg)
    ck = thr["checkpoint_sha256"]
    now_hash = prior_files_hash(cfg)
    changed = [k for k, v in thr["prior_metric_files_sha256"].items() if now_hash.get(k) != v]
    tr_c, _ = load_rep(cfg, "contrastive", "train")
    checks = {
        "Unknown images in any training set": int(sm.train_unknown_count + (tr_c["role"] != "known").sum()),
        "Fusion normalisation / redundancy / weights A-C fitted on known validation only": "YES",
        "Fusion D weights used validation-unknown classes only (protocol-permitted, secondary)": "YES",
        "Thresholds selected from known validation": "YES",
        "Final test used during design/selection": "NO" if not thr["test_used"] else "YES",
        "Stage-3 checkpoint unchanged": "YES" if sha256(resolve(cfg["osr_msp"]["checkpoint"])) == ck["stage3"] else "NO",
        "Contrastive checkpoint unchanged": "YES" if sha256(resolve(cfg["paths"]["models"]) / cfg["contrastive"]["run_name"] / "best_model.pt") == ck["contrastive"] else "NO",
        "Earlier-stage metric files unchanged": "YES" if not changed else f"NO: {changed}",
        "Normalisation reference unchanged": "YES" if sha256(metrics_dir(cfg) / NORM_FILE) == thr["normalisation"]["reference_sha256"] else "NO",
    }
    print("\nPROTOCOL CHECK")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    bad = checks["Unknown images in any training set"] != 0 or any(str(v).startswith("NO") for k, v in checks.items() if k != "Final test used during design/selection") \
        or checks["Final test used during design/selection"] != "NO"
    if bad:
        raise SystemExit("STOP: protocol check failed")
    return checks


def sets_of(d):
    K = d[d["role"] == "known"]
    kf = d[d["role"] == "known_field"]
    return {"B_hard_same_crop": (K, d[d["role"] == "unknown_near"]),
            "A1_unseen_crops": (K, d[d["role"] == "unknown_far_crop"]),
            "A2_non_plant_textures": (K, d[d["role"] == "unknown_far_texture"]),
            "B_strict_known_sensitivity": (K[K["strict_test"]], d[d["role"] == "unknown_near"]),
            "C_plantdoc_all_unknown": (kf, d[d["role"].isin(["unknown_near_field", "unknown_far_field"])]),
            "C_plantdoc_near_unknown": (kf, d[d["role"] == "unknown_near_field"]),
            "C_plantdoc_far_unknown": (kf, d[d["role"] == "unknown_far_field"])}


def matched_detection_bootstrap(methods, K_mask, U_mask, n_boot, seed, q=0.95):
    """Detection at EXACTLY q known-TEST acceptance (threshold re-estimated in every resample; analysis point,
    uses known test labels only). Paired: same resampled images for every method."""
    rng = np.random.default_rng(seed)
    uk = {m: s[K_mask] for m, s in methods.items()}
    uu = {m: s[U_mask] for m, s in methods.items()}
    nk, nu = K_mask.sum(), U_mask.sum()

    def det(a, b):
        t = np.quantile(a, q, method="higher")
        return float((b > t).mean()), float((a <= t).mean())
    point = {m: det(uk[m], uu[m]) for m in methods}
    draws = {m: [] for m in methods}
    for _ in range(n_boot):
        ik, iu = rng.integers(0, nk, nk), rng.integers(0, nu, nu)
        for m in methods:
            draws[m].append(det(uk[m][ik], uu[m][iu])[0])
    out = {}
    for m in methods:
        a = np.array(draws[m])
        out[m] = {"detection": point[m][0], "realised_known_acceptance": point[m][1],
                  "ci95": [float(np.quantile(a, .025)), float(np.quantile(a, .975))]}
    return out, {m: np.array(v) for m, v in draws.items()}


# ----------------------------------------------------------------------------- evaluate
def cmd_evaluate(cfg):
    fo = cfg["osr_fusion"]
    M = metrics_dir(cfg)
    thr = json.loads((M / THR_FILE).read_text())
    if (M / RES_FILE).exists():
        raise SystemExit(f"{M / RES_FILE} exists — Stage-8 test already scored once.")
    sm = SignalModel(cfg)
    checks = protocol_check(cfg, thr, sm)
    with open(resolve(TEST_LOG), "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} stage8 fusion evaluate primary={thr['primary']} "
                 f"thresholds={json.dumps({k: v for k, v in thr['thresholds'].items() if k.startswith('fusion')})} "
                 f"test_sha256={sha256(resolve('data/splits/test.csv'))}\n")

    te, raw, extra = sm.signals(cfg, "test")
    refs = np.load(M / NORM_FILE)
    O = oriented(cfg, raw)
    P = {k: ECDFNormalizer().fit(refs[k]).transform(O[k]) for k in O}
    Fz = fuse_all(P, thr["design"])
    T = thr["thresholds"]
    s4 = json.loads((M / "osr_msp_threshold.json").read_text())["threshold_u"]
    s5 = json.loads((M / "osr_energy_threshold.json").read_text())["primary"]["threshold_energy"]
    s6 = json.loads((M / "mahalanobis_threshold.json").read_text())["threshold_mahalanobis"]
    s7 = json.loads((M / "contrastive_thresholds.json").read_text())["thresholds"]

    # name -> (score u [higher = unknown], threshold, closed-set prediction, label, short name)
    stage3, cpred = extra["stage3_pred"], extra["contrastive_pred"]
    methods = {
        "MSP": (extra["u_msp"], s4, stage3), "Energy": (raw["orig_energy"], s5, stage3),
        "Mahalanobis": (raw["orig_mahalanobis"], s6, stage3),
        "Contrastive Energy*": (raw["contrastive_energy"], s7["contrastive::energy_linear_head"], cpred),
        "Contrastive Mahalanobis": (raw["contrastive_mahalanobis"], s7["contrastive::mahalanobis_normalised"], stage3),
        "Contrastive prototype": (raw["contrastive_prototype"], s7["contrastive::prototype_cosine"], stage3),
        "Fusion A (mean, primary)": (Fz["fusion_mean"], T["fusion_mean"], stage3),
        "Fusion B (max)": (Fz["fusion_max"], T["fusion_max"], stage3),
        "Fusion C (cluster-weighted)": (Fz["fusion_cluster_weighted"], T["fusion_cluster_weighted"], stage3),
        "Fusion D (val-unknown weights, secondary)": (Fz["fusion_valunk_weighted"], T["fusion_valunk_weighted"], stage3),
    }
    PRIMARY = "Fusion A (mean, primary)"
    CORE = ["MSP", "Energy", "Mahalanobis", "Contrastive Energy*", PRIMARY]
    label = te["class_id"].to_numpy()
    res = {}
    for m, (u, t, pred) in methods.items():
        d = te.assign(u=u, pred=pred, label=label)
        if m == "MSP":
            d["msp"] = raw["orig_msp"]
        res[m] = {k: eval_pair(a, b, t, 21) for k, (a, b) in sets_of(d).items()}
        known = d[d["role"] == "known"]; pv = d[d["role"].isin(["known"])]; kf = d[d["role"] == "known_field"]
        res[m]["C_domain_shift_known_pv_vs_known_plantdoc"] = {
            **ranking_metrics(pv["u"].to_numpy(), kf["u"].to_numpy()),
            "plantvillage_known_acceptance": float((known["u"] <= t).mean()), "plantdoc_known_acceptance": float((kf["u"] <= t).mean())}

    # reproduction of earlier stages' saved test metrics
    s4r = json.loads((M / "osr_msp_results.json").read_text())["results"]
    s5r = json.loads((M / "osr_energy_results.json").read_text())["energy_results"]["primary"]
    s6r = json.loads((M / "mahalanobis_results.json").read_text())["mahalanobis_results"]
    s7r = json.loads((M / "contrastive_results.json").read_text())["results"]
    repro = {}
    for m, saved in (("MSP", s4r), ("Energy", s5r), ("Mahalanobis", s6r), ("Contrastive Energy*", s7r["contrastive::energy_linear_head"]),
                     ("Contrastive Mahalanobis", s7r["contrastive::mahalanobis_normalised"]), ("Contrastive prototype", s7r["contrastive::prototype_cosine"])):
        diff = max(abs(res[m][e][k] - saved[e][k]) for e in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown")
                   for k in ("auroc", "unknown_recall", "unknown_f1", "known_acceptance_rate"))
        repro[m] = diff
        if diff > 1e-9:
            raise SystemExit(f"STOP: {m} does not reproduce its saved results (max diff {diff})")
    print("reproduced earlier test metrics (max abs diff):", {k: f"{v:.1e}" for k, v in repro.items()})

    near = list(cfg["osr_protocol"]["unknown_near"])
    roles = te["role"].to_numpy()
    Kb, Ub = roles == "known", roles == "unknown_near"
    nb, seed = fo["bootstrap_resamples"], fo["seed"]

    # ---- paired bootstraps (val thresholds fixed)
    def bp(a, b, kmask, umask):
        ua, ta, _ = methods[a]; ub, tb, _ = methods[b]
        return paired_bootstrap(ua[kmask], ua[umask], ta, ub[kmask], ub[umask], tb, nb, seed)
    boot = {}
    for base in ["Energy", "MSP", "Mahalanobis", "Contrastive Energy*", "Contrastive Mahalanobis"]:
        boot[f"{base} -> Fusion A (B)"] = bp(base, PRIMARY, Kb, Ub)
    for fz in ["Fusion B (max)", "Fusion C (cluster-weighted)", "Fusion D (val-unknown weights, secondary)"]:
        boot[f"Energy -> {fz} (B)"] = bp("Energy", fz, Kb, Ub)
        boot[f"Contrastive Energy* -> {fz} (B)"] = bp("Contrastive Energy*", fz, Kb, Ub)
    for exp, um in (("A1", roles == "unknown_far_crop"), ("A2", roles == "unknown_far_texture")):
        for base in ["Energy", "Contrastive Energy*"]:
            boot[f"{base} -> Fusion A ({exp})"] = bp(base, PRIMARY, Kb, um)
    kf, cu = roles == "known_field", np.isin(roles, ["unknown_near_field", "unknown_far_field"])
    for base in ["Energy", "Contrastive Energy*"]:
        boot[f"{base} -> Fusion A (C)"] = bp(base, PRIMARY, kf, cu)
    cls = te["class_name"].to_numpy()
    for c in near:
        for base in ["Energy", "Mahalanobis", "Contrastive Energy*"]:
            boot[f"{base} -> Fusion A (B class {c})"] = bp(base, PRIMARY, Kb, cls == c)
    print(f"  {len(boot)} paired bootstraps done")

    # ---- matched known-acceptance
    matched, draws = matched_detection_bootstrap({m: methods[m][0] for m in methods}, Kb, Ub, nb, seed)
    for m in methods:
        dd = draws[PRIMARY] - draws[m]
        matched[m]["fusionA_minus_this_ci95"] = [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]
        matched[m]["fusionA_minus_this"] = matched[PRIMARY]["detection"] - matched[m]["detection"]
    val_point = {m: {"known_acceptance": res[m]["B_hard_same_crop"]["known_acceptance_rate"],
                     "known_false_rejection": res[m]["B_hard_same_crop"]["false_rejection_rate"],
                     "unknown_detection": res[m]["B_hard_same_crop"]["unknown_recall"]} for m in methods}
    for m in CORE:
        if m == PRIMARY:
            continue
        b = boot[f"{m} -> Fusion A (B)"]
        val_point[m]["ci95_detection"] = b["unknown_recall"]["A_ci95"]; val_point[m]["ci95_acceptance"] = b["known_acceptance_rate"]["A_ci95"]
        val_point[PRIMARY]["ci95_detection"] = b["unknown_recall"]["B_ci95"]; val_point[PRIMARY]["ci95_acceptance"] = b["known_acceptance_rate"]["B_ci95"]

    # ---- per disease
    rows = []
    for c in near:
        r = {"unknown_class": c, "images": int((cls == c).sum())}
        for m in methods:
            pc = {x["class_name"]: x for x in res[m]["B_hard_same_crop"]["per_unknown_class"]}[c]
            r[f"{m}::auroc"], r[f"{m}::detection"] = pc["auroc_vs_known"], pc["unknown_recall"]
        for k in P:
            r[f"signal::{k}::auroc"] = ranking_metrics(P[k][Kb], P[k][cls == c])["auroc"]
        rows.append(r)
    per_cls = pd.DataFrame(rows)

    # ---- decisions + disagreement on hard unknowns
    dec = {m: methods[m][0] > methods[m][1] for m in methods}   # True = rejected as unknown
    hu = te[Ub].copy()
    for m in CORE:
        hu[f"rej::{m}"] = dec[m][Ub]
    singles = CORE[:-1]

    def pattern(r):
        on = [m for m in CORE if r[f"rej::{m}"]]
        return "+".join(on) if on else "none"
    hu["pattern"] = hu.apply(pattern, axis=1)
    dis = hu.groupby(["pattern"]).size().rename("all_hard_unknowns").to_frame()
    for c in near:
        dis[c] = hu[hu["class_name"] == c].groupby("pattern").size()
    dis = dis.fillna(0).astype(int).sort_values("all_hard_unknowns", ascending=False).reset_index()
    rejS = hu[[f"rej::{m}" for m in singles]].to_numpy()
    rejF = hu[f"rej::{PRIMARY}"].to_numpy()
    summary = {"n": int(len(hu)), "rejected_by_all_five": int((rejS.all(1) & rejF).sum()),
               "rejected_by_none": int((~rejS.any(1) & ~rejF).sum()),
               "fusion_only (no single method rejects)": int((~rejS.any(1) & rejF).sum()),
               "some_single_rejects_but_fusion_accepts": int((rejS.any(1) & ~rejF).sum()),
               "rejected_by_any_single": int(rejS.any(1).sum()), "rejected_by_fusion": int(rejF.sum())}
    for i, m in enumerate(singles):
        only = rejS[:, i] & (rejS.sum(1) == 1)
        summary[f"only {m} (among singles)"] = int(only.sum())
        summary[f"only {m}, fusion also rejects"] = int((only & rejF).sum())
    per_cls_summary = {}
    for c in near:
        mk = (hu["class_name"] == c).to_numpy()
        per_cls_summary[c] = {"n": int(mk.sum()), "any_single": int(rejS[mk].any(1).sum()), "fusion": int(rejF[mk].sum()),
                              **{f"only {m}": int((rejS[mk][:, i] & (rejS[mk].sum(1) == 1)).sum()) for i, m in enumerate(singles)},
                              "fusion_only": int((~rejS[mk].any(1) & rejF[mk]).sum())}

    # ---- deep dives (Esca, NLB)
    def deep(unk, look):
        out = {}
        um, lm = cls == unk, (cls == look) & Kb
        for m in methods:
            u, t, _ = methods[m]
            out[m] = {"auroc_vs_all_known": ranking_metrics(u[Kb], u[um])["auroc"],
                      "auroc_vs_lookalike_only": ranking_metrics(u[lm], u[um])["auroc"],
                      "detection": float((u[um] > t).mean()),
                      "lookalike_false_rejection": float((u[lm] > t).mean()),
                      "share_unknown_below_lookalike_median": float((u[um] <= np.median(u[lm])).mean())}
        sig = {k: {"auroc_vs_all_known": ranking_metrics(P[k][Kb], P[k][um])["auroc"],
                   "auroc_vs_lookalike_only": ranking_metrics(P[k][lm], P[k][um])["auroc"],
                   "mean_percentile_unknown": float(P[k][um].mean()), "mean_percentile_lookalike": float(P[k][lm].mean()),
                   "rejected_at_single_signal_threshold": float((P[k][um] > T[f"single::{k}"]).mean())} for k in P}
        return {"methods": out, "signals": sig, "n_unknown": int(um.sum()), "n_lookalike_test": int(lm.sum()),
                "disagreement": per_cls_summary[unk]}
    deep_res = {u: deep(u, l) for u, l in NEAR_LOOKALIKE.items()}

    # ---- correlations (reported; design used known validation only)
    def sp(mask):
        from src.osr.fusion import spearman_matrix as smx
        n, R = smx({k: O[k][mask] for k in O})
        return {"names": n, "matrix": R.tolist()}
    corr = {"known_validation_design": thr["known_validation_spearman"], "known_test": sp(Kb), "hard_unknown_test": sp(Ub)}

    # ---- oracle (labelled; test labels)
    names6 = list(P)
    best, best_w = -1.0, None
    for w in simplex_grid(len(names6), fo["oracle"]["grid_step"]):
        f = np.sum([wi * P[n] for wi, n in zip(w, names6)], axis=0)
        a = ranking_metrics(f[Kb], f[Ub])["auroc"]
        if a > best:
            best, best_w = a, w
    oracle = {"label": fo["oracle"]["label"], "grid_step": fo["oracle"]["grid_step"], "objective": "Experiment-B TEST AUROC (in-sample, optimistic)",
              "best_auroc": best, "weights": {n: float(w) for n, w in zip(names6, best_w) if w > 0},
              "best_single_signal_per_disease": {c: max(((k, per_cls.set_index('unknown_class').loc[c, f"signal::{k}::auroc"]) for k in P), key=lambda x: x[1])
                                                 for c in near}}
    print(f"  oracle (NOT deployable) B AUROC {best:.4f} weights {oracle['weights']}")

    # ---- reproducible qualitative examples
    rng = np.random.default_rng(seed)
    kq = fo["qualitative_examples_per_category"]
    hu["fusionA"] = Fz["fusion_mean"][Ub]
    hu_sorted = hu.sort_values(["fusionA", "image_path"])
    cats = {"easiest unknown (highest fusion score)": hu_sorted.tail(kq).iloc[::-1],
            "hardest unknown (lowest fusion score)": hu_sorted.head(kq)}
    def pick(mask, name):
        idx = np.flatnonzero(mask.to_numpy())
        sel = rng.choice(idx, size=min(kq, len(idx)), replace=False) if len(idx) else []
        cats[name] = hu.iloc[sorted(sel)]
    allrej = hu[[f"rej::{m}" for m in CORE]].all(axis=1); nonerej = ~hu[[f"rej::{m}" for m in CORE]].any(axis=1)
    s_only = lambda m: hu[f"rej::{m}"] & (hu[[f"rej::{x}" for x in singles]].sum(axis=1) == 1)  # noqa: E731
    pick(allrej, "rejected by all five")
    pick(nonerej, "accepted by all five")
    pick(s_only("Mahalanobis"), "rescued only by Mahalanobis (among single methods)")
    pick(s_only("Energy"), "rescued only by Energy (among single methods)")
    pick(hu[f"rej::{PRIMARY}"] & ~hu[[f"rej::{x}" for x in singles]].any(axis=1), "rescued only by fusion")
    ex = []
    for name, d in cats.items():
        for _, r in d.iterrows():
            ex.append({"category": name, "image_path": r["image_path"], "true_class": r["class_name"],
                       "stage3_pred": cfg["osr_protocol"]["known_classes"][int(stage3[te.index.get_loc(r.name)])],
                       **{f"rejected::{m}": bool(r[f"rej::{m}"]) for m in CORE}, "fusionA": float(r["fusionA"])})
    ex = pd.DataFrame(ex)
    ex_rule = (f"per category: 'easiest'/'hardest' = top/bottom {kq} by Fusion-A score (ties by path); other categories = "
               f"{kq} images drawn with numpy default_rng({seed}) from the category, in that fixed order")

    # ---- write tables / predictions
    tdir = resolve(cfg["paths"]["results"]) / "tables"
    kc = cfg["osr_protocol"]["known_classes"]
    pr = pd.DataFrame({"image_path": te["image_path"], "dataset": te["dataset"], "role": te["role"], "true_class": te["class_name"],
                       "strict_test": te["strict_test"], "stage3_predicted_class": [kc[i] for i in stage3]})
    for k in P:
        pr[f"raw::{k}"] = raw[k]; pr[f"norm::{k}"] = P[k]
    for f in FUSIONS:
        pr[f] = Fz[f]
    pr["fusion_mean_threshold"] = T["fusion_mean"]
    pr["predicted_known_unknown"] = np.where(Fz["fusion_mean"] > T["fusion_mean"], "unknown", "known")
    pr["true_known_unknown"] = np.where(te["role"].isin(["known", "known_field"]), "known", "unknown")
    pr["correct_osr_decision"] = np.where(pr["true_known_unknown"] == "known",
                                          (pr["predicted_known_unknown"] == "known") & (stage3 == label), pr["predicted_known_unknown"] == "unknown")
    for m in CORE:
        pr[f"decision::{m}"] = np.where(dec[m], "unknown", "known")
    pr.to_csv(resolve(cfg["paths"]["results"]) / "predictions" / "osr_fusion_predictions.csv", index=False)

    bench = []
    for m in methods:
        r = {"method": m}
        for e in ("B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown", "B_strict_known_sensitivity"):
            x = res[m][e]
            for k in ("auroc", "aupr_out", "fpr_at_95tpr", "unknown_recall", "unknown_precision", "unknown_f1", "known_acceptance_rate", "false_rejection_rate"):
                r[f"{e}::{k}"] = x[k]
        r["matched95_test_detection"] = matched[m]["detection"]
        bench.append(r)
    bench = pd.DataFrame(bench)
    write_table(bench, tdir / "fusion_vs_all_methods")
    write_table(per_cls, tdir / "fusion_per_unknown_class")
    write_table(dis, tdir / "method_disagreement")
    write_table(ex, tdir / "fusion_qualitative_examples")

    clean = lambda r: {k: v for k, v in r.items() if k != "per_unknown_class"}  # noqa: E731
    out = {"protocol_check": checks, "thresholds": T, "design": thr["design"], "redundancy": thr["redundancy"],
           "reproduction_max_abs_diff": repro,
           "results": {m: {e: (clean(v) if isinstance(v, dict) else v) for e, v in r.items()} for m, r in res.items()},
           "per_unknown_class_records": {m: res[m]["B_hard_same_crop"]["per_unknown_class"] for m in methods},
           "bootstrap": boot, "matched_known_acceptance": {"val_threshold_point": val_point, "exact95_known_test": matched},
           "per_unknown_class": per_cls.to_dict(orient="records"), "disagreement": {"table": dis.to_dict(orient="records"), "summary": summary,
           "per_class": per_cls_summary}, "deep": deep_res, "correlations": corr, "oracle": oracle,
           "examples": {"rule": ex_rule, "rows": ex.to_dict(orient="records")},
           "closed_set": {"stage3_known_test_accuracy": float((stage3[Kb] == label[Kb]).mean())},
           "core_methods": CORE, "primary": PRIMARY, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (M / RES_FILE).write_text(json.dumps(out, indent=2, default=float))

    from src.visualization import fusion_plots as fp
    fdir = resolve(cfg["paths"]["results"]) / "figures" / "final_osr"
    fdir.mkdir(parents=True, exist_ok=True)
    fp.correlation(corr, fdir / "score_correlation_spearman.png")
    fp.benchmark(res, CORE + ["Fusion B (max)", "Fusion C (cluster-weighted)", "Fusion D (val-unknown weights, secondary)"], fdir / "benchmark_auroc.png")
    fp.per_disease(per_cls, CORE, fdir / "per_disease_auroc.png")
    fp.matched(matched, val_point, CORE, fdir / "matched_known_acceptance.png")
    fp.deep(P, te, NEAR_LOOKALIKE, T, fdir / "esca_nlb_normalised_scores.png")
    fp.disagreement(dis, fdir / "method_disagreement.png")
    fp.examples(ex, fdir / "qualitative_examples.png")
    fp.domain_shift(res, CORE, fdir / "plantdoc_domain_vs_disease_novelty.png")

    for m in methods:
        b = res[m]["B_hard_same_crop"]
        print(f"{m:44s} B AUROC {b['auroc']:.4f} det {b['unknown_recall']:.4f} F1 {b['unknown_f1']:.4f} FPR95 {b['fpr_at_95tpr']:.4f} "
              f"acc {b['known_acceptance_rate']:.4f} | A1 {res[m]['A1_unseen_crops']['auroc']:.4f} A2 {res[m]['A2_non_plant_textures']['auroc']:.4f} "
              f"C {res[m]['C_plantdoc_all_unknown']['auroc']:.4f} | matched95 {matched[m]['detection']:.4f}")
    for k, b in boot.items():
        if "class" not in k:
            print(f"  {k:60s} dAUROC {b['auroc']['diff_B_minus_A']:+.4f} {np.round(b['auroc']['diff_ci95'], 4)}  "
                  f"dRecall {b['unknown_recall']['diff_B_minus_A']:+.4f} {np.round(b['unknown_recall']['diff_ci95'], 4)}")
    print(per_cls[["unknown_class"] + [f"{m}::auroc" for m in CORE]].round(4).to_string())
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "evaluate"])
    a = ap.parse_args()
    cfg = load_config()
    {"select": cmd_select, "evaluate": cmd_evaluate}[a.step](cfg)


if __name__ == "__main__":
    main()
