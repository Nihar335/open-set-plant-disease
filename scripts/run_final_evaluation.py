"""Stage 9 — final benchmark package, built ONLY from the frozen Stage 3-8 results (no training, no test re-scoring).

  python -P scripts/run_final_evaluation.py

Writes results/final/*.csv|md (master + secondary tables, statistical summary, ablation) and
results/figures/final/*.png. Every number is read from results/metrics/*.json written by earlier stages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402
from src.visualization import final_plots as fp  # noqa: E402
from src.visualization.osr_plots import _short  # noqa: E402

OUT = ROOT / "results" / "final"
FIG = ROOT / "results" / "figures" / "final"
PRIMARY = ["MSP", "Energy", "Mahalanobis", "Contrastive prototype", "Contrastive Mahalanobis", "Contrastive Energy*", "Fusion A (mean, primary)"]
NOTES = {"MSP": "Stage 4; Stage-3 classifier", "Energy": "Stage 5; Stage-3 classifier, T = 1",
         "Mahalanobis": "Stage 6; tied Ledoit-Wolf on raw Stage-3 features",
         "Contrastive prototype": "Stage 7 pre-specified primary score", "Contrastive Mahalanobis": "Stage 7; L2-normalised SupCon features",
         "Contrastive Energy*": "Stage 7; SupCon features + NEW linear head (* head effect, see matched-head control); best method chosen post hoc on test",
         "Fusion A (mean, primary)": "Stage 8 pre-registered primary fusion; equal-weight mean of 6 known-val percentiles",
         "Fusion B (max)": "Stage 8 secondary", "Fusion D (val-unknown weights, secondary)": "Stage 8 secondary; weights from validation-unknown classes",
         "Original features + linear head (control)": "Stage 7 POST-HOC head-matched control"}


def F(x, d=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def P(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ci(c, pct=False):
    return f"[{100 * c[0]:.2f}, {100 * c[1]:.2f}]" if pct else f"[{c[0]:.4f}, {c[1]:.4f}]"


def sig(c):
    return "significant" if (c[0] > 0 or c[1] < 0) else "not significant (CI includes 0)"


def main():
    cfg = load_config()
    M = resolve(cfg["paths"]["metrics"])
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    R8 = json.loads((M / "fusion_results.json").read_text())
    R7 = json.loads((M / "contrastive_results.json").read_text())
    HC = json.loads((M / "contrastive_head_control.json").read_text())
    R5 = json.loads((M / "osr_energy_results.json").read_text())
    R6 = json.loads((M / "mahalanobis_results.json").read_text())
    B3 = json.loads((M / "baseline_test_metrics.json").read_text())
    PO = json.loads((M / "probe_diagnostic_original.json").read_text())
    res = dict(R8["results"])
    res["Original features + linear head (control)"] = HC["results"]["original_linear_head::energy_linear_head"]
    near = list(cfg["osr_protocol"]["unknown_near"])

    # ---------------- master table (Experiment B)
    def exp_rows(exp, methods, extra=()):
        rows = []
        for m in methods:
            x = res[m].get(exp)
            if x is None:
                rows.append({"method": m, "note": "not evaluated under this protocol"}); continue
            r = {"method": m, "auroc": x["auroc"], "aupr_out": x["aupr_out"], "fpr_at_95tpr": x["fpr_at_95tpr"], "unknown_f1": x["unknown_f1"],
                 "unknown_detection": x["unknown_recall"], "unknown_precision": x["unknown_precision"], "known_acceptance": x["known_acceptance_rate"],
                 "known_false_rejection": 1 - x["known_acceptance_rate"], "n_known": x["n_known"], "n_unknown": x["n_unknown"]}
            for e in extra:
                r[e] = res[m][exp].get(e)
            r["note"] = NOTES.get(m, "")
            rows.append(r)
        return pd.DataFrame(rows)

    master = exp_rows("B_hard_same_crop", PRIMARY + ["Original features + linear head (control)", "Fusion B (max)", "Fusion D (val-unknown weights, secondary)"])
    master.insert(1, "primary_comparison_set", [m in PRIMARY for m in master["method"]])
    master.to_csv(OUT / "final_master_results.csv", index=False)
    md = ["# Final master results — Experiment B (hard same-crop unseen diseases)", "",
          f"Known test images: {int(master['n_known'][0]):,} · unknown test images: {int(master['n_unknown'][0]):,} (5 diseases). "
          "Each threshold accepts 95% of KNOWN VALIDATION images. Positive class for AUROC/AUPR = unknown.", "",
          "| Method | Hard AUROC | AUPR-Out | FPR@95TPR | Unknown F1 | Unknown detection | Known acceptance | Note |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in master.iterrows():
        name = r["method"] if r["primary_comparison_set"] else f"*{r['method']}* (secondary/control)"
        md.append(f"| {name} | {F(r['auroc'])} | {F(r['aupr_out'])} | {P(r['fpr_at_95tpr'])} | {F(r['unknown_f1'])} | {P(r['unknown_detection'])} | "
                  f"{P(r['known_acceptance'])} | {r['note']} |")
    md += ["", f"Closed-set reference (no rejection): Stage-3 ResNet-50 known-test accuracy {P(B3['full_test']['accuracy'])} (full test), "
           f"{P(B3['strict_test']['accuracy'])} (strict test); macro-F1 {F(B3['full_test']['macro_f1'])} / {F(B3['strict_test']['macro_f1'])}. "
           "Closed-set accuracy is not an OSR metric and is not placed in the AUROC columns.", "",
           "\\* Contrastive Energy uses a new linear head; its gain over original Energy is mostly a head effect "
           f"(same head on original features: {F(HC['results']['original_linear_head::energy_linear_head']['B_hard_same_crop']['auroc'])}). "
           "It was identified as best by comparing completed methods on the test set (exploratory)."]
    (OUT / "final_master_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- secondary tables
    allm = PRIMARY + ["Fusion B (max)", "Fusion D (val-unknown weights, secondary)"]
    for exp, fname in (("A1_unseen_crops", "experiment_A1_unseen_crops"), ("A2_non_plant_textures", "experiment_A2_dtd_textures"),
                       ("B_hard_same_crop", "experiment_B_hard_same_crop"), ("B_strict_known_sensitivity", "experiment_B_strict_known_sensitivity"),
                       ("C_plantdoc_all_unknown", "experiment_C_plantdoc_all_unknown"), ("C_plantdoc_near_unknown", "experiment_C_plantdoc_near_unknown"),
                       ("C_plantdoc_far_unknown", "experiment_C_plantdoc_far_unknown")):
        write_table(exp_rows(exp, allm).round(4), OUT / fname)
    dom = pd.DataFrame([{"method": m, **{k: v for k, v in res[m]["C_domain_shift_known_pv_vs_known_plantdoc"].items() if k in
                        ("auroc", "plantvillage_known_acceptance", "plantdoc_known_acceptance")}} for m in allm]).round(4)
    dom = dom.rename(columns={"auroc": "domain_novelty_auroc_pv_known_vs_plantdoc_known"})
    write_table(dom, OUT / "experiment_C_domain_vs_disease_novelty")

    pc = pd.DataFrame(R8["per_unknown_class"])
    core = ["MSP", "Energy", "Mahalanobis", "Contrastive Energy*", "Fusion A (mean, primary)"]
    pdt = pc[["unknown_class", "images"] + [f"{m}::auroc" for m in core] + [f"{m}::detection" for m in core]].copy()
    pdt["best_auroc_method"] = [core[i] for i in pc[[f"{m}::auroc" for m in core]].to_numpy().argmax(1)]
    write_table(pdt.round(4), OUT / "per_unknown_disease")

    mk = R8["matched_known_acceptance"]
    mrows = []
    for m in allm:
        v, e = mk["val_threshold_point"][m], mk["exact95_known_test"][m]
        mrows.append({"method": m, "known_acceptance_val_threshold": v["known_acceptance"], "detection_val_threshold": v["unknown_detection"],
                      "detection_val_threshold_ci95": v.get("ci95_detection"), "detection_exact95_known_test": e["detection"],
                      "detection_exact95_ci95": e["ci95"], "fusionA_minus_method_exact95": e["fusionA_minus_this"],
                      "fusionA_minus_method_exact95_ci95": e["fusionA_minus_this_ci95"]})
    write_table(pd.DataFrame(mrows), OUT / "matched_known_acceptance")

    # ---------------- statistical comparisons (existing bootstraps only)
    st = []

    def add(name, b, src, metric="auroc"):
        x = b[metric]
        st.append({"comparison (A -> B)": name, "metric": metric, "A": x["A"], "B": x["B"], "diff_B_minus_A": x["diff_B_minus_A"],
                   "ci95_low": x["diff_ci95"][0], "ci95_high": x["diff_ci95"][1], "verdict": sig(x["diff_ci95"]), "source": src})
    add("MSP -> Energy (B)", R5["bootstrap_msp_vs_energy_primary"]["B_hard_same_crop"], "Stage 5")
    add("Energy -> Mahalanobis (B)", R6["bootstrap"]["energy_vs_maha::B_hard_same_crop"], "Stage 6")
    for k in ("proto: original vs contrastive (B)", "maha_norm: original vs contrastive (B)", "original Energy vs contrastive proto (B)",
              "original Energy vs contrastive Energy-head (B)"):
        add(k, R7["bootstrap"][k], "Stage 7")
    add("Energy, linear head: original features -> contrastive features (B) [POST-HOC head-matched]",
        HC["bootstrap"]["energy_linear_head: original-linear-head vs contrastive-linear-head (B)"], "Stage 7 post-hoc control")
    add("MSP, linear head: original features -> contrastive features (B) [POST-HOC head-matched]",
        HC["bootstrap"]["msp_linear_head_u: original-linear-head vs contrastive-linear-head (B)"], "Stage 7 post-hoc control")
    for k, b in R8["bootstrap"].items():
        if "class" not in k:
            add(k, b, "Stage 8")
            add(k, b, "Stage 8", "unknown_recall")
    for m in ("MSP", "Energy", "Mahalanobis", "Contrastive Energy*"):
        e = mk["exact95_known_test"][m]
        st.append({"comparison (A -> B)": f"{m} -> Fusion A (B) at EXACTLY 95% known-test acceptance", "metric": "unknown_detection",
                   "A": e["detection"], "B": mk["exact95_known_test"]["Fusion A (mean, primary)"]["detection"], "diff_B_minus_A": e["fusionA_minus_this"],
                   "ci95_low": e["fusionA_minus_this_ci95"][0], "ci95_high": e["fusionA_minus_this_ci95"][1],
                   "verdict": sig(e["fusionA_minus_this_ci95"]), "source": "Stage 8 matched"})
    for c in near:
        for base in ("Energy", "Mahalanobis", "Contrastive Energy*"):
            add(f"{base} -> Fusion A (B class {_short(c)})", R8["bootstrap"][f"{base} -> Fusion A (B class {c})"], "Stage 8 per disease")
    stdf = pd.DataFrame(st)
    write_table(stdf.round(4), OUT / "statistical_comparisons")

    # ---------------- statistical summary (md)
    def line(r):
        unit = "pp" if r["metric"] != "auroc" else ""
        k = 100 if unit else 1
        return (f"- **{r['comparison (A -> B)']}** ({r['metric']}): {r['A'] * k:.4g}{unit} → {r['B'] * k:.4g}{unit}, "
                f"Δ = {r['diff_B_minus_A'] * k:+.4f}{unit} [{r['ci95_low'] * k:.4f}, {r['ci95_high'] * k:.4f}] — {r['source']}")
    s_sig = stdf[stdf["verdict"] == "significant"]; s_ns = stdf[stdf["verdict"] != "significant"]
    hc = HC["bootstrap"]["energy_linear_head: original-linear-head vs contrastive-linear-head (B)"]["auroc"]
    ss = ["# Statistical summary (final)", "",
          "All intervals are **paired, stratified bootstrap** 95% CIs (1,000 resamples; seed 42), computed on the same test images, "
          "with thresholds fixed at their known-validation values — exactly as produced in Stages 5–8. No new test was introduced in Stage 9. "
          "A difference is called *significant* only when its 95% interval excludes 0. Many comparisons are reported; no multiplicity "
          "correction was applied, so borderline intervals should be read cautiously.", "",
          "## Headline comparisons (Experiment B, AUROC unless stated)", "",
          "| Comparison | Δ [95% CI] | Verdict |", "|---|---:|---|"]
    heads = ["MSP -> Energy (B)", "Energy -> Mahalanobis (B)", "original Energy vs contrastive Energy-head (B)",
             "Energy, linear head: original features -> contrastive features (B) [POST-HOC head-matched]",
             "Energy -> Fusion A (B)", "Contrastive Energy* -> Fusion A (B)", "Contrastive Mahalanobis -> Fusion A (B)",
             "Energy -> Fusion A (B) at EXACTLY 95% known-test acceptance"]
    for h in heads:
        r = stdf[(stdf["comparison (A -> B)"] == h) & (stdf["metric"].isin(["auroc", "unknown_detection"]))].iloc[0]
        k, u = (100, " pp") if r["metric"] == "unknown_detection" else (1, "")
        ss.append(f"| {h}{' (detection)' if u else ''} | {r['diff_B_minus_A'] * k:+.4f}{u} [{r['ci95_low'] * k:.4f}, {r['ci95_high'] * k:.4f}] | {r['verdict']} |")
    ss += ["", "## Significant findings", ""] + [line(r) for _, r in s_sig.iterrows()]
    ss += ["", "## Non-significant findings (95% CI includes 0)", ""] + [line(r) for _, r in s_ns.iterrows()]
    ss += ["", "## Wording rules applied in the final documents", "",
           "- \"Improves\" is used only for significant differences.",
           f"- Fusion A vs Energy: significant on AUROC, but **not significant** in detection at exactly matched 95% known acceptance.",
           f"- Contrastive representation effect with the head held fixed: {hc['diff_B_minus_A']:+.4f} {ci(hc['diff_ci95'])} — significant but small.",
           "- The choice of contrastive Energy as the 'best' method is exploratory: it was made by comparing completed methods on the test set."]
    (OUT / "statistical_summary.md").write_text("\n".join(ss) + "\n", encoding="utf-8")

    # ---------------- ablation
    e_or = res["Energy"]["B_hard_same_crop"]["auroc"]; e_oh = res["Original features + linear head (control)"]["B_hard_same_crop"]["auroc"]
    e_c = res["Contrastive Energy*"]["B_hard_same_crop"]["auroc"]
    b = lambda k: R7["bootstrap"][k]["auroc"]  # noqa: E731
    abl = [
        ("Closed-set ResNet-50 (Stage 3)", "—", f"known-test accuracy {P(B3['full_test']['accuracy'])}; no rejection mechanism", "—"),
        ("+ MSP rejection (Stage 4)", F(res["MSP"]["B_hard_same_crop"]["auroc"]), "threshold on max softmax", "—"),
        ("MSP → Energy (Stage 5)", F(res["Energy"]["B_hard_same_crop"]["auroc"]), "score = −logsumexp(logits), same model",
         f"{R5['bootstrap_msp_vs_energy_primary']['B_hard_same_crop']['auroc']['diff_B_minus_A']:+.4f} {ci(R5['bootstrap_msp_vs_energy_primary']['B_hard_same_crop']['auroc']['diff_ci95'])}"),
        ("Energy → feature-space Mahalanobis (Stage 6)", F(res["Mahalanobis"]["B_hard_same_crop"]["auroc"]), "logits → distance in 2048-d feature space",
         f"{R6['bootstrap']['energy_vs_maha::B_hard_same_crop']['auroc']['diff_B_minus_A']:+.4f} {ci(R6['bootstrap']['energy_vs_maha::B_hard_same_crop']['auroc']['diff_ci95'])}"),
        ("Representation: original → contrastive, prototype score (Stage 7)", f"{F(b('proto: original vs contrastive (B)')['A'])} → {F(b('proto: original vs contrastive (B)')['B'])}",
         "same score, SupCon features", f"{b('proto: original vs contrastive (B)')['diff_B_minus_A']:+.4f} {ci(b('proto: original vs contrastive (B)')['diff_ci95'])}"),
        ("Representation: original → contrastive, Mahalanobis (Stage 7)", f"{F(b('maha_norm: original vs contrastive (B)')['A'])} → {F(b('maha_norm: original vs contrastive (B)')['B'])}",
         "same score, SupCon features", f"{b('maha_norm: original vs contrastive (B)')['diff_B_minus_A']:+.4f} {ci(b('maha_norm: original vs contrastive (B)')['diff_ci95'])}"),
        ("HEAD effect: Stage-3 head → linear head, original features (post-hoc control)", f"{F(e_or)} → {F(e_oh)}",
         "same features, new logistic-regression head", f"{e_oh - e_or:+.4f} (point estimate; no paired CI computed)"),
        ("REPRESENTATION effect: original → contrastive features, same linear head (post-hoc control)", f"{F(e_oh)} → {F(e_c)}",
         "same head, SupCon features", f"{hc['diff_B_minus_A']:+.4f} {ci(hc['diff_ci95'])}"),
        ("Contrastive Energy (total vs Stage-5 Energy)", f"{F(e_or)} → {F(e_c)}", "features AND head changed",
         f"{b('original Energy vs contrastive Energy-head (B)')['diff_B_minus_A']:+.4f} {ci(b('original Energy vs contrastive Energy-head (B)')['diff_ci95'])}"),
        ("Fusion A: 6 signals, equal weights (Stage 8)", F(res["Fusion A (mean, primary)"]["B_hard_same_crop"]["auroc"]), "vs contrastive Energy",
         f"{R8['bootstrap']['Contrastive Energy* -> Fusion A (B)']['auroc']['diff_B_minus_A']:+.4f} {ci(R8['bootstrap']['Contrastive Energy* -> Fusion A (B)']['auroc']['diff_ci95'])}"),
    ]
    ab = ["# Ablation summary (Experiment B hard AUROC)", "", "| Step | Hard AUROC | What changed | Δ [95% CI] |", "|---|---:|---|---:|"]
    ab += [f"| {a} | {b_} | {c} | {d} |" for a, b_, c, d in abl]
    ab += ["", "**Key statement.** Contrastive representation learning provides a measurable improvement under matched scoring "
           f"({hc['diff_B_minus_A']:+.3f} AUROC with the head held fixed), but the larger raw gain observed with contrastive Energy "
           f"({e_c - e_or:+.3f}) is substantially attributable to the changed classifier head ({e_oh - e_or:+.3f})."]
    (OUT / "ablation_summary.md").write_text("\n".join(ab) + "\n", encoding="utf-8")

    # ---------------- figures
    fp.method_comparison([(m.replace(" (mean, primary)", " (equal-weight mean)"), res[m]["B_hard_same_crop"]["auroc"], res[m]["B_hard_same_crop"]["fpr_at_95tpr"],
                           res[m]["B_hard_same_crop"]["unknown_f1"]) for m in PRIMARY], FIG / "final_method_comparison.png")
    fp.per_disease(pc, core, FIG / "final_per_disease_auroc.png")
    pr = pd.read_csv(resolve("results/predictions/osr_fusion_predictions.csv"), low_memory=False)
    esca, look = "Grape___Esca_(Black_Measles)", "Grape___Black_rot"
    kn = pr[pr["role"] == "known"]
    norm = {"unknown": {"contrastive_energy": pr.loc[pr["true_class"] == esca, "norm::contrastive_energy"]},
            "lookalike": {"contrastive_energy": kn.loc[kn["true_class"] == look, "norm::contrastive_energy"]},
            "other_known": {"contrastive_energy": kn.loc[kn["true_class"] != look, "norm::contrastive_energy"]}}
    thr = json.loads((M / "fusion_threshold.json").read_text())["thresholds"]
    ex = pd.DataFrame(R8["examples"]["rows"])
    esca_imgs = ex[(ex["category"] == "accepted by all five") & (ex["true_class"] == esca)]["image_path"].tolist()
    if len(esca_imgs) < 2:
        esca_imgs = sorted(pr.loc[pr["true_class"] == esca, "image_path"])[:2]
    look_imgs = sorted(kn.loc[kn["true_class"] == look, "image_path"])[:2]
    fp.failure_mode(esca_imgs[:2], look_imgs, norm, {"contrastive_energy": thr["single::contrastive_energy"]}, FIG / "final_failure_mode_grape_esca.png")
    fp.architecture(FIG / "final_system_architecture.png")
    fp.protocol({"train": 19126, "val": 4116, "test": B3["full_test"]["n"], "strict": B3["strict_test"]["n"]}, FIG / "final_experimental_protocol.png")

    # ---------------- machine-readable summary used by docs
    summ = {"closed_set": {"full_test_accuracy": B3["full_test"]["accuracy"], "strict_test_accuracy": B3["strict_test"]["accuracy"],
                           "full_test_macro_f1": B3["full_test"]["macro_f1"], "strict_test_macro_f1": B3["strict_test"]["macro_f1"]},
            "experiment_B": {m: {k: res[m]["B_hard_same_crop"][k] for k in ("auroc", "aupr_out", "fpr_at_95tpr", "unknown_f1", "unknown_recall", "known_acceptance_rate")}
                             for m in list(res)},
            "experiment_A1_auroc": {m: res[m]["A1_unseen_crops"]["auroc"] for m in allm},
            "experiment_A2_auroc": {m: res[m]["A2_non_plant_textures"]["auroc"] for m in allm},
            "experiment_C_auroc": {m: res[m]["C_plantdoc_all_unknown"]["auroc"] for m in allm},
            "domain_novelty_auroc": {m: res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]["auroc"] for m in allm},
            "plantdoc_known_acceptance": {m: res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]["plantdoc_known_acceptance"] for m in allm},
            "esca_nearest_centroid_black_rot_share_stage6": R6["grape_esca"]["table"][0]["share_unknown_images_nearest_centroid_is_this"],
            "esca_probe_auroc_leaf_grouped": PO["results"]["Grape___Esca_(Black_Measles) vs Grape___Black_rot"]["auroc_mean"],
            "head_effect": e_oh - e_or, "representation_effect": hc["diff_B_minus_A"], "representation_effect_ci95": hc["diff_ci95"],
            "disagreement_summary": R8["disagreement"]["summary"], "per_disease": R8["per_unknown_class"]}
    (OUT / "final_numbers.json").write_text(json.dumps(summ, indent=2, default=float))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    print("figures", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
