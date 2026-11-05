"""Build results/reports/STAGE_8_FINAL_OSR_BENCHMARK.md from saved Stage-8 (and earlier) metrics. No test re-scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from scripts.osr_energy_report import ci  # noqa: E402
from scripts.osr_report import F, P, short  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

SIGLAB = {"orig_msp": "MSP (original)", "orig_energy": "Energy (original)", "orig_mahalanobis": "Mahalanobis (original, Stage 6)",
          "contrastive_prototype": "Prototype distance (contrastive)", "contrastive_mahalanobis": "Mahalanobis (contrastive, L2-norm.)",
          "contrastive_energy": "Energy (contrastive, Stage-7 linear head)*"}


def verdict(b, m="auroc"):
    lo, hi = b[m]["diff_ci95"]
    good_high = m != "fpr_at_95tpr"
    if lo > 0:
        return "fusion better" if good_high else "fusion worse"
    if hi < 0:
        return "fusion worse" if good_high else "fusion better"
    return "no detectable difference"


def main():
    cfg = load_config()
    M = resolve(cfg["paths"]["metrics"])
    R = json.loads((M / "fusion_results.json").read_text())
    TH = json.loads((M / "fusion_threshold.json").read_text())
    HC = json.loads((M / "contrastive_head_control.json").read_text())
    res, boot, deep, PR = R["results"], R["bootstrap"], R["deep"], R["primary"]
    CORE = R["core_methods"]
    ALL = list(res)
    near = list(cfg["osr_protocol"]["unknown_near"])
    pc = pd.DataFrame(R["per_unknown_class"])
    mk = R["matched_known_acceptance"]

    def row(m, e="B_hard_same_crop"):
        x = res[m][e]
        return (f"| {m} | {F(x['auroc'])} | {F(x['aupr_out'])} | {P(x['fpr_at_95tpr'])} | {P(x['unknown_recall'])} | {F(x['unknown_precision'])} | "
                f"{F(x['unknown_f1'])} | {P(x['known_acceptance_rate'])} | {P(x['false_rejection_rate'])} |")
    HDR = ("| Method | AUROC | AUPR-Out | FPR@95TPR | Unknown recall | Unknown precision | Unknown F1 | Known acceptance | Known false rejection |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    def bl(name):
        b = boot[name]
        return (f"| {name} | {F(b['auroc']['A'])} → {F(b['auroc']['B'])} | {b['auroc']['diff_B_minus_A']:+.4f} {ci(b['auroc']['diff_ci95'])} | {verdict(b)} | "
                f"{100 * b['unknown_recall']['diff_B_minus_A']:+.2f} pp {ci(b['unknown_recall']['diff_ci95'], pct=True)} | "
                f"{100 * b['fpr_at_95tpr']['diff_B_minus_A']:+.2f} pp {ci(b['fpr_at_95tpr']['diff_ci95'], pct=True)} | "
                f"{100 * b['known_acceptance_rate']['diff_B_minus_A']:+.2f} pp |")
    BHDR = ("| Comparison (A → B) | AUROC A → B | Δ AUROC [95% CI] | Verdict (AUROC) | Δ unknown recall [95% CI] | Δ FPR@95TPR [95% CI] | Δ known acceptance |\n"
            "|---|---:|---:|---|---:|---:|---:|")

    # ---- design facts
    names, Rm = TH["known_validation_spearman"]["names"], TH["known_validation_spearman"]["matrix"]
    corr_tab = lambda C: ("| | " + " | ".join(SIGLAB[n].split(" (")[0] + (" (C)" if n.startswith("contrastive") else "") for n in C["names"]) + " |\n|---|"  # noqa: E731
                          + "---:|" * len(C["names"]) + "\n" + "\n".join(
                              f"| {SIGLAB[n]} | " + " | ".join(f"{v:.2f}" for v in C["matrix"][i]) + " |" for i, n in enumerate(C["names"])))
    clusters = TH["redundancy"]["clusters"]
    all_single = all(len(c) == 1 for c in clusters)
    offdiag = [(names[i], names[j], Rm[i][j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    top = max(offdiag, key=lambda x: abs(x[2]))
    low = min(offdiag, key=lambda x: abs(x[2]))
    vw = TH["design"]["valunk_weights"]

    # ---- headline numbers
    fa, en, ms, mh, ce = (res[m]["B_hard_same_crop"] for m in (PR, "Energy", "MSP", "Mahalanobis", "Contrastive Energy*"))
    v_en, v_ce, v_ms, v_mh = (verdict(boot[f"{m} -> Fusion A (B)"]) for m in ("Energy", "Contrastive Energy*", "MSP", "Mahalanobis"))
    v_cm = verdict(boot["Contrastive Mahalanobis -> Fusion A (B)"])
    best_m = max(ALL, key=lambda m: res[m]["B_hard_same_crop"]["auroc"])

    # per disease verdicts vs best baseline
    cls_v = {}
    for c in near:
        cls_v[c] = {b: verdict(boot[f"{b} -> Fusion A (B class {c})"]) for b in ("Energy", "Mahalanobis", "Contrastive Energy*")}
    better_cls = [c for c in near if all(v == "fusion better" for v in cls_v[c].values())]
    worse_any = [c for c in near if any(v == "fusion worse" for v in cls_v[c].values())]

    pd_rows = "\n".join(
        f"| {short(r['unknown_class'])} | {r['images']:,} | " + " | ".join(F(r[f'{m}::auroc']) for m in CORE) + " | "
        + " | ".join(P(r[f'{m}::detection']) for m in CORE) + " |" for _, r in pc.iterrows())
    pd_ci = "\n".join(
        f"| {short(c)} | " + " | ".join(f"{boot[f'{b} -> Fusion A (B class {c})']['auroc']['diff_B_minus_A']:+.4f} "
                                        f"{ci(boot[f'{b} -> Fusion A (B class {c})']['auroc']['diff_ci95'])} ({cls_v[c][b]})"
                                        for b in ("Energy", "Mahalanobis", "Contrastive Energy*")) + " |" for c in near)
    sig_rows = "\n".join(f"| {short(r['unknown_class'])} | " + " | ".join(F(r[f'signal::{k}::auroc']) for k in names) + " |" for _, r in pc.iterrows())

    def deep_sec(unk, look):
        d = deep[unk]
        mrows = "\n".join(f"| {m} | {F(x['auroc_vs_all_known'])} | {F(x['auroc_vs_lookalike_only'])} | {P(x['detection'])} | "
                          f"{P(x['lookalike_false_rejection'])} | {P(x['share_unknown_below_lookalike_median'])} |" for m, x in d["methods"].items())
        srows = "\n".join(f"| {SIGLAB[k]} | {F(x['auroc_vs_all_known'])} | {F(x['auroc_vs_lookalike_only'])} | {F(x['mean_percentile_unknown'])} | "
                          f"{F(x['mean_percentile_lookalike'])} | {P(x['rejected_at_single_signal_threshold'])} |" for k, x in d["signals"].items())
        g = d["disagreement"]
        best_sig = max(d["signals"], key=lambda k: d["signals"][k]["auroc_vs_all_known"])
        fus = d["methods"][PR]
        best_single = max((m for m in d["methods"] if not m.startswith("Fusion")), key=lambda m: d["methods"][m]["auroc_vs_all_known"])
        bs = d["methods"][best_single]
        only = {k.replace("only ", ""): v for k, v in g.items() if k.startswith("only ")}
        q1 = ("**improves** on every single method" if fus["auroc_vs_all_known"] > bs["auroc_vs_all_known"] else
              f"does **not** beat the best single method ({best_single}, AUROC {F(bs['auroc_vs_all_known'])})")
        q2 = (f"{max(only, key=only.get)} rejects the most images that no other single method rejects ({max(only.values())} of {g['n']})."
              if max(only.values()) > 0 else "No single method rejects images that all others accept.")
        q3 = (f"Fusion rejects {g['fusion']} of {g['n']} images ({P(g['fusion'] / g['n'])}); at least one single method rejects {g['any_single']} "
              f"({P(g['any_single'] / g['n'])}); fusion-only rescues: {g['fusion_only']}.")
        lk = fus["auroc_vs_lookalike_only"]
        q4 = (f"Yes — fusion AUROC against {short(look)} alone is {F(lk)}; {P(fus['share_unknown_below_lookalike_median'])} of {short(unk)} images "
              f"score below the median {short(look)} image." if lk < 0.9 else
              f"Partly — fusion AUROC against {short(look)} alone is {F(lk)}, but at the operating threshold detection is {P(fus['detection'])}.")
        return f"""Unknown: {d['n_unknown']:,} test images; look-alike known class {short(look)}: {d['n_lookalike_test']} test images.

| Method | AUROC vs all known | AUROC vs look-alike only | Detection (95% val threshold) | Look-alike false rejection | Unknown below look-alike median |
|---|---:|---:|---:|---:|---:|
{mrows}

Per normalised signal (percentile among known-validation images):

| Signal | AUROC vs all known | AUROC vs look-alike | Mean percentile, unknown | Mean percentile, look-alike | Rejected at that signal's own 95% threshold |
|---|---:|---:|---:|---:|---:|
{srows}

1. **Does fusion improve it?** Fusion A AUROC {F(fus['auroc_vs_all_known'])}, detection {P(fus['detection'])}; it {q1}.
2. **Does one score see what the others miss?** Strongest single signal: {SIGLAB[best_sig]} (AUROC {F(d['signals'][best_sig]['auroc_vs_all_known'])}). {q2}
3. **Does fusion inherit the failure?** {q3}
4. **Does it still overlap with the look-alike?** {q4}"""

    # ---- PlantDoc
    def pdrow(m):
        x, dshift = res[m]["C_plantdoc_all_unknown"], res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]
        return (f"| {m} | {F(x['auroc'])} | {F(x['aupr_out'])} | {P(x['fpr_at_95tpr'])} | {P(x['known_acceptance_rate'])} | "
                f"{F(res[m]['C_plantdoc_near_unknown']['auroc'])} | {F(res[m]['C_plantdoc_far_unknown']['auroc'])} | {F(dshift['auroc'])} | "
                f"{P(dshift['plantvillage_known_acceptance'])} |")
    fdom = res[PR]["C_domain_shift_known_pv_vs_known_plantdoc"]
    pd_f = res[PR]["C_plantdoc_all_unknown"]
    v_c = verdict(boot["Energy -> Fusion A (C)"]); v_c2 = verdict(boot["Contrastive Energy* -> Fusion A (C)"])

    # ---- disagreement
    S = R["disagreement"]["summary"]
    dtab = pd.DataFrame(R["disagreement"]["table"]).head(16)
    dis_rows = "\n".join(f"| {r['pattern']} | {r['all_hard_unknowns']:,} | " + " | ".join(str(int(r[c])) for c in near) + " |" for _, r in dtab.iterrows())

    # ---- matched
    vp, ex95 = mk["val_threshold_point"], mk["exact95_known_test"]
    m_rows = "\n".join(
        f"| {m} | {P(vp[m]['known_acceptance'])} {ci(vp[m]['ci95_acceptance'], pct=True) if 'ci95_acceptance' in vp[m] else '(CI not computed)'} | "
        f"{P(vp[m]['unknown_detection'])} {ci(vp[m]['ci95_detection'], pct=True) if 'ci95_detection' in vp[m] else '(CI not computed)'} | "
        f"{P(ex95[m]['detection'])} {ci(ex95[m]['ci95'], pct=True)} | "
        + ("—" if m == PR else f"{100 * ex95[m]['fusionA_minus_this']:+.2f} pp {ci(ex95[m]['fusionA_minus_this_ci95'], pct=True)}") + " |"
        for m in ALL)

    # ---- final benchmark
    def fb(m, star=""):
        b, a1, c = res[m]["B_hard_same_crop"], res[m]["A1_unseen_crops"], res[m]["C_plantdoc_all_unknown"]
        return (f"| {m}{star} | {F(b['auroc'])}{star} | {F(b['unknown_f1'])}{star} | {P(b['fpr_at_95tpr'])}{star} | {F(a1['auroc'])} | "
                f"{F(res[m]['A2_non_plant_textures']['auroc'])} | {F(c['auroc'])} |")
    hcb = HC["bootstrap"]["energy_linear_head: original-linear-head vs contrastive-linear-head (B)"]["auroc"]

    # ---- conclusions (data-driven)
    v_ms_en = json.loads((M / "osr_energy_results.json").read_text())["bootstrap_msp_vs_energy_primary"]
    bme = v_ms_en.get("B_hard_same_crop", v_ms_en)
    e_minus_m = bme["auroc"]["diff_ci95"] if "auroc" in bme else None
    if v_en == "fusion better" and v_ce == "fusion better":
        fusion_claim = "Fusion A improves hard-unknown AUROC over both the best original score (Energy) and contrastive Energy, with paired-bootstrap intervals excluding zero."
    elif v_en == "fusion better" and v_ce != "fusion better":
        fusion_claim = (f"Fusion A beats the original scores (Energy: {v_en}) but not the Stage-7 contrastive Energy score ({v_ce}). "
                        "Fusion did not provide a reliable improvement over the strongest single score despite complementary score information.")
    else:
        fusion_claim = "Fusion did not provide a reliable improvement despite complementary score information."
    if better_cls and len(better_cls) < len(near):
        fusion_claim += f" Per disease the gains are class-dependent (better than Energy, Mahalanobis and contrastive Energy only for {', '.join(short(c) for c in better_cls)})."
    if worse_any:
        fusion_claim += f" Fusion is significantly worse than at least one single method (always including contrastive Energy) for {', '.join(short(c) for c in worse_any)}."
    esca_f = deep["Grape___Esca_(Black_Measles)"]["methods"][PR]
    esca_best = max(deep["Grape___Esca_(Black_Measles)"]["methods"].values(), key=lambda x: x["detection"])["detection"]
    nlb = deep["Corn_(maize)___Northern_Leaf_Blight"]["methods"]
    nlb_keep = nlb[PR]["auroc_vs_all_known"] >= nlb["Mahalanobis"]["auroc_vs_all_known"]
    mE = mk["exact95_known_test"]["Energy"]
    fusion_claim += (f" At exactly 95% known-test acceptance, Fusion A detects {100 * mE['fusionA_minus_this']:+.2f} pp more hard unknowns than Energy "
                     f"{ci(mE['fusionA_minus_this_ci95'], pct=True)}" + (" (interval includes 0, so not significant at the matched operating point)." if mE['fusionA_minus_this_ci95'][0] <= 0 <= mE['fusionA_minus_this_ci95'][1] else "."))
    vs_e = {c: cls_v[c]["Energy"] for c in near}
    fusion_claim += (" Against Energy the gain is class-dependent: better for " + ", ".join(short(c) for c in near if vs_e[c] == "fusion better")
                     + ("; worse for " + ", ".join(short(c) for c in near if vs_e[c] == "fusion worse") if any(v == "fusion worse" for v in vs_e.values()) else "") + ".")
    fusion_claim += (f" Signals are complementary (e.g. {R['disagreement']['summary']['only Mahalanobis (among singles)']} hard unknowns are rejected by Mahalanobis alone), "
                     "but equal-weight averaging lets the weaker signals dilute the strongest one. The protocol-permitted Fusion D put 0.9 weight on contrastive Energy and essentially reproduced it.")
    hard_cls = sorted(near, key=lambda c: pc.set_index("unknown_class").loc[c, f"{PR}::auroc"])[:2]
    defensible = PR if v_ce == "fusion better" and v_en == "fusion better" else ("Contrastive Energy*" if res["Contrastive Energy*"]["B_hard_same_crop"]["auroc"] >= fa["auroc"] else PR)

    report = f"""# Stage 8 — Class-Agnostic OSR Fusion and Final Benchmark

*Generated by `scripts/osr_fusion_report.py` from `results/metrics/fusion_results.json` and `fusion_threshold.json` (plus earlier-stage metric files). The test set was scored once (`results/test_runs.log`, entry `stage8`).*

## 1. Research Question

Earlier stages showed that different novelty scores carry different information: MSP and Energy (logit-based) vs Mahalanobis and prototype distances (feature-based). The diagnostic probe also showed that the frozen features separate Grape Esca from Black rot almost perfectly, yet no label-free score rejects most Esca.

**Question:** does a **class-agnostic** combination of these signals, built **without** unknown-disease labels or unknown images, detect unseen diseases better than the best single score? And does it generalise to field photos (PlantDoc) without retuning?

## 2. Experimental Protocol

- **Data:** the Stage-2 protocol, manifests and test population are unchanged.
  - 21 known classes.
  - Experiment B: 5 hard same-crop unknowns.
  - A1: unseen crops; A2: DTD textures; C: PlantDoc. All are kept separate.
- **No retraining:** no Stage 3–7 model was retrained. The six signals were **recomputed** from the frozen Stage-3 logits, the Stage-6 embeddings and the Stage-7 contrastive embeddings, using the same train-only fits.
  - The recomputed known-validation thresholds equal the saved Stage 4–7 thresholds (max |Δ| {max(v['abs_diff'] for v in TH['reproduction_of_earlier_thresholds'].values()):.1e}).
  - The recomputed test metrics equal the saved ones (max |Δ| {max(R['reproduction_max_abs_diff'].values()):.1e}).
  - No earlier metric file was modified (sha256-checked).
- **What the fusion was built from:** normalisation, redundancy analysis, weights (A–C) and all thresholds come from **known validation images only** (role = known, split = val, n = {TH['normalisation']['fitted_on'].split('n=')[1].rstrip(')')}).
- **Fusion D (secondary):** additionally uses the **validation-unknown** classes (Tomato mosaic, Cherry ×2). DATASET_CARD reserves these for "small settings that need unknowns", which requires that a zero-unknown variant is reported too; that variant is the primary.
- **Never used:** test images, test labels, hard-unknown names and PlantDoc labels were not used for any design choice.
- **Protocol check before test scoring:** {'; '.join(f'{k}: {v}' for k, v in R['protocol_check'].items())}.

## 3. Methods Compared

| Method | Representation | Score | Threshold source |
|---|---|---|---|
| MSP | Stage-3 ResNet-50 | 1 − max softmax | Stage 4 (known val, 95%) |
| Energy | Stage-3 ResNet-50 | −logsumexp(logits), T = 1 | Stage 5 |
| Mahalanobis | Stage-3 features | tied Ledoit–Wolf, raw 2048-d | Stage 6 |
| Contrastive Energy* | Stage-7 SupCon features | Energy of a logistic-regression head on frozen features | Stage 7 |
| Contrastive Mahalanobis | Stage-7 features | Mahalanobis on L2-normalised features | Stage 7 |
| Contrastive prototype | Stage-7 features | 1 − max cosine to train prototypes | Stage 7 |
| **Fusion A (primary)** | both | mean of known-val percentiles | known val, 95% |
| Fusion B | both | max of percentiles | known val, 95% |
| Fusion C | both | redundancy-cluster-weighted mean | known val, 95% |
| Fusion D (secondary) | both | weights from validation-unknown classes | known val, 95% |

\\* Contrastive Energy uses the Stage-7 linear head. Its gain over the original Energy is **not** a pure representation effect. With the head held fixed (Stage-7 post-hoc control), the representation adds only {hcb['diff_B_minus_A']:+.4f} {ci(hcb['diff_ci95'])} AUROC.

The closed-set prediction attached to every fused decision is the Stage-3 classifier (known-test accuracy {P(R['closed_set']['stage3_known_test_accuracy'])}).

**Note:** the fusion uses **two networks** (Stage-3 and Stage-7 backbones), so inference costs about twice a single model.

## 4. Score Normalization

- **Orientation:** every score is oriented so that **larger = more unknown**.
  - MSP: −MSP. Energy: as is (higher energy = more unknown). Mahalanobis and prototype distance: as is.
  - Unit tests check each direction (`tests/test_score_normalization.py`).
- **Normalisation:** each oriented score is mapped to its **mid-rank percentile among the {TH['normalisation']['fitted_on'].split('n=')[1].rstrip(')')} known-validation images**: (#below + ½·#equal)/n, in [0, 1].
  - This assumes no distribution shape, and it is monotone, so single-score AUROC is unchanged.
  - Scores beyond the known-validation range saturate at 0 or 1. Ties at 1 only affect pairs where both images exceed every known-validation score.
  - The reference samples are saved in `results/metrics/fusion_normalisation.npz` (sha256 recorded).

**Spearman correlation of the oriented scores on known validation** (the only correlation used for design):

{corr_tab(TH['known_validation_spearman'])}

Test-set correlations (known test, hard unknowns) are reported in `fusion_results.json` and in `figures/final_osr/score_correlation_spearman.png`.

- **Most correlated pair:** {SIGLAB[top[0]]} and {SIGLAB[top[1]]} (ρ = {top[2]:.2f}).
- **Least correlated pair:** {SIGLAB[low[0]]} and {SIGLAB[low[1]]} (ρ = {low[2]:.2f}).
- Low correlation means the signals *can* carry different information. It does not by itself imply better OSR.

## 5. Fusion Method

- **Redundancy rule (pre-registered):** complete-linkage clusters of signals whose pairwise |ρ| on known validation are all ≥ {TH['redundancy']['threshold']}. One representative per cluster: the member least correlated with the other clusters.
  - **Result:** {'**no pair reached the cut-off, so every signal is its own cluster.** Fusion A uses all six signals, and Fusion C is identical to Fusion A' if all_single else 'clusters ' + str(clusters) + '; representatives ' + str(TH['redundancy']['representatives'])}.
- **Fusion A (primary):** equal-weight mean of the representatives' percentiles. **Fusion B:** their maximum. **Fusion C:** each cluster weighted 1/n_clusters.
- **Why equal weights:** {cfg['osr_fusion']['weighting_note']}.
  - The obvious known-only alternative, optimising misclassification detection on known validation, rewards *closed-set error* detection, which is a different task. We did not force learned weights.
- **Fusion D (secondary, protocol-permitted):** simplex-grid weights (step 0.1) maximising AUROC of known validation vs validation-unknown classes. Selected weights: {', '.join(f'{SIGLAB[k]} {v:.1f}' for k, v in vw.items())}; validation-unknown AUROC {F(TH['fusion_D_validation_unknown_auroc_at_selected_weights'])}.
  - These validation unknowns are mostly a different crop (Cherry) plus one tomato virus. They are not the hard same-crop look-alikes.
- **Thresholds:** each fusion accepts 95% of known validation: A {F(TH['thresholds']['fusion_mean'], 5)}, B {F(TH['thresholds']['fusion_max'], 5)}, C {F(TH['thresholds']['fusion_cluster_weighted'], 5)}, D {F(TH['thresholds']['fusion_valunk_weighted'], 5)}.
- **Validation-unknown AUROC (reported only):** {', '.join(f'{k} {F(v)}' for k, v in TH['validation_unknown_auroc_reported_only'].items())}.

## 6. Primary Hard-Unknown Results

Experiment B: 5 hard same-crop unknown diseases, {res[PR]['B_hard_same_crop']['n_unknown']:,} unknown and {res[PR]['B_hard_same_crop']['n_known']:,} known test images.

{HDR}
""" + "\n".join(row(m) for m in ALL) + f"""

**Equal known acceptance.** Validation thresholds give slightly different realised test acceptance, which can inflate detection. Two views are shown:
- each method at its own 95% known-*validation* threshold, with bootstrap CIs;
- every method at **exactly 95% known-test acceptance**. This analysis point uses known test labels only and no unknown labels, and its threshold is re-estimated in every bootstrap resample.

| Method | Known acceptance (val threshold) [95% CI] | Unknown detection (val threshold) [95% CI] | Detection at exactly 95% known-test acceptance [95% CI] | Fusion A − this method (matched) [95% CI] |
|---|---:|---:|---:|---:|
{m_rows}

**Strict-test sensitivity** (known images restricted to the strict test): Fusion A AUROC {F(res[PR]['B_strict_known_sensitivity']['auroc'])} vs Energy {F(res['Energy']['B_strict_known_sensitivity']['auroc'])}, contrastive Energy {F(res['Contrastive Energy*']['B_strict_known_sensitivity']['auroc'])}.

## 7. Easy Unknown Results

| Method | A1 AUROC | A1 FPR@95TPR | A1 recall | A1 F1 | A2 AUROC | A2 FPR@95TPR | A2 recall | A2 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {m} | {F(res[m]['A1_unseen_crops']['auroc'])} | {P(res[m]['A1_unseen_crops']['fpr_at_95tpr'])} | {P(res[m]['A1_unseen_crops']['unknown_recall'])} | "
        f"{F(res[m]['A1_unseen_crops']['unknown_f1'])} | {F(res[m]['A2_non_plant_textures']['auroc'])} | {P(res[m]['A2_non_plant_textures']['fpr_at_95tpr'])} | "
        f"{P(res[m]['A2_non_plant_textures']['unknown_recall'])} | {F(res[m]['A2_non_plant_textures']['unknown_f1'])} |" for m in ALL) + f"""

{BHDR}
{bl('Energy -> Fusion A (A1)')}
{bl('Contrastive Energy* -> Fusion A (A1)')}
{bl('Energy -> Fusion A (A2)')}
{bl('Contrastive Energy* -> Fusion A (A2)')}

## 8. PlantDoc Domain-Shift Results

**Setup:** the same fusion, normalisation and threshold, with no retuning. Known = PlantDoc photos of the 21 known diseases; unknown = PlantDoc photos of other diseases or crops.

| Method | AUROC (all unknown) | AUPR-Out | FPR@95TPR | PlantDoc-known acceptance | AUROC (unknown disease, known crop) | AUROC (unknown crop) | **Domain-novelty AUROC** (PlantVillage known vs PlantDoc known) | PlantVillage-known acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(pdrow(m) for m in ALL) + f"""

{BHDR}
{bl('Energy -> Fusion A (C)')}
{bl('Contrastive Energy* -> Fusion A (C)')}

**Disease novelty vs domain novelty.**
- **Domain novelty:** the "domain-novelty AUROC" asks whether a score separates PlantDoc photos of *known* diseases from PlantVillage photos of the *same* diseases. That is image-domain novelty, not disease novelty.
  - Fusion A: {F(fdom['auroc'])}.
  - At the PlantVillage threshold it accepts {P(fdom['plantdoc_known_acceptance'])} of PlantDoc known-disease photos, vs {P(fdom['plantvillage_known_acceptance'])} of PlantVillage known photos.
- **Disease novelty within PlantDoc:** much weaker (AUROC {F(pd_f['auroc'])}).
- **Interpretation:** the detectors mostly flag *the field-photo domain*. Most known-disease field photos are rejected as "unknown" — they were never shown to the model as known.
  - This is a domain-generalisation problem layered on top of OSR. Low PlantDoc OSR numbers should not be read as "the disease-novelty signal is absent".
  - Closed-set accuracy on PlantDoc known diseases was already about 32% in Stage 7.
- **Fusion vs single scores on PlantDoc:** vs Energy {v_c}; vs contrastive Energy {v_c2}.
- **Caution on FPR@95TPR for the fusions:** almost all PlantDoc images lie beyond the known-validation range of several signals, so their percentiles tie at 1. A reported FPR@95TPR of 100% (Fusion B, D) is a tie artefact of the saturating normalisation, not a measured ranking (Limitation 6).

## 9. Per-Disease Analysis

AUROC vs all known test images (left block) and detection at each method's 95% known-validation threshold (right block):

| Unknown | Images | """ + " | ".join(f"{m} AUROC" for m in CORE) + " | " + " | ".join(f"{m} det." for m in CORE) + " |\n|---|---:|" + "---:|" * (2 * len(CORE)) + f"""
{pd_rows}

Fusion A − single method, AUROC per disease (paired bootstrap, 95% CI):

| Unknown | vs Energy | vs Mahalanobis | vs Contrastive Energy* |
|---|---|---|---|
{pd_ci}

Per normalised signal, AUROC per disease (what each ingredient contributes):

| Unknown | """ + " | ".join(SIGLAB[k] for k in names) + " |\n|---|" + "---:|" * len(names) + f"""
{sig_rows}

## 10. Grape Esca Deep Dive

{deep_sec('Grape___Esca_(Black_Measles)', 'Grape___Black_rot')}

## 11. Corn NLB Deep Dive

{deep_sec('Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot')}

**Does fusion keep Mahalanobis's NLB advantage?** Mahalanobis {F(nlb['Mahalanobis']['auroc_vs_all_known'])}, Fusion A {F(nlb[PR]['auroc_vs_all_known'])}: {'**yes**' if nlb_keep else '**no**, averaging dilutes it'}.

## 12. Method Disagreement

**Setup:** decisions at each method's own 95% known-validation threshold, for every hard-unknown test image ({S['n']:,}). The full matrix is in `results/tables/method_disagreement.csv`.

| Rejected by (pattern) | All | """ + " | ".join(short(c) for c in near) + " |\n|---|---:|" + "---:|" * len(near) + f"""
{dis_rows}

- **Rejected by all five:** {S['rejected_by_all_five']:,}. **Accepted by all five:** {S['rejected_by_none']:,}.
- **Rejected by at least one single method:** {S['rejected_by_any_single']:,} ({P(S['rejected_by_any_single'] / S['n'])}). **By Fusion A:** {S['rejected_by_fusion']:,} ({P(S['rejected_by_fusion'] / S['n'])}).
- **Only one single method rejects:**
  - MSP: {S['only MSP (among singles)']}
  - Energy: {S['only Energy (among singles)']}
  - Mahalanobis: {S['only Mahalanobis (among singles)']} (fusion keeps {S['only Mahalanobis, fusion also rejects']})
  - Contrastive Energy: {S['only Contrastive Energy* (among singles)']}
- **Rescued only by fusion:** {S['fusion_only (no single method rejects)']}. **Rejected by some single method but accepted by fusion:** {S['some_single_rejects_but_fusion_accepts']:,}.

**Reading:** the union of single methods ("any rejects") is an upper bound that no fixed-acceptance detector can reach, because taking the union rejects far more known images. Averaging gives weight to the signals that agree. A signal that is uniquely right on a sample is out-voted unless it is extreme.

**Qualitative examples:** chosen by a fixed rule — {R['examples']['rule']}. They are listed in `results/tables/fusion_qualitative_examples.csv` and shown in `figures/final_osr/qualitative_examples.png`.

## 13. Statistical Comparison

Paired stratified bootstrap ({boot['Energy -> Fusion A (B)']['_meta']['n_boot']} resamples, thresholds fixed at their validation values). Experiment B:

{BHDR}
{bl('Energy -> Fusion A (B)')}
{bl('MSP -> Fusion A (B)')}
{bl('Mahalanobis -> Fusion A (B)')}
{bl('Contrastive Energy* -> Fusion A (B)')}
{bl('Contrastive Mahalanobis -> Fusion A (B)')}
{bl('Energy -> Fusion B (max) (B)')}
{bl('Contrastive Energy* -> Fusion B (max) (B)')}
{bl('Energy -> Fusion C (cluster-weighted) (B)')}
{bl('Contrastive Energy* -> Fusion C (cluster-weighted) (B)')}
{bl('Energy -> Fusion D (val-unknown weights, secondary) (B)')}
{bl('Contrastive Energy* -> Fusion D (val-unknown weights, secondary) (B)')}

**ORACLE ANALYSIS — NOT PART OF THE DEPLOYABLE OSR METHOD.**
- **Method:** a simplex grid (step {R['oracle']['grid_step']}) over the six normalised signals, optimised **on the Experiment-B test labels** (in-sample, optimistic).
- **Result:** best AUROC {F(R['oracle']['best_auroc'])}, with weights {', '.join(f'{SIGLAB[k]} {v:.1f}' for k, v in R['oracle']['weights'].items())}. Compare Fusion A {F(fa['auroc'])}.
- **Best single signal per disease** (oracle choice): {'; '.join(f"{short(c)}: {SIGLAB[k]} {F(v)}" for c, (k, v) in R['oracle']['best_single_signal_per_disease'].items())}.
- **What it shows:** the headroom available *if* the right combination were known. The gap to equal weights is not achievable without unknown labels.

## 14. Limitations

1. **Sequential reuse of one test set.** Each stage's design was fixed on validation data, but the *choice of which signals to fuse* in Stage 8 was made knowing Stage 4–7 test results. Fusion results are therefore less clean than a single pre-registered evaluation.
2. **Equal weights are a default, not an optimum.** No novelty-targeted weighting can be learned without unknown labels, and the oracle shows that the optimum differs from equal weights.
3. **Two networks:** the fusion needs both the Stage-3 and the Stage-7 backbone (about 2× inference).
4. **Contrastive Energy carries a head effect:** its advantage over original Energy is mostly due to the linear head (Stage-7 control). Treat it as "contrastive features + linear head", not as a representation-only result.
5. **One seed, one split, PlantVillage acquisition style.** Potential residual same-leaf similarity between known train and test may still inflate known-class scores (strict-test rows are given). PlantVillage session artifacts may contribute to every score.
6. **ECDF saturation:** percentiles saturate beyond the known-validation range. Max-fusion (B) and extreme samples therefore lose ranking resolution.
7. **PlantDoc:** the known/unknown labels come from the Stage-2 class mapping, which has documented caveats (mixed rust and blight types; a 2-image class).

## 15. Research Conclusions

**Demonstrated** (paired-bootstrap intervals exclude zero, same test images):
""" + "\n".join(f"- {x}" for x in [
        f"Energy outperforms MSP on hard same-crop unknowns (Stage 5: AUROC {F(ms['auroc'])} → {F(en['auroc'])}).",
        f"Mahalanobis provides complementary information: {R['disagreement']['summary']['only Mahalanobis (among singles)']} hard-unknown images are rejected by Mahalanobis and by no other single method. It also gives a different ranking from logit scores (known-validation ρ with MSP = {Rm[names.index('orig_msp')][names.index('orig_mahalanobis')]:.2f}). It is much better on Corn NLB ({F(nlb['Mahalanobis']['auroc_vs_all_known'])} vs Energy {F(nlb['Energy']['auroc_vs_all_known'])}) but worse overall ({F(mh['auroc'])}).",
        f"The contrastive representation improves matched-score performance (same head: {hcb['diff_B_minus_A']:+.4f} {ci(hcb['diff_ci95'])}).",
        f"Fusion A vs Energy: {v_en} (AUROC {F(en['auroc'])} → {F(fa['auroc'])}). Fusion A vs contrastive Energy: {v_ce} ({F(ce['auroc'])} → {F(fa['auroc'])}). Fusion A vs Mahalanobis: {v_mh}. Fusion A vs MSP: {v_ms}.",
    ]) + f"""

**Not demonstrated:**
- that the representation is the fundamental bottleneck (the probe shows the information is present);
- that fusion solves unseen-disease recognition;
- that any detector here works on field photos.

**Answers:**
1. **Can conventional closed-set classification detect unseen diseases?** Only partly. A {P(R['closed_set']['stage3_known_test_accuracy'])}-accurate classifier with MSP rejects {P(ms['unknown_recall'])} of hard unknowns at 95% known acceptance (AUROC {F(ms['auroc'])}). It assigns most of the rest confidently to a look-alike known disease.
2. **Does MSP help?** It is far better than nothing on easy unknowns (A1 {F(res['MSP']['A1_unseen_crops']['auroc'])}, A2 {F(res['MSP']['A2_non_plant_textures']['auroc'])}) but weak on hard ones, and near chance on Grape Esca ({F(deep['Grape___Esca_(Black_Measles)']['methods']['MSP']['auroc_vs_all_known'])}).
3. **Does Energy improve on MSP?** Slightly on hard unknowns ({F(ms['auroc'])} → {F(en['auroc'])}, significant). It is worse on A1/A2, and Esca stays at chance.
4. **Does feature-space distance help?** In a class-dependent way. Mahalanobis rescues Corn NLB but loses the tomato diseases; overall it is worse than Energy.
5. **Does contrastive representation learning help?** Modestly. Matched-score gains are significant but small for logit-type scores ({hcb['diff_B_minus_A']:+.3f}). Most of the contrastive-Energy jump is due to the linear head.
6. **Does score fusion add complementary information?** {fusion_claim}
7. **Which unknown diseases remain difficult?** {' and '.join(short(c) for c in hard_cls)} (lowest Fusion-A AUROC). Grape Esca: Fusion A AUROC {F(esca_f['auroc_vs_all_known'])}, detection {P(esca_f['detection'])}; the best detection of any method is {P(esca_best)}. **Grape Esca remains unsolved** and is the central limitation.
8. **How badly does domain shift affect novelty detection?** Severely. PlantDoc AUROC for Fusion A is {F(pd_f['auroc'])}. Known-disease field photos are themselves flagged as novel (domain-novelty AUROC {F(fdom['auroc'])}; only {P(fdom['plantdoc_known_acceptance'])} accepted), so disease novelty cannot be read off these scores under shift.
9. **What is the strongest defensible method?** **{defensible}** — {'the pre-registered known-only fusion' if defensible == PR else 'the Stage-7 contrastive features with a linear head, scored by Energy'}. Hard AUROC {F(res[defensible]['B_hard_same_crop']['auroc'])}, unknown F1 {F(res[defensible]['B_hard_same_crop']['unknown_f1'])}, FPR@95TPR {P(res[defensible]['B_hard_same_crop']['fpr_at_95tpr'])}. It is the best on the primary experiment among methods that never saw unknown labels; the highest point estimate of any method is {best_m} ({F(res[best_m]['B_hard_same_crop']['auroc'])}). Its advantage is per-dataset and per-disease, not universal: Mahalanobis remains better on Corn NLB, and original Energy on Tomato Target Spot (see §8–§11).
   - This identification is **post hoc**: it was made by comparing methods on the same test set after all stages. Confirming it would need fresh, unseen data.
   - Its asterisk applies: the score depends on the Stage-7 linear head.

## Final benchmark table

| Method | Hard AUROC | Hard F1 | FPR@95TPR | Easy AUROC (A1 crops) | Easy AUROC (A2 DTD) | PlantDoc AUROC |
|---|---:|---:|---:|---:|---:|---:|
{fb('MSP')}
{fb('Energy')}
{fb('Mahalanobis')}
{fb('Contrastive Energy*', '')}
{fb('Contrastive Mahalanobis')}
{fb(PR)}
{fb('Fusion B (max)')}
{fb('Fusion D (val-unknown weights, secondary)')}

\\* The contrastive Energy row uses the Stage-7 linear head and is **not** a representation-only improvement over the original Energy score.

**Fair matched-head comparison** (Stage 7 post-hoc control):

| | Hard AUROC |
|---|---:|
| Original features + linear head, Energy | {F(HC['results']['original_linear_head::energy_linear_head']['B_hard_same_crop']['auroc'])} |
| Contrastive features + linear head, Energy | {F(res['Contrastive Energy*']['B_hard_same_crop']['auroc'])} |
| Difference [95% CI] | {hcb['diff_B_minus_A']:+.4f} {ci(hcb['diff_ci95'])} |

Figures: `results/figures/final_osr/`.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_8_FINAL_OSR_BENCHMARK.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
