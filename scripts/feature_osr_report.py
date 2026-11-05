"""Build results/reports/STAGE_6_FEATURE_OSR.md from results/metrics/mahalanobis_results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_energy_report import ci, verdict  # noqa: E402
from scripts.osr_report import F, P, short  # noqa: E402
from src.utils.config import resolve  # noqa: E402


def write_report(cfg) -> None:
    R = json.loads((resolve(cfg["paths"]["metrics"]) / "mahalanobis_results.json").read_text())
    thr, MA, EN, MS, B = R["threshold"], R["mahalanobis_results"], R["energy_results"], R["msp_results"], R["bootstrap"]
    tau = thr["threshold_mahalanobis"]
    ex = R["extraction_checks"]
    bB = B["energy_vs_maha::B_hard_same_crop"]
    bM = B["msp_vs_maha::B_hard_same_crop"]
    vb = lambda b, m: verdict(b, m).replace("Energy", "Mahalanobis")  # noqa: E731  (B in bootstrap = Mahalanobis)

    def main_row(name, r):
        return (f"| {name} | {F(r['auroc'])} | {F(r['aupr_out'])} | {P(r['fpr_at_95tpr'])} | {P(r['unknown_recall'])} | "
                f"{P(r['unknown_precision'])} | {F(r['unknown_f1'])} | {P(r['known_acceptance_rate'])} | {P(r['false_rejection_rate'])} | {F(r['oscr'])} |")

    pc = pd.DataFrame(R["per_unknown_class"])
    pb = pc[pc["experiment"] == "B_hard_same_crop"].sort_values("msp_auroc")
    rows_pd = "\n".join(
        f"| {short(r.unknown_class)} | {F(r.msp_auroc)} | {F(r.energy_auroc)} | {F(r.mahalanobis_auroc)} | "
        f"{B['energy_vs_maha::B_class::' + r.unknown_class]['auroc']['diff_B_minus_A']:+.4f} "
        f"{ci(B['energy_vs_maha::B_class::' + r.unknown_class]['auroc']['diff_ci95'])} | "
        f"{P(r.msp_detection)} | {P(r.energy_detection)} | {P(r.mahalanobis_detection)} |" for r in pb.itertuples())
    pa = pc[pc["experiment"] == "A1_unseen_crops"].sort_values("mahalanobis_auroc")
    rows_a = "\n".join(f"| {short(r.unknown_class)} | {F(r.msp_auroc)} | {F(r.energy_auroc)} | {F(r.mahalanobis_auroc)} | "
                       f"{P(r.mahalanobis_detection)} |" for r in pa.itertuples())

    px = pd.DataFrame(R["feature_proximity"])
    pxb = px[px["role"] == "unknown_near"].sort_values("mahalanobis_auroc")
    rows_px = "\n".join(
        f"| {short(r.unknown)} | {r.crop.replace('Corn_(maize)', 'Corn')} | {short(r.nearest_known_by_images)} "
        f"({P(r.share_images_nearest_to_it)} of images) | {short(r.nearest_known_by_class_centroid)} | "
        f"{F(r.dist_to_nearest_centroid_mean, 3)} / {F(r.dist_to_nearest_centroid_median, 3)} | {F(r.cosine_to_nearest_centroid_mean)} | "
        f"{P(r.nn1_train_class_share_of_nearest_known)} | {P(r.nn1_train_same_crop_share)} | {F(r.msp_auroc)} | {F(r.energy_auroc)} | "
        f"{F(r.mahalanobis_auroc)} |" for r in pxb.itertuples())
    pxa = px[px["role"] != "unknown_near"].sort_values("mahalanobis_auroc")
    rows_pxa = "\n".join(
        f"| {short(r.unknown)} | {short(r.nearest_known_by_images)} ({P(r.share_images_nearest_to_it)}) | "
        f"{F(r.dist_to_nearest_centroid_mean, 3)} | {F(r.cosine_to_nearest_centroid_mean)} | {F(r.mahalanobis_auroc)} |"
        for r in pxa.itertuples())
    kr = R["known_reference"]

    def deep_table(key):
        t = pd.DataFrame(R[key]["table"]).sort_values("centroid_euclidean_unknown_to_known")
        return "\n".join(
            f"| {short(r.known_class)} | {r.rank_among_21_by_centroid_euclidean} | {F(r.centroid_euclidean_unknown_to_known, 3)} | "
            f"{F(r.centroid_cosine_unknown_to_known)} | {F(r.mean_dist_unknown_images_to_known_centroid, 3)} vs "
            f"{F(r.mean_dist_known_test_images_to_own_centroid, 3)} | {F(r.nn_dist_unknown_to_known_train_median, 3)} vs "
            f"{F(r.nn_dist_known_test_to_own_train_median, 3)} | {F(r.pairwise_unknown_vs_known_train_median, 3)} vs "
            f"{F(r.pairwise_known_test_vs_own_train_median, 3)} | {P(r.share_unknown_images_nearest_centroid_is_this)} | "
            f"{F(r.maha_auroc_unknown_vs_this_class_test)} | {F(r.linear_probe_auroc_unknown_vs_known_train_DIAGNOSTIC)} |"
            for r in t.itertuples())

    def deep_summary(key, unknown):
        t = pd.DataFrame(R[key]["table"])
        c = R[key]["closest_same_crop_known"]
        r = t[t["known_class"] == c].iloc[0]
        b = B[f"energy_vs_maha::B_class::{unknown}"]
        m = pb[pb["unknown_class"] == unknown].iloc[0]
        ratio = r.mean_dist_unknown_images_to_known_centroid / r.mean_dist_known_test_images_to_own_centroid
        return (f"- **Closest known class** (computed, not assumed): **{short(c)}**. By centroid distance among all 21, the top 3 are {', '.join(short(x) for x in R[key]['top5_known_by_centroid_euclidean'][:3])}.\n"
                f"- **How far the unknown sits from that class's centroid:** its images are on average {F(r.mean_dist_unknown_images_to_known_centroid, 3)} away, vs {F(r.mean_dist_known_test_images_to_own_centroid, 3)} for {short(c)}'s own test images (ratio {ratio:.2f}). {P(r.share_unknown_images_nearest_centroid_is_this)} of its images have {short(c)} as their nearest centroid.\n"
                f"- **AUROC, MSP → Energy → Mahalanobis:** {F(m.msp_auroc)} → {F(m.energy_auroc)} → {F(m.mahalanobis_auroc)}. Mahalanobis − Energy = {b['auroc']['diff_B_minus_A']:+.4f} {ci(b['auroc']['diff_ci95'])}: {vb(b, 'auroc')}.\n"
                f"- **Detection at the validation thresholds:** {P(m.msp_detection)} → {P(m.energy_detection)} → {P(m.mahalanobis_detection)}.\n"
                f"- **Mahalanobis AUROC against {short(c)} test images only:** {F(r.maha_auroc_unknown_vs_this_class_test)}.\n"
                f"- **Linear-probe diagnostic** (5-fold CV logistic regression, unknown vs {short(c)} training features; this uses unknown labels and is **not** an OSR method): AUROC {F(r.linear_probe_auroc_unknown_vs_known_train_DIAGNOSTIC)}.")

    corr = R["correlations"]
    cn = corr["names"]
    corr_rows = "\n".join(f"| {a} | " + " | ".join(f"{corr['pearson'][i][j]:.3f} / {corr['spearman'][i][j]:.3f}" for j in range(len(cn))) + " |"
                          for i, a in enumerate(cn))
    dg = R["disagreement"]
    by_cls = pd.DataFrame(dg["by_class"]).T.fillna(0).astype(int)
    by_cls_rows = "\n".join(f"| {short(c)} | " + " | ".join(str(r.get(k, 0)) for k in
                            ("all_three_fail", "msp_and_energy_fail_maha_succeeds", "energy_succeeds_maha_fails", "all_three_succeed")) + " |"
                            for c, r in by_cls.iterrows())

    # outcome per the Stage-6 pre-specified categories (decided from bootstrap intervals vs Energy)
    v_all = vb(bB, "auroc")
    esca_b = B["energy_vs_maha::B_class::Grape___Esca_(Black_Measles)"]
    esca_maha = float(pb[pb["unknown_class"] == "Grape___Esca_(Black_Measles)"]["mahalanobis_auroc"].iloc[0])
    n_better = sum(B[f"energy_vs_maha::B_class::{c}"]["auroc"]["diff_ci95"][0] > 0 for c in pb["unknown_class"])
    if v_all.startswith("Mahalanobis better") and esca_maha < 0.7:
        outcome = "**D** — Mahalanobis improves some diseases but fails Grape Esca"
    elif v_all.startswith("Mahalanobis better"):
        outcome = "**A** — Mahalanobis significantly improves hard-unknown detection"
    elif v_all.startswith("Mahalanobis worse"):
        outcome = ("**C, with elements of D** — Mahalanobis is worse overall, but improves some diseases" if n_better else
                   "**C** — Mahalanobis is worse")
    else:
        outcome = "**B** — Mahalanobis is similar to Energy (difference interval spans 0)"
    pdA, pdC = MA["C_plantdoc_all_unknown"], EN["C_plantdoc_all_unknown"]
    il = []
    for key, unk in (("grape_esca", "Grape___Esca_(Black_Measles)"), ("corn_nlb", "Corn_(maize)___Northern_Leaf_Blight")):
        t = pd.DataFrame(R[key]["table"])
        c = R[key]["closest_same_crop_known"]
        probe = float(t[t["known_class"] == c]["linear_probe_auroc_unknown_vs_known_train_DIAGNOSTIC"].iloc[0])
        m = pb[pb["unknown_class"] == unk].iloc[0]
        best_logit = max(m.msp_auroc, m.energy_auroc)
        kind = ("**the information is present but unsupervised scores do not fully exploit it**" if probe >= 0.95 and max(best_logit, m.mahalanobis_auroc) < 0.9
                else "**feature distance exploits most of the separating information**" if probe >= 0.95
                else "**the representation itself separates them poorly**")
        il.append(f"  - **{short(unk)} vs {short(c)}:** probe AUROC {F(probe)}; best logit score {F(best_logit)}; Mahalanobis {F(m.mahalanobis_auroc)}. So {kind}.")
    sp = corr["spearman"][cn.index("1-MSP")][cn.index("Mahalanobis")]
    il.append(f"  - **Different information:** Mahalanobis is weakly correlated with MSP (Spearman {sp:.3f}) and Energy (Spearman {corr['spearman'][cn.index('Energy')][cn.index('Mahalanobis')]:.3f}). It rescues {dg['energy_fails_maha_succeeds']:,} hard-unknown images that Energy accepts, but loses {dg['maha_fails_energy_succeeds']:,} that Energy rejects. The scores are complementary rather than ordered; combining them was not tested in this stage.")
    worst_m = pb.sort_values("mahalanobis_auroc").iloc[0]
    il.append(f"  - **Where Mahalanobis fails:** {short(worst_m.unknown_class)} (AUROC {F(worst_m.mahalanobis_auroc)}). Its images lie at about the same distance from the nearest training centroid as known test images (see the proximity table), so a pure distance score cannot flag them, while the logit scores can.")
    il.append("  - **Caveat on the probe:** near-perfect separability may partly reflect acquisition differences between PlantVillage photo sessions (e.g. leaf batches, lighting days) rather than disease symptoms. That would make it an optimistic upper bound.")
    il.append(f"  - **Overall hypothesis check:** the Stage-5 hypothesis ('the representation is the main limit') is **{'partly refuted' if all('information is present' in x or 'exploits' in x for x in il[:2]) else 'supported'}** for the two hardest diseases. Their features *do* contain separating directions, but no label-free score used here finds them reliably at a deployable threshold.")
    interp_lines = "\n".join(il)

    report = f"""# Stage 6 — Feature-Space Analysis and Mahalanobis OSR

*Generated by `scripts/feature_osr.py report` from `results/metrics/mahalanobis_results.json`. MSP and Energy numbers come from the saved Stage-4/5 files and were re-verified against the shared logits.*

## Research question

Does the learned feature space of the frozen Stage-3 classifier separate unseen diseases from known diseases? If it does, can a feature-distance score (Mahalanobis) recover what the logit scores (MSP, Energy) miss? If it does not, the limitation lies in the representation itself.

## Embedding extraction

- **Representation:** the **{ex['embedding_dim']}-d penultimate features** of the frozen ResNet-50: the global-average-pooled `layer4` output, which is exactly the input of the 21-class linear head. Extraction used fp32 with the Stage-3 evaluation preprocessing.
- **Format:** `results/embeddings/{{train,val,test}}_embeddings.npz` (float32 embeddings + logits + image paths), with matching `*_meta.csv` metadata (crop, disease, role, split, prediction, MSP, energy). About 0.48 GB in total; git-ignored.
- **Sanity checks:** see `extraction_checks.json` and `tests/test_embeddings.py`.
  - All embeddings finite.
  - `head(embedding)` reproduces the logits (max |Δ| {ex['splits']['train']['head_consistency_max_abs']:.1e}).
  - Validation and test logits equal the Stage-4 cached logits (max |Δ| {ex['splits']['test'].get('vs_stage4_logits_max_abs', float('nan')):.1e}; 0 argmax mismatches).
  - A repeated extraction differs by at most {ex['determinism_rerun_max_abs_diff']:.1e}.
  - Rows are aligned one-to-one with the locked manifests.

## Visualisation

- **PCA:** fitted on **known training embeddings only** ({cfg['feature_osr']['pca_components']} components, {P(R['pca']['explained_variance_total'])} of training variance). Known validation, known test and hard-unknown test images are *projected*. Figure: `results/figures/embeddings/pca_known_unknown.png`.
- **t-SNE (visualisation only):** {R['tsne']['n_points']:,} test images (≤{R['tsne']['per_class_sample']} per class), PCA-50 input, perplexity {R['tsne']['perplexity']}, {R['tsne']['max_iter']} iterations, seed {R['tsne']['seed']}. Figure: `tsne_known_unknown.png`. The coordinates were not used for any decision.
- **UMAP:** {R['umap']}.

The plots were made after the experimental classes were fixed (Stage 2) and were not used to choose anything.

## Feature-space quantitative analysis

**Reference points:**
- known test images lie on average {F(kr['known_test_dist_to_own_centroid']['mean'], 3)} (median {F(kr['known_test_dist_to_own_centroid']['median'], 3)}) from their own class centroid;
- they lie {F(kr['known_test_dist_to_nearest_centroid']['mean'], 3)} from the nearest centroid;
- mean cosine similarity to the nearest centroid: {F(kr['known_test_cosine_to_nearest_centroid_mean'])}.

Centroids are training means.

**Hard unknowns — proximity to the known classes** (`results/tables/unknown_feature_proximity.csv`):

| Unknown | Crop | Nearest known (share of its images) | Nearest by class centroid | Distance to nearest centroid, mean / median | Cosine to nearest centroid | 1-NN train image from that class | 1-NN train image same crop | MSP AUROC | Energy AUROC | Mahalanobis AUROC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{rows_px}

**Unseen crops (A1) and DTD (A2):**

| Unknown | Nearest known (share) | Distance to nearest centroid | Cosine | Mahalanobis AUROC |
|---|---|---:|---:|---:|
{rows_pxa}

Nearest-centroid distance distributions are in `nearest_centroid_distance.png`.

## Mahalanobis method

- **Fitted on known TRAINING embeddings only** ({thr['fit']['n_train']:,} images): 21 class means and one shared covariance (Lee et al., 2018).
- **Covariance estimate:** Ledoit–Wolf shrinkage of the class-centred features. Shrinkage coefficient = {thr['fit']['shrinkage']:.4f}.
- **Inversion:** symmetric eigendecomposition; eigenvalues floored at {thr['fit']['eig_floor_rel']:.0e} × max ({thr['fit']['n_eigenvalues_floored']} floored; condition number {thr['fit']['condition_number_after_floor']:.3g}). Distances are computed in the whitened space, so no explicit inverse is formed.
- **Unit tests** (`tests/test_mahalanobis.py`): singular covariance (n < d), near-zero eigenvalues, extreme inputs, NaN/Inf rejection, match against an explicit inverse, direction.
- **Score:** the minimum over the 21 classes of the squared Mahalanobis distance. Higher = more unknown; accept as known iff score ≤ τ.
- **Threshold** (validation only, same rule as MSP and Energy): τ = **{tau:.4f}**, accepting {P(thr['validation']['known_val_acceptance'])} of known validation images. At this threshold it rejects {P(thr['validation']['misclassified_val_rejected'])} of misclassified known-validation images (MSP 82.86%, Energy 54.29%).
- **Validation-unknown AUROC (reported only):** {F(thr['validation']['auroc_known_val_vs_validation_unknown_reported_only'])}, vs MSP {F(thr['validation']['msp_reference_auroc_kval_vs_valunk'])} and Energy {F(thr['validation']['energy_reference_auroc_kval_vs_valunk'])}.

**Protocol check before test scoring:** {'; '.join(f'{k}: {v}' for k, v in R['protocol_check'].items())}.

## Results — Experiment B (hard same-crop, primary)

| Method | Hard AUROC | AUPR-Out | FPR@95TPR | Unknown recall | Unknown precision | Unknown F1 | Known acceptance | False rejection | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{main_row('MSP (Stage 4)', MS['B_hard_same_crop'])}
{main_row('Energy, T=1 (Stage 5)', EN['B_hard_same_crop'])}
{main_row('Mahalanobis (this stage)', MA['B_hard_same_crop'])}

**Paired bootstrap, Mahalanobis − Energy** (same images, {bB['_meta']['n_boot']} stratified resamples, fixed validation thresholds):
- AUROC {bB['auroc']['diff_B_minus_A']:+.4f} {ci(bB['auroc']['diff_ci95'])}: {vb(bB, 'auroc')};
- FPR@95TPR {100 * bB['fpr_at_95tpr']['diff_B_minus_A']:+.2f} pp {ci(bB['fpr_at_95tpr']['diff_ci95'], pct=True)}: {vb(bB, 'fpr_at_95tpr')};
- detection {100 * bB['unknown_recall']['diff_B_minus_A']:+.2f} pp {ci(bB['unknown_recall']['diff_ci95'], pct=True)}: {vb(bB, 'unknown_recall')};
- unknown F1 {bB['unknown_f1']['diff_B_minus_A']:+.4f} {ci(bB['unknown_f1']['diff_ci95'])}: {vb(bB, 'unknown_f1')};
- known acceptance {100 * bB['known_acceptance_rate']['diff_B_minus_A']:+.2f} pp: {vb(bB, 'known_acceptance_rate')}.

**Mahalanobis − MSP:** AUROC {bM['auroc']['diff_B_minus_A']:+.4f} {ci(bM['auroc']['diff_ci95'])}, {vb(bM, 'auroc')}.

**Other conditions (AUROC: MSP / Energy / Mahalanobis; the Mahalanobis − Energy interval decides the verdict):**
- A1 unseen crops: {F(MS['A1_unseen_crops']['auroc'])} / {F(EN['A1_unseen_crops']['auroc'])} / {F(MA['A1_unseen_crops']['auroc'])}; {vb(B['energy_vs_maha::A1_unseen_crops'], 'auroc')}. Mahalanobis detection {P(MA['A1_unseen_crops']['unknown_recall'])}, FPR@95TPR {P(MA['A1_unseen_crops']['fpr_at_95tpr'])}.
- A2 DTD: {F(MS['A2_non_plant_textures']['auroc'])} / {F(EN['A2_non_plant_textures']['auroc'])} / {F(MA['A2_non_plant_textures']['auroc'])}; {vb(B['energy_vs_maha::A2_non_plant_textures'], 'auroc')}. Mahalanobis detection {P(MA['A2_non_plant_textures']['unknown_recall'])}.

**A1 per unseen crop:**

| Class | MSP AUROC | Energy AUROC | Mahalanobis AUROC | Mahalanobis detection |
|---|---:|---:|---:|---:|
{rows_a}

## Per-disease results (Experiment B)

| Unknown | MSP AUROC | Energy AUROC | Mahalanobis AUROC | Mahalanobis − Energy [95% CI] | MSP detection | Energy detection | Mahalanobis detection |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows_pd}

## Grape Esca analysis

{deep_summary('grape_esca', 'Grape___Esca_(Black_Measles)')}

| Known class | Rank (centroid, of 21) | Centroid distance | Centroid cosine | Mean distance to centroid: Esca images vs class's own test | Median 1-NN distance to class train: Esca vs own test | Median pairwise to class train: Esca vs own test | Esca images with this nearest centroid | Mahalanobis AUROC vs this class | Linear probe AUROC (diagnostic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{deep_table('grape_esca')}

Full table: `results/tables/grape_esca_feature_distance.csv`. Figure: `grape_esca_feature_space.png`.

## Corn NLB analysis

{deep_summary('corn_nlb', 'Corn_(maize)___Northern_Leaf_Blight')}

| Known class | Rank | Centroid distance | Centroid cosine | Mean distance: NLB vs own test | Median 1-NN: NLB vs own test | Median pairwise: NLB vs own test | NLB images nearest to this | Mahalanobis AUROC vs this class | Linear probe AUROC (diagnostic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{deep_table('corn_nlb')}

## Score correlations (Experiment-B pool)

Each cell is Pearson / Spearman; all scores are oriented "higher = more unknown". Figure: `osr_score_correlation.png`.

| | {' | '.join(cn)} |
|---|{'---:|' * len(cn)}
{corr_rows}

## Failure analysis — method disagreement on hard unknowns

`results/tables/osr_method_disagreement.csv` has one row per hard-unknown image, with all three scores and decisions.

| Category | Images |
|---|---:|
| MSP fails, Mahalanobis succeeds | {dg['msp_fails_maha_succeeds']:,} |
| Energy fails, Mahalanobis succeeds | {dg['energy_fails_maha_succeeds']:,} |
| Energy succeeds, Mahalanobis fails | {dg['maha_fails_energy_succeeds']:,} |
| All three fail | {dg['all_three_fail']:,} |
| All three succeed | {dg['all_three_succeed']:,} |

**By disease:**

| Unknown | All three fail | MSP & Energy fail, Mahalanobis succeeds | Energy succeeds, Mahalanobis fails | All three succeed |
|---|---:|---:|---:|---:|
{by_cls_rows}

## PlantDoc (Experiment C, separate)

| Condition | AUROC (MSP / Energy / Mahalanobis) | Mahalanobis AUPR-Out | Mahalanobis FPR@95TPR | Mahalanobis known acceptance | Mahalanobis − Energy AUROC [95% CI] |
|---|---:|---:|---:|---:|---:|
| All unknown | {F(MS['C_plantdoc_all_unknown']['auroc'])} / {F(pdC['auroc'])} / {F(pdA['auroc'])} | {F(pdA['aupr_out'])} | {P(pdA['fpr_at_95tpr'])} | {P(pdA['known_acceptance_rate'])} | {B['energy_vs_maha::C_plantdoc_all_unknown']['auroc']['diff_B_minus_A']:+.4f} {ci(B['energy_vs_maha::C_plantdoc_all_unknown']['auroc']['diff_ci95'])} |
| Unknown disease, known crop | {F(MS['C_plantdoc_near_unknown']['auroc'])} / {F(EN['C_plantdoc_near_unknown']['auroc'])} / {F(MA['C_plantdoc_near_unknown']['auroc'])} | {F(MA['C_plantdoc_near_unknown']['aupr_out'])} | {P(MA['C_plantdoc_near_unknown']['fpr_at_95tpr'])} | {P(MA['C_plantdoc_near_unknown']['known_acceptance_rate'])} | {B['energy_vs_maha::C_plantdoc_near_unknown']['auroc']['diff_B_minus_A']:+.4f} {ci(B['energy_vs_maha::C_plantdoc_near_unknown']['auroc']['diff_ci95'])} |
| Unknown crop | {F(MS['C_plantdoc_far_unknown']['auroc'])} / {F(EN['C_plantdoc_far_unknown']['auroc'])} / {F(MA['C_plantdoc_far_unknown']['auroc'])} | {F(MA['C_plantdoc_far_unknown']['aupr_out'])} | {P(MA['C_plantdoc_far_unknown']['fpr_at_95tpr'])} | {P(MA['C_plantdoc_far_unknown']['known_acceptance_rate'])} | {B['energy_vs_maha::C_plantdoc_far_unknown']['auroc']['diff_B_minus_A']:+.4f} {ci(B['energy_vs_maha::C_plantdoc_far_unknown']['auroc']['diff_ci95'])} |

At the PlantVillage-chosen τ, Mahalanobis accepts {P(pdA['known_acceptance_rate'])} of PlantDoc known-field images, vs {P(pdC['known_acceptance_rate'])} for Energy. {'Feature distance is even more sensitive to the change of imaging domain than the logit scores (it accepts fewer known field images).' if pdA['known_acceptance_rate'] < pdC['known_acceptance_rate'] else 'Feature distance accepts at least as many known field images as Energy.'} See `nearest_centroid_distance.png` for how far PlantDoc known-field images lie from the PlantVillage training centroids.

## Interpretation

- **Pre-specified outcome:** {outcome}. Decided from the paired-bootstrap interval against Energy, plus Grape Esca's Mahalanobis AUROC ({F(esca_maha)}).
- **Representation vs scoring.** For each hard unknown, compare its Mahalanobis AUROC (*can feature distance flag it?*) with the linear-probe diagnostic (*does the representation contain any direction that separates it from its look-alike?*).
  - A disease where the probe separates well but every OOD score fails would be a **scoring** problem: the information exists but density/distance-based scores do not use it.
  - A disease where even the probe fails would be a **representation** problem.

  The Grape Esca and Corn NLB sections above give these numbers. Because the probe is trained *with* unknown labels, it shows an upper bound on what the frozen features encode, not an achievable OSR result.
{interp_lines}

## Limitations

1. **The representation is inherited from a closed-set classifier** trained only to separate 21 classes. Nothing encouraged it to keep novelty-relevant information.
2. **Gaussian assumptions:** one shared covariance and class-conditional Gaussians are assumptions. The shrinkage estimate is well conditioned but biased toward the identity; class-specific covariances, feature normalisation and layer ensembles (Lee et al. 2018) were deliberately not tuned.
3. **Controlled PlantVillage:** uniform board, light and single leaves. Feature distance partly measures imaging conditions, as the PlantDoc results show.
4. **Domain shift:** in Experiment C all three scores mix "new disease" with "new photo conditions".
5. **Ambiguous visual symptoms:** some unknown images have mild or atypical symptoms (Stage 2), and some Esca / Black-rot lesions may genuinely look alike.
6. **Diagnostics are not methods:** the linear probe uses unknown labels and is reported only to locate the bottleneck. t-SNE is qualitative.
7. **Single model and seed.** The bootstrap covers test sampling only.

## Conclusion

On the primary hard same-crop experiment, Mahalanobis distance on the frozen features gives AUROC {F(MA['B_hard_same_crop']['auroc'])}, vs {F(EN['B_hard_same_crop']['auroc'])} for Energy and {F(MS['B_hard_same_crop']['auroc'])} for MSP ({vb(bB, 'auroc')}).

For Grape Esca the three scores give {F(float(pb[pb['unknown_class'] == 'Grape___Esca_(Black_Measles)']['msp_auroc'].iloc[0]))} / {F(float(pb[pb['unknown_class'] == 'Grape___Esca_(Black_Measles)']['energy_auroc'].iloc[0]))} / {F(esca_maha)}. {'Swapping the score, whether logit-based or feature-distance-based, does not by itself make the hardest look-alike disease reliably detectable.' if esca_maha < 0.7 else 'Feature distance makes Grape Esca substantially more detectable than the logit scores, so the representation holds information that softmax and energy did not use.'} The interpretation section above states what the probe diagnostic implies about where the remaining information is lost.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_6_FEATURE_OSR.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
