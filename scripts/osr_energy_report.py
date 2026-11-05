"""Build results/reports/STAGE_5_ENERGY_OSR.md from results/metrics/osr_energy_results.json.

MSP numbers are read from the saved Stage-4 metrics file (and were re-verified to be reproduced
exactly from the shared logits). Verdicts are derived from paired-bootstrap difference intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_report import F, P, short  # noqa: E402
from src.utils.config import resolve  # noqa: E402

HIGHER_BETTER = {"auroc": True, "unknown_recall": True, "unknown_f1": True, "fpr_at_95tpr": False,
                 "known_acceptance_rate": True}


def verdict(b: dict, metric: str) -> str:
    lo, hi = b[metric]["diff_ci95"]
    good = HIGHER_BETTER[metric]
    if lo > 0:
        return "Energy better" if good else "Energy worse"
    if hi < 0:
        return "Energy worse" if good else "Energy better"
    return "no detectable difference (CI spans 0)"


def ci(x, pct=False, d=4):
    f = (lambda v: f"{100 * v:.2f}") if pct else (lambda v: f"{v:.{d}f}")
    return f"[{f(x[0])}, {f(x[1])}]"


def write_report(cfg) -> None:
    R = json.loads((resolve(cfg["paths"]["metrics"]) / "osr_energy_results.json").read_text())
    S4 = json.loads((resolve(cfg["osr_energy"]["stage4_results"])).read_text())
    thr, E, M, boot = R["threshold"], R["energy_results"]["primary"], S4["results"], R["bootstrap_msp_vs_energy_primary"]
    T, tau = thr["primary"]["temperature"], thr["primary"]["threshold_energy"]
    cs = S4["closed_set_system"]
    pv = thr["per_temperature_validation"]
    bB = boot["B_hard_same_crop"]

    def cmp_row(name, r):
        return (f"| {name} | {F(r['auroc'])} | {F(r['aupr_out'])} | {P(r['fpr_at_95tpr'])} | {P(r['unknown_recall'])} | "
                f"{P(r['unknown_precision'])} | {F(r['unknown_f1'])} | {F(r['oscr'])} |")

    def boot_rows(key):
        b = boot[key]
        return "\n".join(f"| {m} | {F(b[m]['A'])} {ci(b[m]['A_ci95'])} | {F(b[m]['B'])} {ci(b[m]['B_ci95'])} | "
                         f"{b[m]['diff_B_minus_A']:+.4f} {ci(b[m]['diff_ci95'])} | {P(b[m]['share_B_gt_A'])} | {verdict(b, m)} |"
                         for m in ("auroc", "fpr_at_95tpr", "unknown_recall", "unknown_f1", "known_acceptance_rate"))

    exps = [("B hard same-crop (primary)", "B_hard_same_crop"), ("A1 unseen crops", "A1_unseen_crops"),
            ("A2 non-plant (DTD)", "A2_non_plant_textures"), ("C PlantDoc, all unknown", "C_plantdoc_all_unknown"),
            ("C PlantDoc, unknown disease / known crop", "C_plantdoc_near_unknown"),
            ("C PlantDoc, unknown crop", "C_plantdoc_far_unknown")]
    all_rows = "\n".join(
        f"| {n} | {F(M[k]['auroc'])} → {F(E[k]['auroc'])} | {P(M[k]['fpr_at_95tpr'])} → {P(E[k]['fpr_at_95tpr'])} | "
        f"{P(M[k]['unknown_recall'])} → {P(E[k]['unknown_recall'])} | {F(M[k]['unknown_f1'])} → {F(E[k]['unknown_f1'])} | "
        f"{P(M[k]['known_acceptance_rate'])} → {P(E[k]['known_acceptance_rate'])} | "
        + (f"{boot[k]['auroc']['diff_B_minus_A']:+.4f} {ci(boot[k]['auroc']['diff_ci95'])}" if k in boot else "—") + " |"
        for n, k in exps)

    pc = pd.DataFrame(R["per_unknown_class"])
    pb = pc[pc["experiment"] == "B_hard_same_crop"].sort_values("msp_auroc")
    rows_b = "\n".join(
        f"| {short(r.unknown_class)} | {r.images:,} | {F(r.msp_auroc)} | {F(r.energy_auroc)} | "
        f"{boot['B_class::' + r.unknown_class]['auroc']['diff_B_minus_A']:+.4f} {ci(boot['B_class::' + r.unknown_class]['auroc']['diff_ci95'])} | "
        f"{P(r.msp_detection)} | {P(r.energy_detection)} | "
        f"{100 * (r.energy_detection - r.msp_detection):+.2f} pp {ci(boot['B_class::' + r.unknown_class]['unknown_recall']['diff_ci95'], pct=True)} |"
        for r in pb.itertuples())
    pa = pc[pc["experiment"] == "A1_unseen_crops"].sort_values("msp_auroc")
    rows_a = "\n".join(f"| {short(r.unknown_class)} | {r.images:,} | {F(r.msp_auroc)} | {F(r.energy_auroc)} | "
                       f"{P(r.msp_detection)} | {P(r.energy_detection)} |" for r in pa.itertuples())
    n_improved = int(((pb["energy_auroc"] - pb["msp_auroc"]) > 0).sum())

    dd = R["deep_dive"]

    def deep_block(c):
        d = dd[c]
        return f"""### {short(c)} ({d['images']:,} test images; Stage-2 look-alike: {', '.join(short(x) for x in d['resembles'])})

| | MSP | Energy (T={T}) |
|---|---:|---:|
| Detection at validation threshold | {P(d['msp']['detection'])} | {P(d['energy']['detection'])} |
| AUROC vs all known test | {F(d['msp']['auroc_vs_all_known'])} | {F(d['energy']['auroc_vs_all_known'])} |
| AUROC vs same-crop known test | {F(d['msp']['auroc_vs_same_crop_known'])} | {F(d['energy']['auroc_vs_same_crop_known'])} |
| AUROC vs look-alike known class only ({d['lookalike_known_test_images']} images) | {F(d['msp']['auroc_vs_lookalike_known'])} | {F(d['energy']['auroc_vs_lookalike_known'])} |
| Median score | MSP {F(d['msp']['median'], 6)} | E {F(d['energy']['median'], 3)} (all known: {F(d['median_energy_all_known'], 3)}; look-alike known: {F(d['median_energy_lookalike_known'], 3)}) |

- **Predicted as (closed set):** {', '.join(f'{short(k)} {v}' for k, v in d['predicted_as'].items())}.
- **Paired bootstrap, Energy − MSP:** AUROC {d['bootstrap']['auroc']['diff_B_minus_A']:+.4f} {ci(d['bootstrap']['auroc']['diff_ci95'])}; detection {100 * d['bootstrap']['unknown_recall']['diff_B_minus_A']:+.2f} pp {ci(d['bootstrap']['unknown_recall']['diff_ci95'], pct=True)}.
- **Verdict:** {verdict(d['bootstrap'], 'auroc')} (AUROC); {verdict(d['bootstrap'], 'unknown_recall')} (detection)."""

    fa = R["failure_analysis_energy"]
    ov = fa["overlap_with_msp_failures"]
    st = fa["image_statistics_mean"]
    top = "\n".join(f"| {short(d['class_name'])} | {short(d['pred_class'])} | {d['count']} |" for d in fa["accepted_top_predictions"][:8])
    worst = "\n".join(f"| {short(d['class_name'])} | {short(d['pred_class'])} | {d['msp']:.6f} | {d['energy']:.3f} |"
                      for d in fa["most_confident_energy_failures"][:6])
    temp_rows = "\n".join(f"| {k} | {v['threshold_energy']:.4f} | {P(v['known_val_acceptance'])} | {P(v['misclassified_val_rejected'])} | "
                          f"{F(v['auroc_misclassified_vs_correct_known_val'])} | {F(v['auroc_known_val_vs_validation_unknown'])} |"
                          for k, v in pv.items())
    sec = R["energy_results"].get("secondary")
    sec_txt = ("Not computed: the validation-selected temperature equals the primary T." if not sec else
               f"T* = {thr['secondary']['temperature']} (threshold {thr['secondary']['threshold_energy']:.4f}). "
               f"Experiment B: AUROC {F(sec['B_hard_same_crop']['auroc'])}, detection {P(sec['B_hard_same_crop']['unknown_recall'])}, "
               f"F1 {F(sec['B_hard_same_crop']['unknown_f1'])}. Experiment A1: AUROC {F(sec['A1_unseen_crops']['auroc'])}. "
               f"Experiment C: AUROC {F(sec['C_plantdoc_all_unknown']['auroc'])}. This variant is reported for completeness; "
               "the primary T was fixed in advance.")
    vB = {m: verdict(bB, m) for m in ("auroc", "fpr_at_95tpr", "unknown_recall", "unknown_f1", "known_acceptance_rate")}
    improved = vB["auroc"] == "Energy better"
    worse = vB["auroc"] == "Energy worse"
    headline = ("**Energy improves on MSP in hard-unknown AUROC**" if improved else
                "**Energy is worse than MSP in hard-unknown AUROC**" if worse else
                "**Energy does not detectably differ from MSP in hard-unknown AUROC**")

    report = f"""# Stage 5 — Energy-Based Open-Set Recognition vs MSP

*Generated by `scripts/osr_energy.py report` from `results/metrics/osr_energy_results.json`; MSP numbers come from the saved Stage-4 file `results/metrics/osr_msp_results.json`.*

## Research question

Does an energy score computed from the **same frozen classifier and the same logits** detect unseen plant diseases better than the maximum softmax probability (MSP)? The comparison is *same model, different OOD score*.

## Method

- **Energy score** (Liu et al., NeurIPS 2020): `E(x) = −T · logsumexp(f(x) / T)` over the 21 known-class logits. It is computed with the max-shift log-sum-exp trick in float64.
  - Unit tests: finite for logits of ±1e4, where a naive exp-sum overflows; matches torch `logsumexp` to 1e-10.
- **Sign convention, verified on synthetic logits:** **lower energy = more in-distribution.** A confident logit vector `[10, 0, …]` has E = −10.00, a uniform vector has E = −log 21 = −3.04. Decision: accept as known iff `E ≤ τ`, else UNKNOWN. `E` itself is the unknown-ness score for AUROC (positive class = UNKNOWN).
- **How it differs from MSP:** at T = 1, `log MSP = max_c f_c + E`. The two scores differ exactly by the max-logit term. Energy keeps the *magnitude* of the logits, which softmax normalises away; a unit test shows energy changes when a constant is added to all logits, while MSP does not.
- **Same logits as MSP:** no new inference was run. The Stage-4 fp32 logits were reused after verifying image order against the locked manifests and the checkpoint hash. Recomputing MSP from them reproduces every Stage-4 metric to within {R['stage4_msp_reproduction_max_abs_diff']:.1e}.

## Temperature and threshold selection (validation only)

The protocol was written into `config.yaml` **before** any energy value was computed:
- **Primary:** T = 1, fixed a priori, the paper's default, using no unknown data.
- **Threshold:** τ = the energy that accepts 95% of **known validation** images, the same rule as MSP's threshold. The MSP threshold (0.991675) is **not** reused, because the scores are on different scales.
- **Secondary:** T* = the grid value maximising AUROC of known validation vs *validation-unknown* images. Stage 2 permits validation-unknowns for this kind of setting.

| T | τ (energy) | Known-val acceptance | Misclassified known-val rejected | AUROC misclassified-vs-correct (val) | AUROC known-val vs validation-unknown |
|---:|---:|---:|---:|---:|---:|
{temp_rows}
| MSP (reference) | {thr['msp_validation_reference']['threshold_msp']:.6f} | 95.02% | 82.86% | — | {F(thr['msp_validation_reference']['auroc_known_val_vs_validation_unknown'])} |

- **Temperature has little effect** on validation ranking for T ≤ 1.5. At large T (5.0) the score approaches −T·log 21 − mean(logits), which discards the max-logit information, and validation AUROC falls.
- **The secondary T\\* = {thr['secondary']['temperature']}** differs from T = 1 by {pv[str(thr['secondary']['temperature'])]['auroc_known_val_vs_validation_unknown'] - pv[str(T)]['auroc_known_val_vs_validation_unknown']:+.4f} validation AUROC.
- **Selected:** T = {T}, accept as known iff **E ≤ {tau:.4f}** (`results/metrics/osr_energy_threshold.json`, created {thr['created_utc']}, before the single test scoring logged in `results/test_runs.log`).

**Protocol check before test scoring:** {'; '.join(f'{k}: {v}' for k, v in R['protocol_check'].items())}.

## Hard same-crop results (Experiment B, primary)

| Method | Hard AUROC | AUPR-Out | FPR@95TPR | Unknown recall | Unknown precision | Unknown F1 | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|
{cmp_row('MSP (Stage 4)', M['B_hard_same_crop'])}
{cmp_row(f'Energy (T={T})', E['B_hard_same_crop'])}

**Paired bootstrap** over identical images: {bB['_meta']['n_boot']} resamples, known and unknown resampled separately, thresholds fixed at their validation values. A = MSP, B = Energy.

| Metric | MSP [95% CI] | Energy [95% CI] | Energy − MSP [95% CI] | Resamples with Energy > MSP | Verdict |
|---|---:|---:|---:|---:|---|
{boot_rows('B_hard_same_crop')}

For FPR@95TPR lower is better; for the other metrics higher is better. The verdict uses the interval of the *difference*. Overlapping marginal intervals are not used as a test.

## Known-class performance (PlantVillage known test, n = {M['B_hard_same_crop']['n_known']:,})

| Method | Known accuracy (closed-set) | Known acceptance | False rejection | Accuracy among accepted |
|---|---:|---:|---:|---:|
| Closed-set classifier | {P(cs['known_accuracy_pv'])} | 100.00% | 0.00% | {P(cs['known_accuracy_pv'])} |
| MSP | {P(M['B_hard_same_crop']['known_closed_set_accuracy'])} | {P(M['B_hard_same_crop']['known_acceptance_rate'])} | {P(M['B_hard_same_crop']['false_rejection_rate'])} | {P(M['B_hard_same_crop']['known_accuracy_among_accepted'])} |
| Energy (T={T}) | {P(E['B_hard_same_crop']['known_closed_set_accuracy'])} | {P(E['B_hard_same_crop']['known_acceptance_rate'])} | {P(E['B_hard_same_crop']['false_rejection_rate'])} | {P(E['B_hard_same_crop']['known_accuracy_among_accepted'])} |

Classification accuracy is identical by construction: neither score changes the argmax. The two methods differ only in *which* images they reject. "Accuracy among accepted" shows whether a method preferentially rejects the classifier's own mistakes.

## Easy unknown results (Experiment A) and all conditions

Each cell shows MSP → Energy. The last column is the paired-bootstrap Energy − MSP AUROC difference.

| Condition | AUROC | FPR@95TPR | Unknown recall | Unknown F1 | Known acceptance | ΔAUROC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
{all_rows}

**A1 per unseen crop:**

| Unseen-crop class | Images | MSP AUROC | Energy AUROC | MSP detection | Energy detection |
|---|---:|---:|---:|---:|---:|
{rows_a}

## PlantDoc results (Experiment C, separate domain-shift condition)

See the C rows in the table above, and figure `plantdoc_energy_vs_msp.png`. The known-field closed-set accuracy is unchanged at {P(cs['known_accuracy_plantdoc_field'])}, the same model. At its own validation threshold Energy accepts {P(E['C_plantdoc_all_unknown']['known_acceptance_rate'])} of PlantDoc known-field images, vs MSP's {P(M['C_plantdoc_all_unknown']['known_acceptance_rate'])}. Verdicts:
- all unknown: {verdict(boot['C_plantdoc_all_unknown'], 'auroc')};
- unknown disease on a known crop: {verdict(boot['C_plantdoc_near_unknown'], 'auroc')};
- unknown crop: {verdict(boot['C_plantdoc_far_unknown'], 'auroc')}.

## Per-disease results (Experiment B)

| Unknown disease | Images | MSP AUROC | Energy AUROC | ΔAUROC [95% CI] | MSP detection | Energy detection | Δdetection [95% CI, pp] |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows_b}

Energy has the higher point-estimate AUROC for {n_improved} of {len(pb)} hard-unknown diseases.

{deep_block('Grape___Esca_(Black_Measles)')}

{deep_block('Corn_(maize)___Northern_Leaf_Blight')}

Figure: `grape_esca_msp_vs_energy.png`. It shows the MSP and energy distributions against the look-alike known class and all other known classes, plus a per-image MSP-vs-energy scatter.

## MSP vs Energy — what differs

- **The scores are highly rank-correlated:** Spearman ρ between `1 − MSP` and energy on the Experiment-B pool = {fa['spearman_msp_u_vs_energy_B_pool']:.4f}.
- **Accept/reject decisions on hard unknowns largely coincide:**
  - both accept {ov['both_accept']:,};
  - both reject {ov['both_reject']:,};
  - Energy-only accept {ov['energy_only_accept']:,};
  - MSP-only accept {ov['msp_only_accept']:,}.
- **On validation, Energy rejects fewer of the classifier's own mistakes:** {P(pv[str(T)]['misclassified_val_rejected'])} vs MSP 82.86% at equal 95% known acceptance. Energy rewards large logits even when two classes compete.

## Failure analysis (Energy-accepted hard unknowns)

Energy accepts {fa['accepted_unknowns']:,} of {fa['accepted_unknowns'] + fa['rejected_unknowns']:,} hard-unknown images as known:
- {P(fa['accepted_predicted_same_crop'])} are predicted as a same-crop disease;
- {P(fa['accepted_predicted_designated_lookalike'])} as the Stage-2 look-alike class;
- {P(fa['accepted_predicted_healthy'])} as a healthy class.

By unknown class: {', '.join(f'{short(k)} {v}' for k, v in fa['accepted_by_class'].items())}.

| Unknown disease | Accepted as | Count |
|---|---|---:|
{top}

**Most confident Energy failures** (lowest energy among accepted unknowns):

| True unknown | Predicted | MSP | Energy |
|---|---|---:|---:|
{worst}

**Image characteristics:** mean brightness / border mean / border std are {st['accepted']['brightness']:.1f} / {st['accepted']['border_mean']:.1f} / {st['accepted']['border_std']:.1f} for accepted unknowns vs {st['rejected']['brightness']:.1f} / {st['rejected']['border_mean']:.1f} / {st['rejected']['border_std']:.1f} for rejected ones. These are small differences under a shared PlantVillage background.

**Plausible explanation, not proven:** the same-crop / look-alike pattern seen with MSP persists. Both scores are functions of the same 21 logits, and those logits are produced by features trained only to separate the known classes. If an unknown disease activates its look-alike's logit strongly, both the softmax maximum and the log-sum-exp are high. A score computed from these logits cannot recover information the features never encoded.

## Secondary temperature variant

{sec_txt}

## Limitations

1. **Same frozen logits:** Energy and MSP are two read-outs of one classifier. Neither can exceed the information in its 21 logits.
2. **One model, one seed:** the bootstrap captures test-sampling uncertainty only, not training variability.
3. **Fixed thresholds in the bootstrap:** they are not re-estimated per resample, so validation-threshold uncertainty is not included.
4. **The temperature grid is small** (6 values) and was chosen with validation data only. The primary T was fixed a priori.
5. **PlantVillage is controlled; PlantDoc adds domain shift and label noise.** Experiment C mixes new imaging conditions with new diseases.
6. **Potential residual same-leaf similarity** (Stage 2) affects the known test set; the strict-known row in the JSON (`B_strict_known_sensitivity`) gives the sensitivity.

## Conclusion

- {headline}: Energy − MSP = {bB['auroc']['diff_B_minus_A']:+.4f}, 95% CI {ci(bB['auroc']['diff_ci95'])}.
- **At the validation-selected thresholds:**
  - unknown detection: {verdict(bB, 'unknown_recall')} ({P(bB['unknown_recall']['A'])} → {P(bB['unknown_recall']['B'])});
  - unknown F1: {verdict(bB, 'unknown_f1')};
  - known acceptance: {verdict(bB, 'known_acceptance_rate')}.
- **Matched operating point:** on test, the two validation thresholds accept different shares of known images ({P(M['B_hard_same_crop']['known_acceptance_rate'])} for MSP, {P(E['B_hard_same_crop']['known_acceptance_rate'])} for Energy). So part of the detection gain comes from Energy rejecting more of *everything*. At equal 95% known acceptance on test (FPR@95TPR), detection is {P(1 - M['B_hard_same_crop']['fpr_at_95tpr'])} (MSP) vs {P(1 - E['B_hard_same_crop']['fpr_at_95tpr'])} (Energy): {verdict(bB, 'fpr_at_95tpr')}, difference in FPR@95TPR {ci(bB['fpr_at_95tpr']['diff_ci95'], pct=True)} pp. The matched-point gain is the fairer measure of the better *ranking*.
- **Grape Esca:** AUROC {F(dd['Grape___Esca_(Black_Measles)']['msp']['auroc_vs_all_known'])} → {F(dd['Grape___Esca_(Black_Measles)']['energy']['auroc_vs_all_known'])} ({verdict(dd['Grape___Esca_(Black_Measles)']['bootstrap'], 'auroc')}); detection {P(dd['Grape___Esca_(Black_Measles)']['msp']['detection'])} → {P(dd['Grape___Esca_(Black_Measles)']['energy']['detection'])} ({verdict(dd['Grape___Esca_(Black_Measles)']['bootstrap'], 'unknown_recall')}). Under **both** scores Esca stays at chance level: **Energy does not fix the hardest failure.**
- **Corn NLB:** AUROC {F(dd['Corn_(maize)___Northern_Leaf_Blight']['msp']['auroc_vs_all_known'])} → {F(dd['Corn_(maize)___Northern_Leaf_Blight']['energy']['auroc_vs_all_known'])} ({verdict(dd['Corn_(maize)___Northern_Leaf_Blight']['bootstrap'], 'auroc')}), but detection at the validation threshold is {P(dd['Corn_(maize)___Northern_Leaf_Blight']['msp']['detection'])} → {P(dd['Corn_(maize)___Northern_Leaf_Blight']['energy']['detection'])} ({verdict(dd['Corn_(maize)___Northern_Leaf_Blight']['bootstrap'], 'unknown_recall')}). The ranking improves only among images that remain on the accepted side of the threshold.
- **Easy unknowns get worse:**
  - A1: AUROC {F(M['A1_unseen_crops']['auroc'])} → {F(E['A1_unseen_crops']['auroc'])}, {verdict(boot['A1_unseen_crops'], 'auroc')};
  - A2: {F(M['A2_non_plant_textures']['auroc'])} → {F(E['A2_non_plant_textures']['auroc'])}, {verdict(boot['A2_non_plant_textures'], 'auroc')}.

  Energy is therefore **not** uniformly better than MSP. It trades easy-unknown performance for hard-unknown performance.
- **PlantDoc:** {verdict(boot['C_plantdoc_all_unknown'], 'auroc')} (AUROC {F(M['C_plantdoc_all_unknown']['auroc'])} → {F(E['C_plantdoc_all_unknown']['auroc'])}). Both scores react to the domain shift.

Swapping the post-hoc logit score does not resolve the core failure: unseen diseases that the network maps onto a same-crop look-alike. This points to the *representation*, not the scoring rule, as the limiting factor. Feature-space methods are the natural next test, and must be evaluated under the same protocol.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_5_ENERGY_OSR.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
