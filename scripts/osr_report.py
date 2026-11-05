"""Build results/reports/STAGE_4_MSP_OSR.md from results/metrics/osr_msp_results.json (no model, no data pass).

    python scripts/osr_msp.py report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import resolve  # noqa: E402


def short(c: str) -> str:
    return (c.replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("Cherry_(including_sour)", "Cherry")
             .replace("___", " · ").replace("Cercospora_leaf_spot Gray_leaf_spot", "Gray leaf spot").replace("_", " ")
             .replace("Spider mites Two-spotted spider mite", "Spider mites").replace("Tomato Yellow Leaf Curl Virus", "YLCV")
             .replace("Leaf blight (Isariopsis Leaf Spot)", "Leaf blight").replace("Haunglongbing (Citrus greening)", "HLB")
             .replace("dtd/", "DTD "))


P = lambda v: "—" if v is None or v != v else f"{100 * v:.2f}%"  # noqa: E731
F = lambda v, d=4: "—" if v is None or v != v else f"{v:.{d}f}"  # noqa: E731


def write_report(cfg) -> None:
    R = json.loads((resolve(cfg["paths"]["metrics"]) / "osr_msp_results.json").read_text())
    thr, res, cs = R["threshold"], R["results"], R["closed_set_system"]
    tau = thr["threshold_msp"]
    B, A1, A2 = res["B_hard_same_crop"], res["A1_unseen_crops"], res["A2_non_plant_textures"]
    A1b, Bs = res["A1_unseen_crops_balanced300"], res["B_strict_known_sensitivity"]
    C, Cn, Cf = res["C_plantdoc_all_unknown"], res["C_plantdoc_near_unknown"], res["C_plantdoc_far_unknown"]
    near_cfg = cfg["osr_protocol"]["unknown_near"]
    s_val = thr["strategies"]

    # ---------- per unknown class (B)
    pb = sorted(B["per_unknown_class"], key=lambda r: r["unknown_recall"])
    hardest, easiest = pb[0], pb[-1]
    rows_b = "\n".join(f"| {short(r['class_name'])} | {r['crop'].replace('Corn_(maize)', 'Corn')} | {r['images']:,} | "
                       f"{P(r['unknown_recall'])} | {P(r['unknown_precision'])} | {P(r['unknown_f1'])} | "
                       f"{F(r['mean_msp'])} | {F(r['median_msp'])} | {F(r['auroc_vs_known'])} | "
                       f"{', '.join(short(x) for x in near_cfg[r['class_name']]['resembles'])} |" for r in pb)
    pa = sorted(A1["per_unknown_class"], key=lambda r: r["unknown_recall"])
    rows_a = "\n".join(f"| {short(r['class_name'])} | {r['images']:,} | {P(r['unknown_recall'])} | {F(r['median_msp'])} | "
                       f"{F(r['auroc_vs_known'])} |" for r in pa)

    def row(name, r, closed=False):
        if closed:
            return f"| {name} | {P(cs['known_accuracy_pv'])} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        return (f"| {name} | {P(r['known_closed_set_accuracy'])} | {P(r['unknown_recall'])} | {P(r['unknown_precision'])} | "
                f"{P(r['unknown_f1'])} | {F(r['auroc'])} | {F(r['aupr_out'])} | {P(r['fpr_at_95tpr'])} | {F(r['oscr'])} |")

    def full(name, r):
        return (f"| {name} | {r['n_known']:,} / {r['n_unknown']:,} | {P(r['known_acceptance_rate'])} | "
                f"{P(r['known_accuracy_among_accepted'])} | {P(r['unknown_recall'])} | {P(r['unknown_precision'])} | "
                f"{P(r['unknown_f1'])} | {P(r['false_known_rate'])} | {P(r['open_set_accuracy'])} | "
                f"{F(r['open_set_macro_f1'])} | {F(r['auroc'])} | {F(r['aupr_out'])} | {P(r['fpr_at_95tpr'])} | {F(r['oscr'])} |")

    # ---------- closed-set failure demo
    cfs = pd.DataFrame(R["closed_set_failure_summary"])
    cfs_b = cfs[cfs["role"] == "unknown_near"]
    rows_cf = "\n".join(f"| {short(r.unknown_class)} | {r.images:,} | {short(r.most_predicted_known)} ({P(r.share_most_predicted)}) | "
                        f"{P(r.share_predicted_same_crop)} | {F(r.median_msp)} | {P(r['share_msp_ge_0.99'])} | "
                        f"{P(r['share_msp_ge_0.999'])} |" for _, r in cfs_b.iterrows())
    allu = cfs[cfs["role"].isin(["unknown_near", "unknown_far_crop", "unknown_far_texture"])]
    w99 = (allu["share_msp_ge_0.99"] * allu["images"]).sum() / allu["images"].sum()
    demo = pd.read_csv(resolve(cfg["paths"]["tables"]) / "osr_closed_set_failure_examples.csv")
    rows_demo = "\n".join(f"| {short(r.true_unknown_disease)} | {short(r.top1_predicted)} | {100 * r.confidence_msp:.2f}% | "
                          f"{'; '.join(short(x) for x in r.top3.split(';'))} | {r.msp_decision} |" for r in demo.itertuples())

    # ---------- per known class
    pk = pd.DataFrame(R["per_known_class"]).sort_values("false_rejection_rate", ascending=False)
    rows_pk = "\n".join(f"| {short(r.class_name)} | {r.images} | {P(r.closed_set_accuracy)} | {r.accepted} | {r.rejected} | "
                        f"{P(r.false_rejection_rate)} | {P(r.accuracy_among_accepted)} |" for r in pk.itertuples())

    # ---------- examples
    ex = pd.read_csv(resolve(cfg["paths"]["tables"]) / "osr_msp_unknown_examples.csv")
    rows_ex = "\n".join(f"| {r.rule} | {short(r.true_unknown_disease)} | {short(r.classifier_prediction)} | "
                        f"{r.classifier_confidence_msp:.6f} | {r.msp_decision} | {r.final_result} |" for r in ex.itertuples())

    fa = R["failure_analysis"]
    stats = fa["image_statistics_mean"]
    top_acc = "\n".join(f"| {short(d['class_name'])} | {short(d['pred_class'])} | {d['count']} |"
                        for d in fa["accepted_top_predictions"][:10])
    calib = thr["calibration_known_validation"]
    rp = R["reproduction_check"]

    # ---------- data-conditional sentences
    acc_pairs = pd.DataFrame(fa["accepted_top_predictions"])
    by_cls = pd.DataFrame(B["per_unknown_class"]).assign(accepted=lambda d: (d["images"] * (1 - d["unknown_recall"])).round())
    by_cls = by_cls.sort_values("accepted", ascending=False)
    top2 = by_cls.head(2)
    top2_share = top2["accepted"].sum() / fa["accepted_unknowns"]
    cross = acc_pairs[[a.split("___")[0] != b.split("___")[0] for a, b in zip(acc_pairs["class_name"], acc_pairs["pred_class"])]]
    cross_txt = ("; the largest cross-crop exception is " + ", ".join(
        f"{short(r.class_name)} → {short(r.pred_class)} ({r.count})" for r in cross.head(1).itertuples())) if len(cross) else ""
    hs = fa["accepted_predicted_healthy"]
    healthy_txt = (f"**Healthy-looking leaves: not supported as a main cause in this run.** Only {P(hs)} of accepted "
                   "unknowns were assigned a *healthy* class. Mild-symptom images exist (Stage 2), but the dominant "
                   "failure is not absorption into healthy classes." if hs < 0.10 else
                   f"**Healthy-looking leaves:** {P(hs)} of accepted unknowns were assigned a *healthy* class. This is "
                   "consistent with mild-symptom images being absorbed by the crop's healthy class.")
    ratio = lambda r: r["n_unknown"] / r["n_known"]  # noqa: E731
    gap = A1["auroc"] - B["auroc"]
    outcome = ("**Outcome A** (MSP detects unknown diseases effectively)" if B["unknown_recall"] >= 0.9 and B["auroc"] >= 0.95 else
               "**Outcome B** (MSP works for easy unknowns but struggles on same-crop diseases)"
               if A1["auroc"] - B["auroc"] >= 0.03 or A1["unknown_recall"] - B["unknown_recall"] >= 0.1 else
               "**Outcome C** (MSP performs poorly because the classifier is confident on unseen diseases)")

    report = f"""# Stage 4 — Open-Set Recognition with Maximum Softmax Probability (MSP)

*Generated by `scripts/osr_msp.py report` from `results/metrics/osr_msp_results.json`; every number below comes from that run.*

## Research question

Can a conventional closed-set classifier, the frozen Stage-3 ResNet-50, recognise that an image belongs to an **unseen disease class** instead of confidently assigning it to a known disease, using only its maximum softmax probability?

## Method

- **Score:** `MSP(x) = max_c softmax(logits(x))_c` over the 21 known classes. The model, weights, preprocessing and class mapping are exactly those of Stage 3. Checkpoint SHA-256: `{R['checkpoint_sha256'][:16]}…`. The classifier has **no** unknown output and was never trained on unknown images.
- **Decision:** accept as *known*, and output the argmax disease, iff `MSP ≥ τ`; otherwise output **UNKNOWN**.
- **What a low MSP means:** the classifier is *uncertain*. Uncertainty can also come from hard, blurry or atypical known images, so a rejection is not proof that a disease is new.
- **Numerics:** MSP is often within 1e-6 of 1.0. The unknown-ness score `u = 1 − MSP` is therefore computed in float64 as the sum of the non-maximum probabilities (no `1 − x` cancellation), from fp32 logits. Stage 3 used mixed precision. The fp32 re-run reproduces Stage 3's known-test predictions with **{rp['argmax_mismatches_vs_stage3']} argmax mismatches** out of {rp['n']:,} (max |ΔMSP| = {rp['max_abs_msp_diff_vs_stage3_amp']:.2e}).

## Experimental protocol (locked in Stage 2; not changed)

| Experiment | Known images | Unknown images | Unknown definition |
|---|---:|---:|---|
| **B — hard, same crop (primary)** | {B['n_known']:,} PlantVillage known test | {B['n_unknown']:,} | Tomato Septoria, Tomato Target Spot, Apple Black rot, Grape Esca, Corn Northern Leaf Blight. All on crops whose other diseases *are* known |
| A1 — unseen crops | {A1['n_known']:,} | {A1['n_unknown']:,} | 9 classes from 7 crops never seen (also reported on the 300-per-class balanced subset) |
| A2 — non-plant | {A2['n_known']:,} | {A2['n_unknown']:,} | DTD textures |
| C — cross-domain (separate) | {C['n_known']:,} PlantDoc known-field | {C['n_unknown']:,} | PlantDoc field images of unknown diseases/crops |

**Validation-unknown classes:** Tomato mosaic virus and Cherry ×2. They appear in no test set, and were used only to *report* the threshold's behaviour on validation.

Experiments are **never pooled** into one number.

**Protocol check, run before any test image was scored:**
- Known classes in training: {R['protocol_check']['Known classes in training']}
- Unknown classes in training: {R['protocol_check']['Unknown classes in training']}
- Threshold selected from validation: {R['protocol_check']['Threshold selected from validation']}
- Final test used for threshold selection: {R['protocol_check']['Final test used for threshold selection']}
- Unknown test images used during training: {R['protocol_check']['Unknown test images used during training']}
- Unknown test images used during threshold selection: {R['protocol_check']['Unknown test images used during threshold selection']}
- Baseline checkpoint unchanged: {R['protocol_check']['Baseline checkpoint unchanged']}

## Threshold selection (validation only)

The decision rule was fixed before any test image was scored (`results/metrics/osr_msp_threshold.json`, created {thr['created_utc']}). Strategies considered, all on the **{thr['validation']['n_known']:,} known validation images**:

| Strategy | Rule | τ (MSP) | Known-val acceptance | Misclassified val rejected | Correct val rejected | Validation-unknown rejected (reported only) |
|---|---|---:|---:|---:|---:|---:|
| A — fixed grid (characterisation) | e.g. τ = 0.90 / 0.99 / 0.999 | {' / '.join(F(x['threshold_msp'], 3) for x in s_val['A_fixed_grid'] if x['threshold_msp'] in (0.9, 0.99, 0.999))} | {' / '.join(P(x['known_val_acceptance']) for x in s_val['A_fixed_grid'] if x['threshold_msp'] in (0.9, 0.99, 0.999))} | {' / '.join(P(x['misclassified_val_rejected']) for x in s_val['A_fixed_grid'] if x['threshold_msp'] in (0.9, 0.99, 0.999))} | {' / '.join(P(x['correct_val_rejected']) for x in s_val['A_fixed_grid'] if x['threshold_msp'] in (0.9, 0.99, 0.999))} | {' / '.join(P(x['validation_unknown_rejected']) for x in s_val['A_fixed_grid'] if x['threshold_msp'] in (0.9, 0.99, 0.999))} |
| **B — validation percentile (selected)** | accept {P(s_val['B_validation_percentile']['target_known_acceptance'])} of known validation | **{F(s_val['B_validation_percentile']['threshold_msp'], 6)}** | {P(s_val['B_validation_percentile']['known_val_acceptance'])} | {P(s_val['B_validation_percentile']['misclassified_val_rejected'])} | {P(s_val['B_validation_percentile']['correct_val_rejected'])} | {P(s_val['B_validation_percentile']['validation_unknown_rejected'])} |
| C — error-aware | maximise (misclassified rejected − correct rejected) | {F(s_val['C_error_aware']['threshold_msp'], 6)} | {P(s_val['C_error_aware']['known_val_acceptance'])} | {P(s_val['C_error_aware']['misclassified_val_rejected'])} | {P(s_val['C_error_aware']['correct_val_rejected'])} | {P(s_val['C_error_aware']['validation_unknown_rejected'])} |

**Why B:**
- It uses no unknown images at all.
- It is the conventional operating point (95% known acceptance, as in FPR@95TPR).
- It is the rule written into the Stage-2 protocol.
- C depends on only {s_val['C_error_aware']['n_misclassified_val']} misclassified validation images, so its threshold is statistically fragile.
- A's fixed values (0.5, 0.9…) are arbitrary for a model whose correct predictions have median MSP ≈ 1 − 10⁻⁶ (figure `msp_validation_distribution.png`).

**Threshold too low** (e.g. 0.5): almost everything is accepted, so unknowns are confidently mislabelled. **Threshold too high:** many legitimate known images are rejected. See `msp_threshold_analysis.png`.

**Selected:** accept as known iff **MSP ≥ {tau:.6f}**.

## Results

### Closed-set failure (no rejection)

Without a reject option the classifier must name one of the 21 known classes for every image. Across all PlantVillage and DTD unknown test images, **{P(w99)} received MSP ≥ 0.99**.

| Unknown disease (B) | Images | Most frequent prediction (share) | Predicted as same crop | Median MSP | MSP ≥ 0.99 | MSP ≥ 0.999 |
|---|---:|---|---:|---:|---:|---:|
{rows_cf}

**Seeded random examples** (3 per hard unknown class; seed {R['seed']}, not selected by outcome):

| True unknown disease | Top-1 prediction | Confidence | Top-3 | MSP decision |
|---|---|---:|---|---|
{rows_demo}

### Main comparison — Experiment B (hard same-crop unknowns, primary)

| System | Known accuracy | Unknown recall | Unknown precision | Unknown F1 | AUROC | AUPR-Out | FPR@95TPR | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{row("Closed-set classifier", None, closed=True)}
{row("ResNet-50 + MSP (τ = %.6f)" % tau, B)}

The closed-set classifier has no reject option, so unknown metrics are not applicable. By construction it assigns **100%** of unknown images to a known disease.

Known accuracy is identical in both rows: MSP does not change the argmax. It only adds rejection.

### All experiments at the selected threshold

| Experiment | Known / unknown images | Known acceptance | Known accuracy among accepted | Unknown recall (detection) | Unknown precision | Unknown F1 | False known rate | Open-set accuracy | Open-set macro-F1 (22 labels) | AUROC | AUPR-Out | FPR@95TPR | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{full("B hard same-crop", B)}
{full("A1 unseen crops", A1)}
{full("A1 balanced (300/class)", A1b)}
{full("A2 DTD textures", A2)}
{full("B, strict known test (sensitivity)", Bs)}

**Metric conventions:**
- **AUROC and AUPR-Out:** positive = UNKNOWN, score = 1 − MSP. Direction was verified with synthetic tests: perfect separation gives 1.0 and swapped labels give 0.0.
- **FPR@95TPR:** the share of unknowns *accepted* when the threshold accepts 95% of known **test** images. It is a threshold-free ranking metric, not a decision rule.
- **OSCR:** Dhamija et al. 2018, the exact area under correct-classification rate vs false-known rate. Its maximum equals closed-set accuracy.
- **Unknown precision and AUPR-Out depend on the known:unknown ratio of each pool,** which differs across experiments. Compare AUROC, FPR@95TPR and recall across experiments; read precision within an experiment.
- **Why PR curves help here:** unknown:known ratios are {ratio(B):.2f} (B), {ratio(A1):.2f} (A1) and {ratio(A2):.2f} (A2). PR curves show how many *rejections* are actually unknowns at that prevalence. ROC curves hide this.

## Per-unknown-class results (Experiment B)

| Unknown disease | Crop | Images | Unknown recall | Unknown precision* | F1* | Mean MSP | Median MSP | AUROC vs known | Visually related known classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{rows_b}

\\* Rejections are not class-specific, so per-class precision and F1 are computed as if the pool were the full known test set plus that one unknown class.

- **Hardest to reject:** {short(hardest['class_name'])}, detection {P(hardest['unknown_recall'])}, AUROC {F(hardest['auroc_vs_known'])}.
- **Easiest:** {short(easiest['class_name'])}, detection {P(easiest['unknown_recall'])}, AUROC {F(easiest['auroc_vs_known'])}.

**Experiment A1 per class:**

| Unseen-crop class | Images | Unknown recall | Median MSP | AUROC vs known |
|---|---:|---:|---:|---:|
{rows_a}

## Effect on known classes (PlantVillage known test)

An aggressive threshold can "detect" unknowns simply by rejecting legitimate images, so the cost is measured per class.

| Known class | Images | Closed-set accuracy | Accepted | Rejected | False rejection rate | Accuracy among accepted |
|---|---:|---:|---:|---:|---:|---:|
{rows_pk}

Overall, {fa['false_rejections_known']['n']:,} known test images are rejected; {fa['false_rejections_known']['of_which_misclassified']} of them were misclassified anyway.

## Confidence analysis

- **Figures:**
  - `msp_known_vs_unknown.png`: B / A1 / A2 distributions.
  - `msp_unknown_by_class.png`: every unknown class, with the share accepted at τ.
  - `msp_roc_curve.png`, `msp_pr_curve.png`, `msp_threshold_analysis.png`.
- **Calibration on known validation (no test data used):**
  - ECE = {calib['ece']:.4f} ({calib['n_bins']} bins);
  - mean confidence {calib['mean_confidence']:.4f} vs accuracy {calib['accuracy']:.4f};
  - {calib['bins'][-1]['n']:,} of {thr['validation']['n_known']:,} validation images fall in the top confidence bin.

  The model is well calibrated **on known classes**. That says nothing about its confidence on unknown inputs, which is exactly what MSP relies on.

## Failure analysis

**False-known unknowns (accepted hard unknowns, Experiment B):** {fa['accepted_unknowns']:,} of {fa['accepted_unknowns'] + fa['rejected_unknowns']:,}.
- {P(fa['accepted_predicted_same_crop'])} were predicted as a known disease **of the same crop**.
- {P(fa['accepted_predicted_designated_lookalike'])} were predicted as one of the look-alike classes named *before* the experiment in Stage 2.
- {P(fa['accepted_predicted_healthy'])} were predicted as a **healthy** class.

Most frequent accepted (unknown → predicted known) pairs:

| Unknown disease | Accepted as | Count |
|---|---|---:|
{top_acc}

**Plausible explanations:** these are consistent with the evidence above but not proven.
- **Symptom similarity (best supported).** {P(fa['accepted_predicted_same_crop'])} of accepted unknowns go to a same-crop disease, and {P(fa['accepted_predicted_designated_lookalike'])} go to the look-alike classes that were written down in Stage 2 *before* any result{cross_txt}.
  - The failures are concentrated: {short(top2.iloc[0]['class_name'])} and {short(top2.iloc[1]['class_name'])} alone account for {P(top2_share)} of all accepted unknowns.
  - A plausible explanation is that the features learned to separate the 21 known classes do not encode what distinguishes these diseases from their look-alike (e.g. Esca's interveinal striping vs Black rot's lesions). The network therefore places them inside the look-alike's region with near-certain confidence.
- {healthy_txt}
- **Overconfidence outside the training distribution:** softmax normalises over known classes only, so a new pattern that partly activates one known class can still produce MSP ≈ 1.
- **Background and lighting:** mean brightness / border mean / border std are {F(stats['accepted']['brightness'], 1)} / {F(stats['accepted']['border_mean'], 1)} / {F(stats['accepted']['border_std'], 1)} for accepted unknowns vs {F(stats['rejected']['brightness'], 1)} / {F(stats['rejected']['border_mean'], 1)} / {F(stats['rejected']['border_std'], 1)} for rejected ones. All images share the PlantVillage board, so background is not a strong separator in B.

**False-unknown knowns:** {fa['false_rejections_known']['n']:,} known test images were rejected, most in {', '.join(f'{short(k)} ({v})' for k, v in list(fa['false_rejections_known']['by_class'].items())[:4])}. Known classes with low per-class MSP pay the largest rejection cost (table above).

**Representative unknown images.** The selection rule is fixed: the highest-confidence failure, the lowest- and median-confidence unknown, one random successful rejection, and one random image per class (seeded). This is not cherry-picked. Figure: `msp_unknown_examples.png`.

| Rule | True unknown | Classifier prediction | Confidence (MSP) | MSP decision | Result |
|---|---|---|---:|---|---|
{rows_ex}

## Cross-domain: Experiment C (PlantDoc), reported separately

| Condition | Known-field / unknown-field | Known closed-set accuracy | Known acceptance | Unknown recall | Unknown precision | AUROC | AUPR-Out | FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C: all unknown-field | {C['n_known']:,} / {C['n_unknown']:,} | {P(C['known_closed_set_accuracy'])} | {P(C['known_acceptance_rate'])} | {P(C['unknown_recall'])} | {P(C['unknown_precision'])} | {F(C['auroc'])} | {F(C['aupr_out'])} | {P(C['fpr_at_95tpr'])} |
| C: unknown disease, known crop | {Cn['n_known']:,} / {Cn['n_unknown']:,} | {P(Cn['known_closed_set_accuracy'])} | {P(Cn['known_acceptance_rate'])} | {P(Cn['unknown_recall'])} | {P(Cn['unknown_precision'])} | {F(Cn['auroc'])} | {F(Cn['aupr_out'])} | {P(Cn['fpr_at_95tpr'])} |
| C: unknown crop | {Cf['n_known']:,} / {Cf['n_unknown']:,} | {P(Cf['known_closed_set_accuracy'])} | {P(Cf['known_acceptance_rate'])} | {P(Cf['unknown_recall'])} | {P(Cf['unknown_precision'])} | {F(Cf['auroc'])} | {F(Cf['aupr_out'])} | {P(Cf['fpr_at_95tpr'])} |

**Domain shift is visible on the known side.**
- Closed-set accuracy on PlantDoc known-field images is {P(cs['known_accuracy_plantdoc_field'])}, vs {P(cs['known_accuracy_pv'])} on PlantVillage.
- At the PlantVillage-chosen τ, only {P(C['known_acceptance_rate'])} of known-field images are accepted, vs {P(B['known_acceptance_rate'])} on PlantVillage.

So under domain shift MSP rejects many *known* field images too. Rejection is driven by unfamiliar imaging conditions, not only by new diseases. The threshold was **not** re-tuned for PlantDoc. Figure: `plantdoc_msp_known_vs_unknown.png`.

## Limitations

1. **MSP is a simple baseline.** It uses one scalar from the classifier and no feature-space information.
2. **Softmax confidence can be overconfident outside the training classes.** Calibration on *known* classes (ECE above) does not imply sensible confidence on unknowns.
3. **Same-crop, visually similar unknowns are the hardest case.** Detection depends on the specific disease; see the per-class table.
4. **PlantVillage is controlled.** The board, light and single leaf mean B measures semantic novelty under ideal imaging, not field conditions.
5. **PlantDoc introduces domain shift and label noise.** Experiment C mixes "new disease" with "new imaging conditions", and its known-field accuracy is low.
6. **Potential residual same-leaf similarity** (Stage 2) affects only the known test set. The strict-known sensitivity row (B, strict) shows its effect on the OSR metrics.
7. **Validation choices:** one threshold strategy, one seed, one model. The threshold rule (95% known acceptance) is a convention, not an optimum.

## Conclusion

- **Experiment B (hard same-crop):** AUROC {F(B['auroc'])}, FPR@95TPR {P(B['fpr_at_95tpr'])}. At τ = {tau:.6f}, MSP detects {P(B['unknown_recall'])} of unseen same-crop diseases while accepting {P(B['known_acceptance_rate'])} of known images, so {P(B['false_known_rate'])} of unseen-disease images are still confidently given a known disease name.
- **Experiment A1 (unseen crops):** AUROC {F(A1['auroc'])}, detection {P(A1['unknown_recall'])}. The AUROC gap A1 − B is {gap:+.4f}.
- **Experiment A2 (non-plant textures):** AUROC {F(A2['auroc'])}.
- **Experiment C (PlantDoc, cross-domain):** AUROC {F(C['auroc'])} overall; {F(Cn['auroc'])} for unknown diseases on known crops{' (below 0.5: PlantDoc *known* images receive lower MSP than these unknowns)' if Cn['auroc'] < 0.5 else ''}. Only {P(C['known_acceptance_rate'])} of known-field images are accepted, and closed-set accuracy on them is {P(cs['known_accuracy_plantdoc_field'])}. Under domain shift MSP mostly measures *unfamiliar imaging*, not *unfamiliar disease*.
- **Outcome:** in terms of the pre-registered outcomes these results correspond to {outcome}. The categorisation uses simple fixed cut-offs written in `scripts/osr_report.py` and should be read together with the numbers above.

MSP adds a useful but incomplete reject option to the closed-set model. It does **not** solve open-set recognition, and its failures concentrate on unknown diseases that resemble a known disease of the same crop. This motivates the stronger feature-space and energy-based methods planned for Stage 5.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_4_MSP_OSR.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
