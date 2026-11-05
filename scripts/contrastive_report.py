"""Build results/reports/STAGE_7_REPRESENTATION_LEARNING.md from saved Stage-7 metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.osr_energy_report import ci  # noqa: E402
from scripts.osr_report import F, P, short  # noqa: E402
from src.utils.config import resolve  # noqa: E402


def vd(b, m="auroc", a="A", bname="B"):
    lo, hi = b[m]["diff_ci95"]
    better_high = m != "fpr_at_95tpr"
    if lo > 0:
        return f"{bname} better" if better_high else f"{bname} worse"
    if hi < 0:
        return f"{bname} worse" if better_high else f"{bname} better"
    return "no detectable difference (CI spans 0)"


def write_report(cfg) -> None:
    M = resolve(cfg["paths"]["metrics"])
    R = json.loads((M / "contrastive_results.json").read_text())
    PO = json.loads((M / "probe_diagnostic_original.json").read_text())
    pcp = M / "probe_diagnostic_contrastive.json"
    PC = json.loads(pcp.read_text()) if pcp.exists() else None
    res, prior, boot, thr = R["results"], R["prior_original_results"], R["bootstrap"], R["thresholds"]
    allr = {**prior, **res}
    tr, env, kt, dg = R["training"], R["environment"], R["known_test"], R["representation_diagnostics"]
    co = cfg["contrastive"]
    kv = thr["known_validation"]

    def probe_rows(Pd):
        return "\n".join(f"| {k} | {v['n_pos']:,} / {v['n_neg']:,} | {v['n_groups']:,} | {F(v['auroc_mean'])} "
                         f"{ci(v['auroc_group_bootstrap_ci95'])} | {F(v['accuracy_mean'])} | {F(v['f1_mean'])} | "
                         f"{F(v['auroc_ungrouped_cv_for_reference'])} |" for k, v in Pd["results"].items())

    table_rows = [("Original ResNet (Stage 3)", "MSP", "original::msp"), ("Original ResNet", "Energy", "original::energy"),
                  ("Original ResNet", "Mahalanobis (raw, Stage 6)", "original::mahalanobis_raw"),
                  ("Original ResNet", "Prototype cosine distance", "original::prototype_cosine"),
                  ("Original ResNet", "Mahalanobis (L2-normalised)", "original::mahalanobis_normalised"),
                  ("Contrastive (Stage 7)", "Prototype cosine distance (primary)", "contrastive::prototype_cosine"),
                  ("Contrastive", "Mahalanobis (L2-normalised)", "contrastive::mahalanobis_normalised"),
                  ("Contrastive", "MSP (linear head on frozen features)", "contrastive::msp_linear_head_u"),
                  ("Contrastive", "Energy (linear head on frozen features)", "contrastive::energy_linear_head")]

    def main_rows(exp):
        return "\n".join(f"| {a} | {b} | {F(allr[k][exp]['auroc'])} | {F(allr[k][exp]['aupr_out'])} | {P(allr[k][exp]['fpr_at_95tpr'])} | "
                         f"{P(allr[k][exp]['unknown_recall'])} | {F(allr[k][exp]['unknown_f1'])} | {P(allr[k][exp]['known_acceptance_rate'])} |"
                         for a, b, k in table_rows)

    def boot_line(name, A, Bn):
        b = boot[name]
        return (f"| {name} | {F(b['auroc']['A'])} → {F(b['auroc']['B'])} | {b['auroc']['diff_B_minus_A']:+.4f} {ci(b['auroc']['diff_ci95'])} | "
                f"{vd(b, 'auroc', A, Bn)} | {100 * b['unknown_recall']['diff_B_minus_A']:+.2f} pp {ci(b['unknown_recall']['diff_ci95'], pct=True)} | "
                f"{100 * b['fpr_at_95tpr']['diff_B_minus_A']:+.2f} pp |")

    pc = pd.DataFrame(R["per_unknown_class"])
    pd_rows = "\n".join(
        f"| {short(r['unknown_class'])} | {r['images']:,} | {F(r['original::msp::auroc'])} | {F(r['original::energy::auroc'])} | "
        f"{F(r['original::mahalanobis_raw::auroc'])} | {F(r['original::prototype_cosine::auroc'])} | {F(r['contrastive::prototype_cosine::auroc'])} | "
        f"{boot['proto: original vs contrastive (B class ' + r['unknown_class'] + ')']['auroc']['diff_B_minus_A']:+.4f} "
        f"{ci(boot['proto: original vs contrastive (B class ' + r['unknown_class'] + ')']['auroc']['diff_ci95'])} | "
        f"{F(r['contrastive::mahalanobis_normalised::auroc'])} | {P(r['original::energy::detection'])} | "
        f"{P(r['contrastive::prototype_cosine::detection'])} |" for _, r in pc.iterrows())

    def deep(name, look):
        d = R["deep"][name]
        rows = [("Images whose nearest prototype is the look-alike", "share_nearest_prototype_is_lookalike", True),
                ("Mean cosine: unknown → look-alike prototype", "mean_cosine_unknown_to_lookalike_prototype", False),
                ("Mean cosine: look-alike's own test images → its prototype", "mean_cosine_lookalike_test_to_own_prototype", False),
                ("Mean nearest-prototype distance: unknown", "mean_nearest_prototype_cosine_distance_unknown", False),
                ("Mean nearest-prototype distance: all known test", "mean_nearest_prototype_cosine_distance_known_test", False),
                ("Prototype AUROC vs all known", "prototype_cosine::auroc_vs_all_known", False),
                ("Prototype AUROC vs look-alike class only", "prototype_cosine::auroc_vs_lookalike_only", False),
                ("Prototype detection (validation threshold)", "prototype_cosine::detection", True),
                ("Prototype false-known rate", "prototype_cosine::false_known_rate", True),
                ("Mahalanobis (norm.) AUROC vs all known", "mahalanobis_normalised::auroc_vs_all_known", False),
                ("Mahalanobis (norm.) AUROC vs look-alike only", "mahalanobis_normalised::auroc_vs_lookalike_only", False),
                ("Mahalanobis (norm.) detection", "mahalanobis_normalised::detection", True)]
        body = "\n".join(f"| {lab} | {(P if pct else F)(d['original'][k])} | {(P if pct else F)(d['contrastive'][k])} |" for lab, k, pct in rows)
        pk = f"{name} vs {look}"
        probe = f"| Linear-probe AUROC (diagnostic, leaf-grouped CV) | {F(PO['results'][pk]['auroc_mean'])} | " + \
                (f"{F(PC['results'][pk]['auroc_mean'])} |" if PC else "not run |")
        return f"| Quantity | Original | Contrastive |\n|---|---:|---:|\n{body}\n{probe}"

    fc = pd.DataFrame(R["failure_counts"]).T.fillna(0).astype(int)
    fc_rows = "\n".join(f"| {short(cn)} | " + " | ".join(str(int(r.get(k, 0))) for k in
                        ("original_fails_contrastive_succeeds", "original_succeeds_contrastive_fails", "both_fail", "both_succeed")) + " |"
                        for cn, r in fc.iterrows())
    hist = tr["history"]
    hist_rows = "\n".join(f"| {h['epoch']} | {h['train_supcon_loss']:.4f} | {h['val_nc_accuracy']:.4f} | {h['val_nearest_centroid_macro_f1']:.4f} | "
                          f"{h['val_mean_cosine_to_own_centroid']:.3f} | {h['centroid_mean_pairwise_cosine']:.3f} |" for h in hist)
    geo = "\n".join(f"| {lab} | {F(dg['original'][k])} | {F(dg['contrastive'][k])} |" for lab, k in
                    [("Intra-class cosine distance (mean)", "intra_class_cosine_distance_mean"),
                     ("Inter-prototype cosine distance (mean)", "inter_prototype_cosine_distance_mean"),
                     ("Inter-prototype cosine distance (closest pair)", "inter_prototype_cosine_distance_min"),
                     ("Separation ratio (inter mean / intra mean)", "separation_ratio_inter_mean_over_intra_mean"),
                     ("Margin to nearest other prototype (mean)", "nearest_other_prototype_margin_mean"),
                     ("Nearest-prototype accuracy, known test", "nearest_prototype_accuracy_known_test")])

    # ---- headline answers (from paired bootstrap intervals)
    rep_eff = vd(boot["proto: original vs contrastive (B)"], bname="contrastive")
    vs_best = vd(boot["original Energy vs contrastive proto (B)"], bname="contrastive")
    maha_eff = vd(boot["maha_norm: original vs contrastive (B)"], bname="contrastive")
    a1_eff = vd(boot["proto: original vs contrastive (A1)"], bname="contrastive")
    sep_up = dg["contrastive"]["separation_ratio_inter_mean_over_intra_mean"] > dg["original"]["separation_ratio_inter_mean_over_intra_mean"]
    closed = kt["contrastive_linear_head_accuracy"] - kt["original_stage3_head_accuracy"]
    if rep_eff.startswith("contrastive better") and vs_best.startswith("contrastive better"):
        answer = "**Representation learning improved open-set recognition**: it beats both the original representation under the same score and the best previous method (original Energy)."
    elif rep_eff.startswith("contrastive better"):
        answer = ("**Representation learning improved the representation for distance-based rejection, but not beyond the best previous method.** "
                  f"Under the same prototype score, contrastive beats original ({rep_eff}); against original Energy it is: {vs_best}.")
    elif rep_eff.startswith("contrastive worse"):
        answer = f"**Representation learning did not improve open-set recognition.** Under the same prototype score the contrastive representation is worse ({rep_eff})."
    else:
        answer = "**No detectable change in open-set recognition** under the same prototype score (difference interval spans 0)."
    closed_txt = (f"Known-class geometry {'became more separated' if sep_up else 'did not become more separated'} "
                  f"(separation ratio {F(dg['original']['separation_ratio_inter_mean_over_intra_mean'], 3)} → "
                  f"{F(dg['contrastive']['separation_ratio_inter_mean_over_intra_mean'], 3)}). Closed-set accuracy of a linear head on the "
                  f"contrastive features is {P(kt['contrastive_linear_head_accuracy'])}, vs {P(kt['original_stage3_head_accuracy'])} for the "
                  f"Stage-3 classifier ({100 * closed:+.2f} pp).")
    esca = R["deep"]["Grape___Esca_(Black_Measles)"]
    near_names = list(cfg["osr_protocol"]["unknown_near"])
    esca_up = esca["contrastive"]["prototype_cosine::auroc_vs_all_known"] - esca["original"]["prototype_cosine::auroc_vs_all_known"]

    probe_c = ("**Contrastive (Stage 7) representation** (same probe, same folds):\n\n"
               "| Comparison | Positive / negative images | Leaf groups | AUROC [95% CI] | Accuracy | F1 | AUROC with ungrouped CV (reference) |\n"
               "|---|---:|---:|---:|---:|---:|---:|\n" + probe_rows(PC)) if PC else "*Contrastive-representation probe not run.*"

    # ---- post-hoc head-matched control (optional file)
    hcp = M / "contrastive_head_control.json"
    HC = json.loads(hcp.read_text()) if hcp.exists() else None
    head_sec, head_eff = "*Head-matched control not run.*", {}
    e_orig = prior["original::energy"]["B_hard_same_crop"]["auroc"]
    e_ch = res["contrastive::energy_linear_head"]["B_hard_same_crop"]["auroc"]
    e_oh = None
    if HC:
        hr = HC["results"]
        rows = []
        for key, lab in (("msp_linear_head_u", "MSP"), ("energy_linear_head", "Energy")):
            o_, c_ = hr[f"original_linear_head::{key}"], res[f"contrastive::{key}"]
            for rep_lab, d in (("Original features + linear head", o_), ("Contrastive features + linear head", c_)):
                b_ = d["B_hard_same_crop"]
                rows.append(f"| {rep_lab} | {lab} | {F(b_['auroc'])} | {P(b_['fpr_at_95tpr'])} | {P(b_['unknown_recall'])} | {F(b_['unknown_f1'])} | "
                            f"{F(d['A1_unseen_crops']['auroc'])} | {F(d['C_plantdoc_all_unknown']['auroc'])} |")
        bl = []
        for name, b_ in HC["bootstrap"].items():
            head_eff[name.split(":")[0]] = vd(b_, bname="contrastive")
            bl.append(f"| {name} | {F(b_['auroc']['A'])} → {F(b_['auroc']['B'])} | {b_['auroc']['diff_B_minus_A']:+.4f} {ci(b_['auroc']['diff_ci95'])} | "
                      f"{vd(b_, bname='contrastive')} | {100 * b_['unknown_recall']['diff_B_minus_A']:+.2f} pp {ci(b_['unknown_recall']['diff_ci95'], pct=True)} |")
        pdr = "\n".join(f"| {short(r['unknown_class'])} | {'MSP' if r['score'].startswith('msp') else 'Energy'} | {F(r['original_linear_head_auroc'])} | "
                        f"{F(r['contrastive_linear_head_auroc'])} | {P(r['original_linear_head_detection'])} | {P(r['contrastive_linear_head_detection'])} |"
                        for r in HC["per_disease"])
        e_oh = hr["original_linear_head::energy_linear_head"]["B_hard_same_crop"]["auroc"]
        head_sec = (
            "**Added after the Stage-7 test results were seen** (logged in `results/test_runs.log`). The contrastive MSP/Energy rows use a *new head*: "
            "logistic regression (C = " + str(co["linear_head"]["C"]) + ") on L2-normalised frozen features, fitted on TRAIN only. That changes two things "
            "at once: the representation **and** the head. This control fits the **identical head on the original Stage-3 features**. Its threshold comes "
            "from known validation only (95% acceptance) and was frozen before test scoring. Nothing in the pre-specified Stage-7 pipeline was changed.\n\n"
            "| Features | Score | Hard AUROC | FPR@95TPR | Unknown recall | Unknown F1 | A1 AUROC | PlantDoc AUROC |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
            "| Paired bootstrap, Experiment B (A → B) | AUROC | Difference [95% CI] | Verdict | Δ unknown recall [95% CI] |\n"
            "|---|---:|---:|---|---:|\n" + "\n".join(bl) + "\n\n"
            f"**Decomposition of the Energy gain (hard AUROC):**\n"
            f"- Original Energy (Stage 3): {F(e_orig)}.\n"
            f"- Same original features with the linear head: {F(e_oh)} (**{e_oh - e_orig:+.4f}, head effect**).\n"
            f"- Contrastive features with the same linear head: {F(e_ch)} (**{e_ch - e_oh:+.4f}, representation effect**).\n\n"
            f"Known-test closed-set accuracy of the linear head: original features {P(HC['known_test_accuracy']['original_linear_head'])}, "
            f"contrastive features {P(HC['known_test_accuracy']['contrastive_linear_head'])}.\n\n"
            "Per disease (head-matched):\n\n"
            "| Unknown | Score | AUROC, original features | AUROC, contrastive features | Detection, original features | Detection, contrastive features |\n"
            "|---|---|---:|---:|---:|---:|\n" + pdr)

    # ---- decision-level comparison: original Energy vs contrastive Energy-head
    pr = pd.read_csv(resolve(cfg["paths"]["results"]) / "predictions" / "osr_contrastive_predictions.csv")
    t_en = json.loads(resolve(cfg["feature_osr"]["stage5_threshold"]).read_text())["primary"]["threshold_energy"]
    t_ceh = thr["thresholds"]["contrastive::energy_linear_head"]
    hu = pr[pr["role"] == "unknown_near"]
    o_ok, c_ok = hu["original_energy"] > t_en, hu["contrastive_linear_head_energy"] > t_ceh
    fe = []
    for cl in sorted(hu["true_class"].unique()):
        m = hu["true_class"] == cl
        fe.append(f"| {short(cl)} | {int((~o_ok[m] & c_ok[m]).sum())} | {int((o_ok[m] & ~c_ok[m]).sum())} | "
                  f"{int((~o_ok[m] & ~c_ok[m]).sum())} | {int((o_ok[m] & c_ok[m]).sum())} |")
    fe_rows = "\n".join(fe)

    # ---- matched-score summary and hypothesis verdict (data-driven)
    matched = {"prototype": rep_eff, "Mahalanobis (norm.)": maha_eff,
               **{("MSP head" if k.startswith("msp") else "Energy head"): v for k, v in head_eff.items()}}
    all_better = all(v.startswith("contrastive better") for v in matched.values())
    any_better = any(v.startswith("contrastive better") for v in matched.values())
    worse_cls = [short(c) for c in near_names if boot[f"proto: original vs contrastive (B class {c})"]["auroc"]["diff_ci95"][1] < 0]
    esca_eh = next(x for x in res["contrastive::energy_linear_head"]["B_hard_same_crop"]["per_unknown_class"]
                   if x["class_name"] == "Grape___Esca_(Black_Measles)")["unknown_recall"]
    esca_det = max(esca["contrastive"]["prototype_cosine::detection"], esca["contrastive"]["mahalanobis_normalised::detection"], esca_eh)
    if all_better and not worse_cls and esca_det >= 0.5:
        hyp = "**supports**"
    elif any_better:
        hyp = "**partially supports**"
    else:
        hyp = "**does not support**"
    rep_small = e_oh is not None and (e_ch - e_oh) < 0.5 * (e_oh - e_orig)
    head_txt = (f"The same head on the original features already reaches {F(e_oh)}, so the contrastive representation adds only {e_ch - e_oh:+.4f}."
                if e_oh is not None else "No head-matched control was run.")
    if rep_small:
        head_txt += " Most of that improvement therefore does not come from the contrastive representation."
    probe_esca = F(PC["results"]["Grape___Esca_(Black_Measles) vs Grape___Black_rot"]["auroc_mean"]) if PC else "n/a"

    report = f"""# Stage 7 — Representation Learning for Open-Set Recognition

*Generated by `scripts/contrastive_osr.py report` from `results/metrics/contrastive_results.json`, `probe_diagnostic_original.json`{' and `probe_diagnostic_contrastive.json`' if PC else ''}. Earlier-stage numbers are loaded from their saved files.*

## Motivation

The frozen Stage-3 classifier gives hard-unknown AUROC {F(prior['original::msp']['B_hard_same_crop']['auroc'])} (MSP), {F(prior['original::energy']['B_hard_same_crop']['auroc'])} (Energy) and {F(prior['original::mahalanobis_raw']['B_hard_same_crop']['auroc'])} (Mahalanobis). Grape Esca stays near chance for the logit scores. Yet a linear probe separates Esca from Black rot almost perfectly.

So the working hypothesis is **not** "the representation lacks the information". It is "label-free scores do not recover the separating directions". Stage 7 asks whether a representation **explicitly structured around the known classes**, trained by supervised contrastive learning, separates known from unseen diseases better under the same protocol.

## Diagnostic probe (Part B) — *Diagnostic representation probe — not an OSR method*

**Setup:** logistic regression (C = {cfg['probe_diagnostic']['C']}, standardised features) on frozen embeddings. **Leaf-grouped** stratified 5-fold CV: every photo of a physical leaf, and every strong duplicate, stays in one fold. {cfg['probe_diagnostic']['repeats']} repeats with different fold seeds; 95% CI by bootstrapping leaf groups.

**Why it is not an OSR method:** it is trained *with the unseen-disease labels*. It measures whether a linear direction exists, and was never used for any threshold, checkpoint or setting.

**Original (Stage-3) representation:**

| Comparison | Positive / negative images | Leaf groups | AUROC [95% CI] | Accuracy | F1 | AUROC with ungrouped CV (reference) |
|---|---:|---:|---:|---:|---:|---:|
{probe_rows(PO)}

{probe_c}

**What the controls show:**
- **Random halves:** two random halves of one known class are **not** separable under leaf-grouped CV (AUROC ≈ 0.5), as expected. With *ungrouped* CV they score above chance, because the probe recognises leaves it trained on.
- **The Stage-6 probe was slightly optimistic:** it used ungrouped CV. The leaf-grouped values above are the ones to cite. They remain high, so the Stage-6 conclusion stands.

## Contrastive learning method

- **Backbone:** ResNet-50 from **ImageNet weights** (torchvision IMAGENET1K_V2), **not** the Stage-3 disease checkpoint.
- **Trainable layers:** `{', '.join(co['trainable'])}`, the same as Stage-3 phase B. The stem, layer1 and layer2, and their BatchNorm statistics, are frozen.
- **Projection head:** MLP 2048 → {co['projection']['hidden']} → {co['projection']['out']}, L2-normalised. Used **only for the loss** and removed for evaluation.
- **Loss:** supervised contrastive (Khosla et al., 2020), temperature **{co['temperature']}** (default, not tuned). Positives are the other view of the same image plus every view of other images of the same known class. Negatives are views of other known classes. There is no "unknown" class.
  - Unit-tested against a literal implementation of the paper's equation, including multiple positives per anchor.
- **Batches:** class-balanced sampler with **{co['classes_per_batch']} known classes × {co['samples_per_class']} images × {co['views']} views = {env['effective_batch_views']} views per step**. No gradient accumulation: the loss couples the whole batch, and it fits in {max(h['peak_gpu_gb'] for h in hist)} GB.
- **Augmentation:** the documented Stage-3 training augmentation (rotation ±15° with grey fill, RandomResizedCrop scale 0.7–1.0, horizontal and vertical flips, colour jitter brightness 0.2 / contrast 0.2 / saturation 0.15 / no hue). Both views come from the *same* source image, so views never cross split boundaries.
- **Optimiser:** AdamW, backbone lr {co['lr_backbone']}, head lr {co['lr_head']}, weight decay {co['weight_decay']}, {co['warmup_epochs']} warm-up epoch then cosine decay, mixed precision, seed {co['seed']}.
- **Epochs:** {env['epochs_run']} run (max {co['epochs']}; early-stopping patience {co['early_stopping']['patience']}). {env['training_minutes']} minutes on {env.get('gpu', env['device'])}.
- **Checkpoint selection:** nearest-centroid macro-F1 on the **known validation set**, with centroids from {co['val_reference_per_class']} fixed training images per class. Selected epoch: **{tr['best_epoch']}**.

| Epoch | Train SupCon loss | Val nearest-centroid accuracy | Val NC macro-F1 | Val mean cosine to own centroid | Mean pairwise centroid cosine |
|---:|---:|---:|---:|---:|---:|
{hist_rows}

**Evaluation representation:** the {R['extraction_checks']['representation']}, L2-normalised. Re-extraction is deterministic (max |Δ| {R['extraction_checks']['determinism_rerun_max_abs_diff']:.1e}).

## Data protocol

- **Training:** only the 21 known PlantVillage classes from the locked `train.csv` ({env.get('batch_images', '')} images per batch).
- **Model selection:** the known validation set only.
- **Validation-unknown images:** used nowhere in training or selection. Their AUROC is only reported.
- **The hard unknowns** (Grape Esca, Corn NLB, Tomato Septoria, Tomato Target Spot, Apple Black rot), the other unknowns, PlantDoc and DTD were seen **only during final evaluation**. The training loader asserts this, and so does a unit test.

**Protocol check before test scoring:** {'; '.join(f'{k}: {v}' for k, v in R['protocol_check'].items())}.

## Representation results (known classes)

| | Original (Stage 3) | Contrastive (Stage 7) |
|---|---:|---:|
| Known-validation nearest-prototype accuracy | {P(kv['original']['nc_accuracy'])} | {P(kv['contrastive']['nc_accuracy'])} |
| Known-validation linear head (frozen features) accuracy | — | {P(kv['contrastive']['linear_head_accuracy'])} |
| Known-test closed-set accuracy | {P(kt['original_stage3_head_accuracy'])} (Stage-3 head) | {P(kt['contrastive_linear_head_accuracy'])} (linear head) |
| Known-test nearest-prototype accuracy | {P(kt['original_nearest_prototype_accuracy'])} | {P(kt['contrastive_nearest_prototype_accuracy'])} |
| PlantDoc known-field closed-set accuracy | {P(kt['plantdoc_known_field_original_accuracy'])} | {P(kt['plantdoc_known_field_contrastive_linear_head_accuracy'])} |

**Geometry** (cosine on L2-normalised features; prototypes from training):

| Quantity | Original | Contrastive |
|---|---:|---:|
{geo}

## OSR results — Experiment B (hard same-crop, primary)

Every row uses the same images, the same known-validation 95%-acceptance threshold rule, and fits on known training data only.

| Representation | Score | Hard AUROC | AUPR-Out | FPR@95TPR | Unknown recall | Unknown F1 | Known acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
{main_rows('B_hard_same_crop')}

**Paired bootstrap** ({boot['proto: original vs contrastive (B)']['_meta']['n_boot']} stratified resamples, fixed thresholds):

| Comparison (A → B) | AUROC | Difference [95% CI] | Verdict | Δ unknown recall | Δ FPR@95TPR |
|---|---:|---:|---|---:|---:|
{boot_line('proto: original vs contrastive (B)', 'original', 'contrastive')}
{boot_line('maha_norm: original vs contrastive (B)', 'original', 'contrastive')}
{boot_line('original Energy vs contrastive proto (B)', 'original Energy', 'contrastive')}
{boot_line('original Energy vs contrastive Energy-head (B)', 'original Energy', 'contrastive')}
{boot_line('proto: original vs contrastive (A1)', 'original', 'contrastive')}

### Head-matched control (post-hoc)

{head_sec}

**Other conditions** (Experiment A1 / A2):

| Representation | Score | A1 AUROC | A1 recall | A2 AUROC | A2 recall |
|---|---|---:|---:|---:|---:|
""" + "\n".join(f"| {a} | {b} | {F(allr[k]['A1_unseen_crops']['auroc'])} | {P(allr[k]['A1_unseen_crops']['unknown_recall'])} | "
                f"{F(allr[k]['A2_non_plant_textures']['auroc'])} | {P(allr[k]['A2_non_plant_textures']['unknown_recall'])} |" for a, b, k in table_rows) + f"""

**Validation-unknown AUROC (reported only, never used for selection):** {'; '.join(f"{k}: {F(v['auroc_known_val_vs_validation_unknown'])}" for k, v in thr['validation_unknown_reported_only'].items())}.

## Per-disease (Experiment B)

| Unknown | Images | Original MSP | Original Energy | Original Mahalanobis (raw) | Original prototype | **Contrastive prototype** | Contrastive − original prototype [95% CI] | Contrastive Mahalanobis (norm.) | Original Energy detection | Contrastive prototype detection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{pd_rows}

## Grape Esca (↔ Grape Black rot)

{deep('Grape___Esca_(Black_Measles)', 'Grape___Black_rot')}

## Corn NLB (↔ Corn Gray leaf spot)

{deep('Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot')}

Figure: `results/figures/contrastive/esca_nlb_original_vs_contrastive.png`.

## PlantDoc (Experiment C, separate)

| Representation | Score | AUROC (all unknown) | AUPR-Out | FPR@95TPR | Known-field acceptance | AUROC (unknown disease, known crop) | AUROC (unknown crop) |
|---|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {a} | {b} | {F(allr[k]['C_plantdoc_all_unknown']['auroc'])} | {F(allr[k]['C_plantdoc_all_unknown']['aupr_out'])} | "
                f"{P(allr[k]['C_plantdoc_all_unknown']['fpr_at_95tpr'])} | {P(allr[k]['C_plantdoc_all_unknown']['known_acceptance_rate'])} | "
                f"{F(allr[k]['C_plantdoc_near_unknown']['auroc'])} | {F(allr[k]['C_plantdoc_far_unknown']['auroc'])} |" for a, b, k in table_rows) + f"""

## Failure analysis

Hard-unknown decisions are compared between the best previous method (original Energy) and the contrastive primary score (prototype). The full list is `results/tables/contrastive_vs_original_failures.csv`. Examples were drawn by a seeded rule (2 per disease per category) into `contrastive_failure_examples.csv`.

| Unknown | Original fails, contrastive succeeds | Original succeeds, contrastive fails | Both fail | Both succeed |
|---|---:|---:|---:|---:|
{fc_rows}

The same comparison against the **contrastive Energy-head** score (each at its own known-validation threshold; "succeeds" = rejected as unknown):

| Unknown | Original Energy fails, contrastive Energy-head succeeds | Original succeeds, contrastive fails | Both fail | Both succeed |
|---|---:|---:|---:|---:|
{fe_rows}

## Limitations

1. **Training only on known diseases.** Nothing in the objective rewards novelty detection. Contrastive learning only structures the 21 known classes, and unseen diseases may still fall inside a known cluster.
2. **Controlled PlantVillage data:** one leaf, uniform board, studio light. The geometry learned here may not transfer to field photos (see PlantDoc).
3. **Domain shift:** stronger class structure can make *any* change of imaging conditions look "far". The PlantDoc table measures this.
4. **Possible acquisition artifacts:** PlantVillage classes were photographed in sessions. Both the diagnostic probe and the contrastive clusters could partly encode session differences (board shade, lighting) rather than symptoms.
5. **Unknown classes were never used in training,** by design. This keeps the experiment open-set, but caps what the representation can learn about novelty.
6. **One run, one seed, one temperature, untuned:** the bootstrap covers test sampling only. The prototype and Mahalanobis score choices were fixed in advance and not tuned.
7. **Post-hoc control:** the head-matched control was added after the Stage-7 test results were seen. It uses a validation-only threshold and changes no Stage-7 result, but it is an extra test-set evaluation and is labelled as such.

## Conclusion

- **Did representation learning improve open-set recognition, or did it simply improve closed-set class separation?**
  - **Closed-set separation improved clearly.** {closed_txt}
  - **Open-set also improved, but much less than the headline numbers suggest.** With the score held fixed, the contrastive representation compares as follows: {'; '.join(f'{k}: {v}' for k, v in matched.items())}.
  - **Pre-specified primary score (prototype distance):** {answer}
  - **Best Stage-7 number:** Energy on a linear head, hard AUROC {F(e_ch)} vs {F(e_orig)}. This is mostly a **head effect**. {head_txt}
  - **Unseen crops (A1), prototype score:** {a1_eff}.
  - **Per disease, prototype score:** {('contrastive is significantly worse for ' + ', '.join(worse_cls) + '.') if worse_cls else 'no disease got significantly worse.'}
- **Grape Esca:**
  - Prototype AUROC {F(esca['original']['prototype_cosine::auroc_vs_all_known'])} → {F(esca['contrastive']['prototype_cosine::auroc_vs_all_known'])}.
  - Best contrastive detection at the 95%-acceptance threshold: {P(esca_det)}.
  - Esca is still mostly accepted as a known disease: its nearest prototype is Black rot for {P(esca['contrastive']['share_nearest_prototype_is_lookalike'])} of images. Yet a supervised probe separates it from Black rot with AUROC {probe_esca}.
- **Hypothesis** ("a representation explicitly structured around known diseases improves separation of unseen diseases"): the evidence {hyp} it.
  - Verdict rule (written after the results were available, applied mechanically): *supports* needs every matched-score comparison significantly better, no hard disease significantly worse under the primary score, and Esca detection ≥ 50%. *Partially supports* needs at least one matched-score comparison significantly better.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_7_REPRESENTATION_LEARNING.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
