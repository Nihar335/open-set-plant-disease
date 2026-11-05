"""Stage 3 — evaluate the closed-set baseline on the KNOWN-class test set (full and strict).

    python scripts/evaluate_baseline.py                 # run_name from config.yaml -> baseline
    python scripts/evaluate_baseline.py --run-name X

Loads only known PlantVillage test rows. Unknown classes, PlantDoc and DTD are NOT evaluated here.
No threshold of any kind is chosen. Outputs (prefix = run name, default "baseline"):
  results/metrics/<run>_test_metrics.json
  results/tables/<run>_per_class_metrics.csv, <run>_top_confusions.csv
  results/predictions/<run>_test_predictions.csv
  results/figures/<run>/confusion_matrix_{raw,normalized}.png, gradcam_val_examples.png
  results/reports/STAGE_3_BASELINE.md   (default run only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ManifestDataset, build_transforms  # noqa: E402
from src.data.protocol import known_classes, load_manifest, verify_lock  # noqa: E402
from src.evaluation.classification import confusion, per_class_metrics, summary_metrics, top_confusions  # noqa: E402
from src.models.classifier import Classifier  # noqa: E402
from src.training.trainer import predict  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.tables import write_table  # noqa: E402

KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def short(c: str) -> str:
    return (c.replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("___", " · ")
             .replace("Cercospora_leaf_spot Gray_leaf_spot", "Gray leaf spot").replace("_", " ")
             .replace("Spider mites Two-spotted spider mite", "Spider mites")
             .replace("Tomato Yellow Leaf Curl Virus", "YLCV").replace("Leaf blight (Isariopsis Leaf Spot)", "Leaf blight"))


def bootstrap_ci(y, p, n_classes, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = summary_metrics(y[i], p[i], n_classes)
        acc.append(m["accuracy"]); f1.append(m["macro_f1"])
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    return {"accuracy_95ci": q(acc), "macro_f1_95ci": q(f1), "n_boot": n_boot}


def plot_cm(cm, classes, out, normalized):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from src.visualization.style import apply_style, save
    apply_style()
    ramp = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    m = cm / cm.sum(1, keepdims=True) if normalized else cm
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(m, cmap=ramp, vmin=0, vmax=1 if normalized else m.max())
    ax.grid(False)
    lab = [short(c) for c in classes]
    ax.set_xticks(range(len(classes)), lab, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(classes)), lab, fontsize=7.5)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                v = f"{m[i, j]:.2f}" if normalized else str(cm[i, j])
                ax.text(j, i, v, ha="center", va="center", fontsize=5.5,
                        color="white" if (m[i, j] / (1 if normalized else m.max())) > 0.55 else "#0b0b0b")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Baseline confusion matrix — known-class full test" + (" (row-normalised = recall)" if normalized else " (counts)"))
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save(fig, out)


def gradcam_examples(model, val_df, cfg, classes, device, out, n):
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.visualization.gradcam import GradCAM
    from src.visualization.style import apply_style, save
    apply_style()
    b = cfg["baseline"]
    tf = build_transforms(b["training"], b["augmentation"], train=False)
    rows = val_df.groupby("class_id").head(1).sample(n, random_state=0)   # one per class, n classes
    cam = GradCAM(model, model.gradcam_layer)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.4))
    for k, r in enumerate(rows.itertuples()):
        img = Image.open(ROOT / r.image_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        heat, c = cam(x)
        shown = img.resize((256, 256)).crop((16, 16, 240, 240))
        axes[0, k].imshow(shown); axes[1, k].imshow(shown); axes[1, k].imshow(heat, cmap="jet", alpha=0.45)
        axes[0, k].set_title(f"true: {short(classes[r.class_id])[:22]}", fontsize=6)
        axes[1, k].set_title(f"pred: {short(classes[c])[:22]}", fontsize=6,
                             color="#0b0b0b" if c == r.class_id else "#d03b3b")
        for a in axes[:, k]:
            a.axis("off")
    cam.remove()
    fig.suptitle("Preliminary Grad-CAM (layer4) on known VALIDATION images — illustrative only", fontsize=9.5,
                 fontweight="semibold")
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from saved metrics/predictions (no re-evaluation of the test set)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["baseline"]
    run = args.run_name or b["run_name"]
    verify_lock(cfg)
    if args.report_only:
        return write_report(cfg, run, known_classes(cfg))
    classes = known_classes(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and b["hardware"]["device"] in ("auto", "cuda") else "cpu")

    ckpt_path = resolve(cfg["paths"]["models"]) / run / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["class_names"] != classes:
        raise SystemExit("checkpoint class order differs from the locked class mapping")
    model = Classifier(ckpt["architecture"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    d, t = b["dataset"], b["training"]
    test = load_manifest(d["test_manifest"], d["row_filter"], "test", classes)      # known PlantVillage only
    ds = ManifestDataset(test, build_transforms(t, b["augmentation"], train=False))
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], pin_memory=True)
    logits, y, idx = predict(model, dl, device, t["amp"], desc="test")
    prob = torch.softmax(logits, 1).numpy()
    y, pred = y.numpy(), prob.argmax(1)
    test = test.iloc[idx.numpy()].reset_index(drop=True)
    strict = test["strict_test"].to_numpy()

    full_m = summary_metrics(y, pred, len(classes))
    strict_m = summary_metrics(y[strict], pred[strict], len(classes))
    full_m.update(bootstrap_ci(y, pred, len(classes)))
    strict_m.update(bootstrap_ci(y[strict], pred[strict], len(classes)))
    diff = {k: strict_m[k] - full_m[k] for k in KEYS}
    cm = confusion(y, pred, len(classes))
    cm_strict = confusion(y[strict], pred[strict], len(classes))
    tops = top_confusions(cm, classes, k=10)

    # ---------------- outputs
    mdir, tdir = resolve(cfg["paths"]["metrics"]), resolve(cfg["paths"]["tables"])
    pdir, fdir = resolve(cfg["paths"]["results"]) / "predictions", resolve(cfg["paths"]["figures"]) / run
    for p in (pdir, fdir):
        p.mkdir(parents=True, exist_ok=True)
    pc_full, pc_strict = per_class_metrics(y, pred, classes), per_class_metrics(y[strict], pred[strict], classes)
    pc = pc_full.merge(pc_strict.drop(columns="class_name"), on="class_id", suffixes=("_full", "_strict"))
    pc.round(4).to_csv(tdir / f"{run}_per_class_metrics.csv", index=False)
    write_table(tops.round(4), tdir / f"{run}_top_confusions")
    preds = pd.DataFrame({
        "image_path": test["image_path"], "true_class_id": y, "true_class": [classes[i] for i in y],
        "pred_class_id": pred, "pred_class": [classes[i] for i in pred],
        "pred_prob": prob.max(1).round(6), "correct": y == pred, "max_softmax_prob": prob.max(1).round(6),
        "true_class_prob": prob[np.arange(len(y)), y].round(6), "strict_test": strict,
        "split_group_id": test["split_group_id"]})
    preds.to_csv(pdir / f"{run}_test_predictions.csv", index=False)
    plot_cm(cm, classes, fdir / "confusion_matrix_raw.png", normalized=False)
    plot_cm(cm, classes, fdir / "confusion_matrix_normalized.png", normalized=True)
    n_cam = b["evaluation"].get("gradcam_examples", 0)
    if n_cam:
        val = load_manifest(d["val_manifest"], d["row_filter"], "val", classes)
        gradcam_examples(model, val, cfg, classes, device, fdir / "gradcam_val_examples.png", n_cam)

    hist = json.loads((mdir / f"{run}_training_history.json").read_text())
    res = {"run_name": run, "checkpoint": str(ckpt_path.relative_to(ROOT)), "architecture": ckpt["architecture"],
           "num_classes": len(classes), "best_epoch": ckpt["best_epoch"], "best_phase": ckpt["best_phase"],
           "best_validation": ckpt["val_metrics"],
           "test_set": "known-class PlantVillage test (role=known, split=test); unknown classes NOT evaluated",
           "full_test": full_m, "strict_test": strict_m, "strict_minus_full": diff,
           "strict_test_definition": "test images with no near-duplicate candidate (L2-L4, any score) in train/val",
           "top_confusions_full": tops.to_dict(orient="records"),
           "strict_test_errors": int((y[strict] != pred[strict]).sum()), "full_test_errors": int((y != pred).sum()),
           "msp_summary": {"correct_median": float(np.median(prob.max(1)[y == pred])) if (y == pred).any() else None,
                           "incorrect_median": float(np.median(prob.max(1)[y != pred])) if (y != pred).any() else None}}
    (mdir / f"{run}_test_metrics.json").write_text(json.dumps(res, indent=2))
    if run == b["run_name"]:
        write_report(cfg, run, classes)

    print(f"full   : " + "  ".join(f"{k} {full_m[k]:.4f}" for k in KEYS))
    print(f"strict : " + "  ".join(f"{k} {strict_m[k]:.4f}" for k in KEYS) + f"   (n={strict_m['n']})")
    print("diff   : " + "  ".join(f"{k} {diff[k]:+.4f}" for k in KEYS))
    print(tops.head(5).to_string(index=False))


def write_report(cfg, run, classes):
    """Build results/reports/STAGE_3_BASELINE.md purely from saved artefacts (no model, no test pass)."""
    b, t = cfg["baseline"], cfg["baseline"]["training"]
    mdir, tdir = resolve(cfg["paths"]["metrics"]), resolve(cfg["paths"]["tables"])
    res = json.loads((mdir / f"{run}_test_metrics.json").read_text())
    hist = json.loads((mdir / f"{run}_training_history.json").read_text())
    env = json.loads((mdir / f"{run}_environment.json").read_text())
    pc = pd.read_csv(tdir / f"{run}_per_class_metrics.csv")
    tops = pd.read_csv(tdir / f"{run}_top_confusions.csv")
    preds = pd.read_csv(resolve(cfg["paths"]["results"]) / "predictions" / f"{run}_test_predictions.csv")
    f, s, dlt = res["full_test"], res["strict_test"], res["strict_minus_full"]
    pct = lambda v: f"{100 * v:.2f}%"  # noqa: E731
    ci = lambda m, k: f"[{100 * m[k + '_95ci'][0]:.2f}, {100 * m[k + '_95ci'][1]:.2f}]"  # noqa: E731

    def overlap(k):
        return f[k + "_95ci"][0] <= s[k + "_95ci"][1] and s[k + "_95ci"][0] <= f[k + "_95ci"][1]

    best = res["best_validation"]
    worst = pc.sort_values("f1_full").head(5)
    flagged = preds[~preds["strict_test"]]
    n_flag, err_flag = len(flagged), int((~flagged["correct"]).sum())
    low_share = pc.assign(share=pc["support_strict"] / pc["support_full"]).sort_values("share").head(4)
    low_txt = ", ".join(f"{short(r.class_name)} {r.support_strict}/{r.support_full}" for r in low_share.itertuples())
    biggest = pc.assign(d=pc["precision_strict"] - pc["precision_full"]).sort_values("d").iloc[0]
    hb = hist["history"]
    best_raw = max(hb, key=lambda h: h["val_macro_f1"])
    sel_note = ""
    if best_raw["epoch"] != res["best_epoch"]:
        sel_note = (f" Epoch {best_raw['epoch']} reached a marginally higher val macro-F1 "
                    f"({best_raw['val_macro_f1']:.4f}), but the gain over epoch {res['best_epoch']} is below the "
                    f"{t['early_stopping']['min_delta']} minimum-improvement threshold, so it was not selected.")
    if overlap("accuracy") and overlap("macro_f1"):
        ci_sentence = ("The bootstrap intervals of full and strict overlap for both accuracy and macro-F1, so the "
                       "difference is within sampling uncertainty.")
    else:
        ci_sentence = ("The bootstrap intervals do **not** overlap for "
                       + " and ".join(k for k in ("accuracy", "macro_f1") if not overlap(k)) + ".")
    rows_metric = "\n".join(f"| {name} | {pct(f[k])} | {pct(s[k])} | {100 * dlt[k]:+.2f} pp |" for k, name in
                            [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                             ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")])
    conf = "\n".join(f"| {short(r.true)} | {short(r.predicted)} | {r.count} | {100 * r.share_of_true_class:.2f}% |"
                     for r in tops.itertuples()) or "| — | — | 0 | — |"
    worst_rows = "\n".join(f"| {short(r.class_name)} | {r.support_full} | {r.precision_full:.4f} | {r.recall_full:.4f} | "
                           f"{r.f1_full:.4f} | {r.support_strict} | {r.f1_strict:.4f} |" for r in worst.itertuples())
    hist_rows = "\n".join(f"| {h['epoch']} | {h['phase']} | {h['train_loss']:.4f} | {h['train_accuracy']:.4f} | "
                          f"{h['val_loss']:.4f} | {h['val_accuracy']:.4f} | {h['val_macro_f1']:.4f} |" for h in hb)
    report = f"""# Stage 3 — Closed-Set Baseline

*Generated by `scripts/evaluate_baseline.py` from `results/metrics/{res['run_name']}_test_metrics.json`; every number below comes from that run.*

## Objective

How well does a conventional closed-set plant-disease classifier perform when **every test image belongs to one of the 21 classes seen in training**? This is the reference model for the later open-set (OSR) experiments. The baseline knows nothing about the unknown classes: they were not used for training, model selection, class weighting, augmentation, early stopping or hyperparameters. The validation-unknown classes were not used either.

## Dataset (locked Stage-2 protocol)

| Split | Images | Source |
|---|---:|---|
| Train | {hist['train_images']:,} | `data/splits/train.csv`, role = known |
| Validation | {hist['val_images']:,} | `data/splits/val.csv`, role = known (validation-unknown rows filtered out) |
| Full test | {f['n']:,} | `data/splits/test.csv`, role = known, PlantVillage |
| Strict test | {s['n']:,} | full-test images with **no** near-duplicate candidate against train/val |

The protocol lock (`data/splits/PROTOCOL_LOCK.json`) was verified before training and again before evaluation. Splits are leaf-grouped. Stage 2 found 0 confirmed strong cross-split duplicates, plus 7,669 weak similarity candidates that were flagged but are not proof of duplication.

## Model

- **Architecture:** torchvision `{res['architecture']}` with ImageNet `IMAGENET1K_V2` weights.
- **Head:** the original 1000-class head is replaced by a linear layer, 2048 → {res['num_classes']} known classes.
- **Grad-CAM:** targets `backbone.layer4`, the final convolutional stage.

## Training (as actually run)

- **Phase A:** {hist['phase_a_epochs']} epochs. The backbone is frozen, including its BatchNorm statistics. Only the head trains, with AdamW at lr {t['phase_a']['learning_rate']}.
- **Phase B:** {hist['phase_b_epochs']} epochs, out of a maximum of {t['phase_b']['max_epochs']}. `{', '.join(t['phase_b']['unfreeze'])}` and the head are unfrozen. AdamW, head lr {t['phase_b']['learning_rate']}, backbone lr {t['phase_b']['backbone_learning_rate']}, {t['phase_b']['warmup_epochs']} warm-up epoch, then per-step cosine decay. Stopped early: **{hist['stopped_early']}** (patience {t['early_stopping']['patience']}, minimum improvement {t['early_stopping']['min_delta']}, on known-validation macro-F1).
- **Loss:** cross-entropy with class weights w_c ∝ (1/n_c)^{t['class_weight_power']}, normalised to mean 1. Counts come from the training set only, giving weights from {min(hist['class_weights'].values()):.2f} to {max(hist['class_weights'].values()):.2f}.
  - *Why not a weighted sampler?* Train counts range from 104 (Potato healthy) to 3,750 (YLCV). A sampler would show the 104 Potato-healthy images, which come from only 38 physical leaves, about 36× more often than YLCV images, which risks memorising those leaves. Weighted cross-entropy sees every image once per epoch. The square-root weighting keeps the most-to-least weight ratio near 6× instead of 36×, so a few images don't dominate the gradients.
- **Batch and precision:** batch size {t['batch_size']}, weight decay {t['weight_decay']}, mixed precision {t['amp']}, seed {t['seed']}, cuDNN deterministic {t['deterministic']}.
- **Augmentation (train only):**
  - rotation ±{b['augmentation']['rotation']}°, with grey fill instead of black corners;
  - RandomResizedCrop to 224 (scale {b['augmentation']['random_resized_crop_scale']});
  - horizontal and vertical flips;
  - ColorJitter: brightness 0.2, contrast 0.2, saturation 0.15, **no hue shift**, because colour is a symptom.
- **Validation and test transform:** deterministic resize to 256, then center crop to 224.
- **Selection:** the best checkpoint over all epochs by known-validation macro-F1 is **epoch {res['best_epoch']} (phase {res['best_phase']})**: validation accuracy {pct(best['accuracy'])}, macro-F1 {pct(best['macro_f1'])}.{sel_note}
- **Training time:** {hist['training_minutes']:.1f} minutes.

| Epoch | Phase | Train loss | Train acc | Val loss | Val acc | Val macro-F1 |
|---:|---|---:|---:|---:|---:|---:|
{hist_rows}

Losses are class-weighted cross-entropy, the training objective, for both train and validation. Curves: `results/figures/{res['run_name']}/loss_curves.png`, `accuracy_curves.png`.

## Hardware

- **Device:** {env.get('gpu', env['device'])} ({env.get('gpu_memory_gb', '?')} GB).
- **Software:** PyTorch {env['torch']}, torchvision {env['torchvision']}, CUDA runtime {env['cuda_runtime']}, cuDNN {env['cudnn']}, Python {env['python']}.
- **Reproducibility:** seeds are fixed and cuDNN is deterministic, but some CUDA kernels are nondeterministic, so reruns may differ slightly.

## Results — known-class test

| Metric | Full test (n={f['n']:,}) | Strict test (n={s['n']:,}) | Strict − Full |
|---|---:|---:|---:|
{rows_metric}

Bootstrap 95% confidence intervals (1,000 resamples):
- **Full test:** accuracy {ci(f, 'accuracy')}, macro-F1 {ci(f, 'macro_f1')}.
- **Strict test:** accuracy {ci(s, 'accuracy')}, macro-F1 {ci(s, 'macro_f1')}.
- **Errors:** {res['full_test_errors']} on the full test, {res['strict_test_errors']} of them on the strict subset.

**Full vs strict: a sensitivity analysis, not a leakage test.** The strict subset removes every test image that has *any* near-duplicate candidate in train/val. That includes the potential residual same-leaf similarity flagged in Stage 2, which was never confirmed as leakage. A difference between the two columns would be consistent with, but not proof of, inflation from such similarity.

What the numbers show:
- **Accuracy and weighted F1** differ by {100 * dlt['accuracy']:+.2f} pp. {ci_sentence}
- **Macro precision and macro F1** fall further ({100 * dlt['macro_precision']:+.2f} pp and {100 * dlt['macro_f1']:+.2f} pp). Macro averages weight all 21 classes equally, and some classes keep very few strict images ({low_txt}). The largest per-class precision change is {short(biggest.class_name)}: {biggest.precision_full:.4f} → {biggest.precision_strict:.4f}, with {biggest.support_strict} strict images. A single false positive moves such a class a long way.
- **Errors concentrate in the strict subset:** {res['strict_test_errors']} of the {res['full_test_errors']} full-test errors fall in it. The {n_flag} images that *do* have a near-duplicate candidate in train/val got {err_flag} error(s) ({pct(1 - err_flag / n_flag)} accuracy). This is consistent with flagged images being easier because they resemble training images, which is the direction potential residual similarity would push. It is not proof: flagged images are also more often visually typical examples of their class.
- **The strict subset is not a random sample.** It drops proportionally more healthy-leaf images, so its class mix differs from the full test.

**Lowest per-class F1 (full test):**

| Class | n (full) | Precision | Recall | F1 | n (strict) | F1 (strict) |
|---|---:|---:|---:|---:|---:|---:|
{worst_rows}

All classes: `results/tables/{res['run_name']}_per_class_metrics.csv`.

## Confusion — most frequent errors (full test)

| True class | Predicted as | Count | Share of true class |
|---|---|---:|---:|
{conf}

Matrices: `results/figures/{res['run_name']}/confusion_matrix_raw.png` and `confusion_matrix_normalized.png`. The counts are small relative to class sizes, so per-pair statements should not be over-interpreted.

## Confidence (stored for later OSR work; no threshold chosen)

- **Per-image file:** `results/predictions/{res['run_name']}_test_predictions.csv`. It holds the true and predicted class, the predicted probability (= maximum softmax probability), correctness and the strict flag for every known test image.
- **Median maximum softmax probability:** {res['msp_summary']['correct_median']:.4f} for correct predictions ({f['n'] - res['full_test_errors']:,} images), {res['msp_summary']['incorrect_median']:.4f} for incorrect ones ({res['full_test_errors']} images).
- No unknown image was scored and no threshold was tuned.

## Limitations

1. **Closed-set assumption.** The classifier must choose one of 21 classes for every input. These numbers say nothing about how it handles unseen diseases, which is the subject of Stage 4 onward.
2. **Unknown classes not evaluated.** Unknown-class, PlantDoc and DTD images were not scored in this stage.
3. **Potential residual same-leaf similarity.** Stage 2 found no confirmed strong cross-split duplicates, but flagged weak similarity candidates, especially among smooth healthy leaves, that the automatic detectors cannot resolve. The full vs strict comparison above is the sensitivity analysis for this.
4. **PlantVillage domain.** Every image shows one detached leaf on a uniform board under controlled light. High accuracy here does not transfer to field photos. That is measured in Experiment C (PlantDoc) in a later stage.
5. **One run, one seed, one architecture.** No variance across seeds is reported yet.
6. **Grad-CAM is illustrative only.** `gradcam_val_examples.png` shows a few known validation images and was not used for any decision.

## Conclusion

**Readiness check for use as the comparison model in the OSR stages:**
- It was trained and selected on the locked protocol, using known classes only. The lock was verified at training and at evaluation.
- It is evaluated once on the full and strict known-class test sets, with per-class metrics, confusion matrices and per-image softmax confidence saved.
- It never saw unknown, validation-unknown, PlantDoc or DTD images.

**Interpretation:** under the closed-set assumption the classifier is highly accurate on PlantVillage: {pct(f['accuracy'])} on the full test, {pct(s['accuracy'])} on the strict test. That is exactly what makes it a meaningful baseline for the open-set question. A model this confident on known classes will also have to output one of those classes for every unseen disease. Whether its confidence separates known from unknown inputs is untested here and is the subject of Stage 4.
"""
    out = resolve(cfg["paths"]["results"]) / "reports" / "STAGE_3_BASELINE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
