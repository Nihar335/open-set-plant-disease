"""Stage-9 publication figures, drawn ONLY from saved results (no model inference, no test re-scoring)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from src.utils.config import resolve
from src.visualization.osr_plots import _short
from src.visualization.style import INK, INK2, MUTED, SERIES, apply_style, save

METHOD_COLORS = {"MSP": SERIES[0], "Energy": SERIES[1], "Mahalanobis": SERIES[2], "Contrastive prototype": SERIES[3],
                 "Contrastive Mahalanobis": SERIES[4], "Contrastive Energy*": SERIES[6], "Fusion A (mean, primary)": SERIES[7]}


def method_comparison(rows, out: Path):
    """rows: list of (label, auroc, fpr95, f1). Three panels, each on its full natural scale."""
    apply_style()
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    specs = [(1, "AUROC  (↑ better; 0.5 = chance)", (0.5, 1.0), "{:.3f}"), (2, "FPR @ 95% TPR  (↓ better)", (0, 1), "{:.1%}"),
             (3, "Unknown F1 at 95% known-val acceptance  (↑ better)", (0, 1), "{:.3f}")]
    for ax, (i, t, lim, fmt) in zip(axes, specs):
        v = [r[i] for r in rows]
        cols = [METHOD_COLORS.get(l.replace(" †", ""), MUTED) for l in labels]
        ax.barh(y, v, color=cols, edgecolor="white", height=0.7)
        for yy, vv in zip(y, v):
            ax.text(vv + (lim[1] - lim[0]) * 0.01, yy, fmt.format(vv), va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(lim[0], lim[1] * 1.12 if lim[1] == 1 and i != 1 else lim[1] * 1.08)
        if i == 1:
            ax.axvline(0.5, color=MUTED, ls="--", lw=1)
        ax.set_title(t, fontsize=10); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels, fontsize=9)
    fig.suptitle("OSR on hard same-crop unseen diseases (Experiment B: 5 unknown diseases, 6,164 unknown vs 4,091 known test images)",
                 fontweight="semibold", fontsize=11)
    fig.text(0.01, -0.02, "* Contrastive Energy uses the Stage-7 linear head; with the head held fixed the representation adds +0.023 AUROC "
             "[0.020, 0.026]. AUROC axis starts at 0.5 (chance), other axes at 0.", fontsize=8, color=INK2)
    save(fig, out)


def per_disease(pc, methods, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(pc))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        v = pc[f"{m}::auroc"].to_numpy()
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, v, w, color=METHOD_COLORS[m], edgecolor="white",
               label=m.replace("Fusion A (mean, primary)", "Fusion A (equal-weight mean)"))
    best = pc[[f"{m}::auroc" for m in methods]].to_numpy().argmax(1)
    for i, b in enumerate(best):
        xx = i + (b - (len(methods) - 1) / 2) * w
        ax.text(xx, pc.iloc[i][f"{methods[b]}::auroc"] + 0.008, "▼ best", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(len(pc) - 0.5, 0.505, "chance", color=MUTED, fontsize=8, ha="right")
    ax.set_xticks(x, [_short(c) for c in pc["unknown_class"]]); ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("AUROC vs all known test images"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncols=5, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.set_title("Per-disease AUROC — a different method wins on different diseases; Grape Esca stays hardest")
    save(fig, out)


def _box(ax, x, y, w, h, text, fc, ec=INK2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, a, b, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.2, linestyle=ls))


def failure_mode(esca_paths, look_paths, norm_df, thr, out: Path):
    """Esca story: real dataset images (no generation) + contrastive-Energy percentile distributions."""
    apply_style()
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.25, 1.6, 1.9], hspace=0.35, wspace=0.15)
    for r, (paths, lab) in enumerate(((look_paths, "KNOWN: Grape · Black rot (train class)"), (esca_paths, "UNSEEN: Grape · Esca (never trained)"))):
        for c in range(2):
            ax = fig.add_subplot(gs[r, c]); ax.axis("off")
            ax.imshow(Image.open(resolve(paths[c])).convert("RGB"))
            if c == 0:
                ax.set_title(lab, fontsize=9, loc="left", color=SERIES[5] if r == 0 else SERIES[1], fontweight="bold")
    ax = fig.add_subplot(gs[:, 3])
    for key, lab, col in (("other_known", "other known test images", SERIES[0]), ("lookalike", "Grape · Black rot (known)", SERIES[5]),
                          ("unknown", "Grape · Esca (UNSEEN)", SERIES[1])):
        ax.hist(norm_df[key]["contrastive_energy"], bins=np.linspace(0, 1, 26), density=True, histtype="step", lw=2, color=col, label=lab)
    ax.axvline(thr["contrastive_energy"], color=INK, lw=1.3)
    ax.text(thr["contrastive_energy"] - 0.01, ax.get_ylim()[1] * 0.92, "reject →\n(95% known\nacceptance)", ha="right", fontsize=7.5, color=INK2)
    ax.set_xlabel("contrastive-Energy novelty (percentile among known-validation)"); ax.set_ylabel("density")
    ax.set_title("Best detector (contrastive Energy)", fontsize=10); ax.legend(fontsize=7.5, loc="upper left")
    ax = fig.add_subplot(gs[:, 4]); ax.axis("off"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    _box(ax, 0.03, 0.80, 0.94, 0.14, "Known disease → compact class cluster\n→ low novelty score → ACCEPT (correct)", "#e8f2fc", fs=7.6)
    _box(ax, 0.03, 0.56, 0.94, 0.18, "Unseen same-crop disease → features land\nnext to the look-alike class (99.7% of Esca\nnearest to the Black rot centroid, Stage 6)", "#fdebe3", fs=7.6)
    _box(ax, 0.03, 0.34, 0.94, 0.16, "→ score higher than most known images, but\nmostly BELOW the 95%-acceptance threshold\n→ ACCEPTED as a known disease", "#fdebe3", fs=7.6)
    _box(ax, 0.03, 0.06, 0.94, 0.22, "A supervised probe separates Esca from Black rot\n(AUROC 0.999; DIAGNOSTIC ONLY — it uses Esca labels)\n→ the information exists, but no label-free\nscore isolates it", "#f1f0ea", fs=7.6)
    for a, b in (((0.5, 0.56), (0.5, 0.50)), ((0.5, 0.34), (0.5, 0.28))):
        _arrow(ax, a, b)
    fig.suptitle("Why unseen-disease detection fails: Grape Esca accepted as Grape Black rot", fontweight="semibold", fontsize=11.5)
    fig.text(0.01, 0.005, "Images: PlantVillage test images chosen by a fixed rule (first two by sorted path among images of the class in the test "
             "split / accepted by all five Stage-8 detectors). PlantVillage, Hughes & Salathé (2015); licence CC BY-SA 3.0 per the HF card (verify).",
             fontsize=7, color=INK2)
    save(fig, out)


def architecture(out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(15, 6.9)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 7.45)
    TR, INF, FU = "#e8f2fc", "#eaf7ef", "#fdebe3"
    ax.add_patch(FancyBboxPatch((0.2, 5.15), 14.6, 2.2, boxstyle="round,pad=0.02", fc="none", ec=SERIES[0], ls="--", lw=1.2))
    ax.text(0.35, 7.15, "TRAINING TIME (known classes only — 21 PlantVillage classes, train split)", fontsize=9, color=SERIES[0], fontweight="bold")
    _box(ax, 0.5, 5.5, 3.0, 1.2, "Stage 3: ResNet-50 (ImageNet init)\ncross-entropy, 21 known classes", TR)
    _box(ax, 4.0, 5.5, 3.4, 1.2, "Stage 7: ResNet-50 (ImageNet init)\nSupCon loss (T = 0.07), class-balanced\nprojection head used ONLY for the loss", TR)
    _box(ax, 7.9, 5.5, 3.3, 1.2, "Train-only fits on frozen features:\nclass means + tied Ledoit-Wolf covariance,\nprototypes, logistic-regression head", TR)
    _box(ax, 11.7, 5.5, 3.0, 1.2, "Known-VALIDATION only:\n95%-acceptance thresholds,\nscore percentiles (fusion)", TR)
    ax.text(0.35, 4.75, "INFERENCE TIME (OSR)", fontsize=9, color=SERIES[2], fontweight="bold")
    _box(ax, 0.3, 2.3, 1.6, 1.0, "Input leaf\nimage", INF, bold=True)
    _box(ax, 2.3, 2.3, 1.7, 1.0, "Preprocess\n(resize 256,\ncentre-crop 224)", INF)
    _box(ax, 4.4, 3.2, 2.2, 1.0, "Stage-3 ResNet-50\nbackbone", INF)
    _box(ax, 4.4, 1.2, 2.2, 1.0, "Stage-7 contrastive\nResNet-50 backbone", INF)
    _box(ax, 7.0, 3.2, 1.8, 1.0, "2048-d\nfeatures", INF)
    _box(ax, 7.0, 1.2, 1.8, 1.0, "2048-d features\n(L2-normalised)", INF)
    _box(ax, 9.2, 3.95, 2.0, 0.75, "Classification head\n→ logits", INF)
    _box(ax, 9.2, 2.85, 2.0, 0.75, "Mahalanobis\n(tied covariance)", INF)
    _box(ax, 9.2, 1.75, 2.0, 0.75, "Prototype cosine\ndistance; Mahalanobis", INF)
    _box(ax, 9.2, 0.65, 2.0, 0.75, "Linear head (LR)\n→ logits", INF)
    _box(ax, 11.6, 3.95, 1.6, 0.75, "MSP, Energy\n+ known disease", INF)
    _box(ax, 11.6, 0.65, 1.6, 0.75, "Contrastive\nEnergy*", INF)
    _box(ax, 13.6, 3.35, 1.3, 1.35, "Fusion A\n(mean of 6\nknown-val\npercentiles)", FU, fs=8)
    _box(ax, 11.45, 2.05, 1.95, 1.3, "OSR decision\n(each score, own τ)\nscore ≤ τ → KNOWN\nscore > τ → UNKNOWN", "#f1f0ea", fs=8, bold=True)
    for a, b in [((1.9, 2.8), (2.3, 2.8)), ((4.0, 2.95), (4.4, 3.6)), ((4.0, 2.65), (4.4, 1.7)), ((6.6, 3.7), (7.0, 3.7)), ((6.6, 1.7), (7.0, 1.7)),
                 ((8.8, 3.85), (9.2, 4.3)), ((8.8, 3.55), (9.2, 3.2)), ((8.8, 1.85), (9.2, 2.1)), ((8.8, 1.5), (9.2, 1.0)),
                 ((11.2, 4.3), (11.6, 4.3)), ((11.2, 1.0), (11.6, 1.0)), ((12.4, 3.95), (12.4, 3.35)), ((12.4, 1.4), (12.4, 2.05)),
                 ((11.2, 3.2), (11.45, 2.9)), ((11.2, 2.1), (11.45, 2.4))]:
        _arrow(ax, a, b)
    for a in [(13.2, 4.45), (13.2, 1.1)]:
        _arrow(ax, a, (13.6, 4.0), color=SERIES[7], ls="--")
    ax.text(13.55, 2.25, "+ the 4 distance\nscores (Mahalanobis\n×2, prototype) also\nenter Fusion A", fontsize=6.8, color=SERIES[7])
    _arrow(ax, (13.75, 3.35), (13.4, 2.95), color=SERIES[7])
    ax.text(0.3, 0.15, "Single-score detectors: MSP / Energy (Stage-3 logits), Mahalanobis (Stage-3 features), prototype / Mahalanobis (contrastive features), "
            "contrastive Energy* (linear head). Fusion A combines all six (dashed). The known-disease label always comes from the Stage-3 classifier "
            "(Stage-7 linear head for contrastive Energy).", fontsize=7.8, color=INK2)
    fig.suptitle("System architecture: training-time components (top) vs inference-time OSR components (bottom)", fontweight="semibold")
    save(fig, out)


def protocol(counts, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(15, 6.8)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6.8)
    K, U, X, C = "#e8f2fc", "#fdebe3", "#f1f0ea", "#eaf7ef"
    _box(ax, 0.2, 4.6, 2.2, 1.4, "PlantVillage\n(lab images,\n38 classes)", X, bold=True)
    _box(ax, 3.0, 5.3, 3.0, 1.1, f"21 KNOWN classes\n(6 crops)", K, bold=True)
    _box(ax, 3.0, 3.7, 3.0, 1.3, "UNKNOWN classes (whole classes\nexcluded from training):\n5 hard same-crop · 9 far-crop classes", U, bold=True)
    _box(ax, 3.0, 2.4, 3.0, 1.0, "3 validation-unknown classes\n(Tomato mosaic, Cherry ×2)", U)
    _box(ax, 6.8, 5.3, 2.6, 1.1, f"Train {counts['train']:,} / Val {counts['val']:,} /\nTest {counts['test']:,} (strict {counts['strict']:,})", K)
    _box(ax, 6.8, 3.85, 2.6, 1.0, "Held-out FINAL TEST only\n(Exp. B hard, Exp. A1 crops)", U)
    _box(ax, 6.8, 2.4, 2.6, 1.0, "Validation only\n(reported; Fusion D weights)", U)
    _box(ax, 0.2, 1.1, 2.2, 1.0, "PlantDoc\n(field photos)", X, bold=True)
    _box(ax, 0.2, 0.0, 2.2, 0.9, "DTD textures", X, bold=True)
    _box(ax, 6.8, 1.1, 2.6, 1.0, "Exp. C: cross-domain test\n(known vs unknown diseases)", C)
    _box(ax, 6.8, 0.0, 2.6, 0.9, "Exp. A2: non-plant OOD test", C)
    for a, b in [((2.4, 5.3), (3.0, 5.85)), ((2.4, 5.0), (3.0, 4.35)), ((2.4, 4.7), (3.0, 2.9)), ((6.0, 5.85), (6.8, 5.85)), ((6.0, 4.35), (6.8, 4.35)),
                 ((6.0, 2.9), (6.8, 2.9)), ((2.4, 1.6), (6.8, 1.6)), ((2.4, 0.45), (6.8, 0.45))]:
        _arrow(ax, a, b)
    _box(ax, 10.0, 3.7, 4.8, 2.7, "LEAKAGE CONTROLS (before any training)\n• split by physical leaf (maintainer leaf map)\n• + strong duplicates (md5 / pixel / D4 pHash+dHash)\n"
         "• + embedding near-duplicates (cos ≥ 0.98)\n• + SIFT/RANSAC-verified re-shots\n• PROTOCOL_LOCK.json hashes every manifest\n• strict test = no residual same-leaf similarity", "#fff8e1")
    _box(ax, 10.0, 0.3, 4.8, 3.0, "CALIBRATION & EVALUATION RULES\n• training: known classes only (asserted + unit-tested)\n• checkpoints: known-validation only\n"
         "• thresholds: 95% known-VALIDATION acceptance\n• test set scored once per stage (test_runs.log)\n• unknown test labels: evaluation only\n• paired bootstrap CIs (1,000 resamples)", "#fff8e1")
    fig.suptitle("Experimental protocol — what was trained on, what was held out, and how thresholds were set", fontweight="semibold")
    save(fig, out)
