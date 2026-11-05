"""Stage-7 figures: original vs contrastive representation under the same protocol."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.osr_plots import _short
from src.visualization.style import INK2, MUTED, SERIES, apply_style, save

NAMES = [("original::msp", "Original · MSP"), ("original::energy", "Original · Energy"),
         ("original::mahalanobis_raw", "Original · Mahalanobis (raw, Stage 6)"),
         ("original::prototype_cosine", "Original · Prototype"), ("original::mahalanobis_normalised", "Original · Mahalanobis (norm.)"),
         ("contrastive::prototype_cosine", "Contrastive · Prototype"), ("contrastive::mahalanobis_normalised", "Contrastive · Mahalanobis (norm.)"),
         ("contrastive::msp_linear_head_u", "Contrastive · MSP (linear head)"), ("contrastive::energy_linear_head", "Contrastive · Energy (linear head)")]


def comparison(res, prior, out: Path):
    apply_style()
    allr = {**prior, **res}
    exps = [("B_hard_same_crop", "B hard same-crop"), ("A1_unseen_crops", "A1 unseen crops"),
            ("A2_non_plant_textures", "A2 DTD"), ("C_plantdoc_all_unknown", "C PlantDoc")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.09
    x = np.arange(len(exps))
    for i, (k, lab) in enumerate(NAMES):
        hatch = "" if k.startswith("original") else "//"
        ax.bar(x + (i - 4) * w, [allr[k][e]["auroc"] for e, _ in exps], w, color=SERIES[i % 8], hatch=hatch,
               edgecolor="white", label=lab)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x, [l for _, l in exps]); ax.set_ylim(0.3, 1.02); ax.set_ylabel("AUROC (positive = unknown)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncols=3, loc="lower left")
    ax.set_title("Original (solid) vs contrastive (hatched) representation — same images, same protocol")
    save(fig, out)


def per_disease(per_cls, out: Path):
    apply_style()
    keys = [("original::energy", "Original · Energy", SERIES[1], ""), ("original::prototype_cosine", "Original · Prototype", SERIES[0], ""),
            ("contrastive::prototype_cosine", "Contrastive · Prototype", SERIES[0], "//"),
            ("original::mahalanobis_normalised", "Original · Mahalanobis (norm.)", SERIES[2], ""),
            ("contrastive::mahalanobis_normalised", "Contrastive · Mahalanobis (norm.)", SERIES[2], "//")]
    d = per_cls.sort_values("original::energy::auroc")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    h = 0.16
    for i, (k, lab, col, hatch) in enumerate(keys):
        ax.barh(y + (i - 2) * h, d[f"{k}::auroc"], h, color=col, hatch=hatch, edgecolor="white", label=lab)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y, [_short(c) for c in d["unknown_class"]]); ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("AUROC vs all known test images"); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("Hard unknowns per disease: does the contrastive representation help?")
    save(fig, out)


def geometry(diag, out: Path):
    apply_style()
    keys = [("intra_class_cosine_distance_mean", "Intra-class distance\n(known test → own prototype)"),
            ("inter_prototype_cosine_distance_mean", "Inter-prototype distance\n(mean)"),
            ("inter_prototype_cosine_distance_min", "Inter-prototype distance\n(closest pair)"),
            ("separation_ratio_inter_mean_over_intra_mean", "Separation ratio\n(inter / intra)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (k, lab) in zip(axes, keys):
        v = [diag["original"][k], diag["contrastive"][k]]
        ax.bar([0, 1], v, color=[SERIES[0], SERIES[1]], edgecolor="white")
        for i, vv in enumerate(v):
            ax.text(i, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["original", "contrastive"]); ax.set_title(lab, fontsize=9); ax.grid(axis="x", visible=False)
    fig.suptitle("Representation geometry on known test images (cosine, L2-normalised 2048-d features; prototypes from TRAIN)",
                 fontweight="semibold", fontsize=10)
    save(fig, out)


def deep_dive(dfs, near, T, out: Path):
    apply_style()
    cases = [("Grape___Esca_(Black_Measles)", "Grape___Black_rot"),
             ("Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.subplots_adjust(hspace=0.45)
    for r, (unk, look) in enumerate(cases):
        for cidx, rep in enumerate(("original", "contrastive")):
            ax = axes[r, cidx]
            d = dfs[rep]
            k = d[d["role"] == "known"]
            groups = [(k[k["class_name"] != look]["prototype_cosine"], "other known", SERIES[0]),
                      (k[k["class_name"] == look]["prototype_cosine"], f"look-alike: {_short(look)}", SERIES[5]),
                      (d[d["class_name"] == unk]["prototype_cosine"], f"UNKNOWN {_short(unk)}", SERIES[1])]
            hi = np.percentile(np.concatenate([g[0] for g in groups]), 99.5)
            bins = np.linspace(0, hi, 50)
            for v, lab, col in groups:
                ax.hist(np.clip(v, 0, hi), bins=bins, density=True, histtype="step", lw=2, color=col, label=f"{lab} (n={len(v):,})")
            ax.axvline(T[f"{rep}::prototype_cosine"], color="#52514e", lw=1.4)
            ax.set_xlabel("1 − max cosine to train prototypes (higher = more unknown)")
            ax.set_title(f"{rep.title()} representation — {_short(unk)}", fontsize=9.5)
            ax.legend(fontsize=6.5)
    fig.suptitle("Prototype distance: original vs contrastive (vertical line = known-validation threshold)",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)
