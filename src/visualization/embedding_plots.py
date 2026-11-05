"""Stage-6 feature-space figures (PCA fitted on TRAIN known only; t-SNE is visualisation only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.osr_plots import _short
from src.visualization.style import INK2, MUTED, SERIES, apply_style, save

UNK_COLORS = [SERIES[1], SERIES[3], SERIES[4], SERIES[6], SERIES[7]]


def pca_plot(df, P_te, P_va, va, pca, out: Path, near: dict):
    apply_style()
    ev = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    a = axes[0]
    k = df["role"].to_numpy() == "known"
    a.scatter(P_te[k, 0], P_te[k, 1], s=3, color="#c3c2b7", alpha=0.5, label="known test (21 classes)")
    kv = (va["role"] == "known").to_numpy()
    a.scatter(P_va[kv, 0], P_va[kv, 1], s=3, color="#898781", alpha=0.25, label="known validation")
    for col, c in zip(UNK_COLORS, near):
        m = (df["class_name"] == c).to_numpy()
        a.scatter(P_te[m, 0], P_te[m, 1], s=4, color=col, alpha=0.6, label=f"UNKNOWN {_short(c)}")
    a.set_title("All known vs hard unknowns"); a.legend(fontsize=6.5, markerscale=3, loc="best")
    for ax, crop, unk in ((axes[1], "Grape___", "Grape___Esca_(Black_Measles)"),
                          (axes[2], "Corn_(maize)___", "Corn_(maize)___Northern_Leaf_Blight")):
        same = sorted(c for c in df.loc[df["role"] == "known", "class_name"].unique() if c.startswith(crop))
        for i, c in enumerate(same):
            m = ((df["class_name"] == c) & (df["role"] == "known")).to_numpy()
            ax.scatter(P_te[m, 0], P_te[m, 1], s=7, color=[SERIES[0], SERIES[2], SERIES[5], SERIES[6]][i], alpha=0.6,
                       label=f"known {_short(c)} (test)")
        m = (df["class_name"] == unk).to_numpy()
        ax.scatter(P_te[m, 0], P_te[m, 1], s=7, color=SERIES[1], alpha=0.5, label=f"UNKNOWN {_short(unk)}")
        ax.set_title(f"{crop.split('_')[0]}: same-crop known classes vs the unknown"); ax.legend(fontsize=6.5, markerscale=2)
    for ax in axes:
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of train variance)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    fig.suptitle("PCA of 2048-d penultimate features — PCA fitted on KNOWN TRAINING embeddings only; test projected",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)


def tsne_plot(sub, Y, out: Path, near: dict, ts: dict):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    k = (sub["role"] == "known").to_numpy()
    ax.scatter(Y[k, 0], Y[k, 1], s=5, color="#c3c2b7", alpha=0.7)
    for c in sub.loc[k, "class_name"].unique():
        m = ((sub["class_name"] == c) & k).to_numpy()
        ax.text(np.median(Y[m, 0]), np.median(Y[m, 1]), _short(c).replace(" · ", "\n"), fontsize=6, color=INK2,
                ha="center", va="center")
    for col, c in zip(UNK_COLORS, near):
        m = (sub["class_name"] == c).to_numpy()
        ax.scatter(Y[m, 0], Y[m, 1], s=9, color=col, alpha=0.8, label=f"UNKNOWN {_short(c)}")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title(f"t-SNE (visualisation only): {len(sub):,} test images, ≤{ts['per_class_sample']}/class; input = PCA-50 "
                 f"(fit on train); perplexity {ts['perplexity']}, {ts['max_iter']} iterations, seed {ts['seed']}\n"
                 "grey = known classes (labels at class medians); colours = hard unknowns", fontsize=9)
    save(fig, out)


def centroid_distance_plot(df, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    groups = [("known", "Known test", SERIES[0]), ("unknown_near", "B: hard same-crop unknown", SERIES[1]),
              ("unknown_far_crop", "A1: unseen crops", SERIES[2]), ("unknown_far_texture", "A2: DTD", SERIES[3]),
              ("known_field", "PlantDoc known-field", SERIES[6])]
    hi = np.percentile(df["nc_dist"], 99.5)
    bins = np.linspace(0, hi, 70)
    for role, lab, col in groups:
        d = df.loc[df["role"] == role, "nc_dist"]
        ax.hist(np.clip(d, 0, hi), bins=bins, density=True, histtype="step", lw=2, color=col, label=f"{lab} (n={len(d):,})")
    ax.set_xlabel("Euclidean distance to nearest known-class centroid (centroids from TRAINING embeddings)")
    ax.set_ylabel("Density"); ax.legend(fontsize=7.5)
    ax.set_title("Nearest-centroid distance in feature space")
    save(fig, out)


def correlation_plot(corr: dict, out: Path):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, key in zip(axes, ("pearson", "spearman")):
        M = np.array(corr[key])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(M)), corr["names"], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(M)), corr["names"], fontsize=8); ax.grid(False)
        for i in range(len(M)):
            for j in range(len(M)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 else "#0b0b0b")
        ax.set_title(f"{key.title()} correlation")
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("OSR score correlations — Experiment B pool (known test + hard unknowns); all oriented 'higher = more unknown'",
                 fontweight="semibold", fontsize=10)
    save(fig, out)


def method_comparison(per_cls, msp, en, maha, boot, out: Path):
    apply_style()
    d = per_cls[per_cls["experiment"] == "B_hard_same_crop"].sort_values("msp_auroc")
    labels = [_short(c) for c in d["unknown_class"]] + ["B overall", "A1 unseen crops", "A2 DTD", "C PlantDoc"]
    keys = ["B_hard_same_crop", "A1_unseen_crops", "A2_non_plant_textures", "C_plantdoc_all_unknown"]
    vals = {"MSP": list(d["msp_auroc"]) + [msp[k]["auroc"] for k in keys],
            "Energy": list(d["energy_auroc"]) + [en[k]["auroc"] for k in keys],
            "Mahalanobis": list(d["mahalanobis_auroc"]) + [maha[k]["auroc"] for k in keys]}
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, (name, col) in enumerate(zip(vals, (SERIES[0], SERIES[1], SERIES[2]))):
        ax.bar(x + (i - 1) * 0.27, vals[name], 0.26, color=col, label=name, edgecolor="white")
    for i, v in enumerate(vals["Mahalanobis"]):
        ax.text(x[i] + 0.27, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5, color=INK2, rotation=90)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1); ax.axvline(len(d) - 0.5, color=MUTED, lw=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8); ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC (positive = unknown)"); ax.grid(axis="x", visible=False)
    ax.set_title("MSP vs Energy vs Mahalanobis — same frozen model; left: hard unknowns per disease, right: experiments")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out)


def esca_feature_plot(df, P_te, near, out: Path, esca_t, nlb_t):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, t, title in ((axes[0], esca_t, "Grape Esca (UNKNOWN)"), (axes[1], nlb_t, "Corn NLB (UNKNOWN)")):
        t = t.sort_values("centroid_euclidean_unknown_to_known")
        y = np.arange(len(t))
        ax.barh(y + 0.2, t["mean_dist_unknown_images_to_known_centroid"], 0.38, color=SERIES[1],
                label="unknown images → this class's centroid")
        ax.barh(y - 0.2, t["mean_dist_known_test_images_to_own_centroid"], 0.38, color=SERIES[0],
                label="this class's own test images → its centroid")
        for i, r in enumerate(t.itertuples()):
            ax.text(max(r.mean_dist_unknown_images_to_known_centroid, r.mean_dist_known_test_images_to_own_centroid) + 0.2,
                    i, f"probe AUROC {r.linear_probe_auroc_unknown_vs_known_train_DIAGNOSTIC:.3f}", va="center",
                    fontsize=7, color=INK2)
        ax.set_yticks(y, [_short(c) for c in t["known_class"]], fontsize=8)
        ax.set_xlabel("Mean Euclidean distance to centroid (training means)")
        ax.set_title(f"{title}: distance to same-crop known classes", fontsize=9.5)
        ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="y", visible=False)
    fig.suptitle("Is the unknown embedded inside its look-alike? (probe = 5-fold CV linear separability, DIAGNOSTIC only)",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)
