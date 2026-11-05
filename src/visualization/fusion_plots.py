"""Stage-8 figures: score correlation, benchmark, per-disease, matched acceptance, Esca/NLB, disagreement, examples, PlantDoc."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.config import resolve
from src.visualization.osr_plots import _short
from src.visualization.style import INK2, MUTED, SERIES, apply_style, save

SIGLAB = {"orig_msp": "MSP", "orig_energy": "Energy", "orig_mahalanobis": "Mahalanobis",
          "contrastive_prototype": "C-prototype", "contrastive_mahalanobis": "C-Mahalanobis", "contrastive_energy": "C-Energy*"}


def correlation(corr, out: Path):
    apply_style()
    keys = [("known_validation_design", "Known validation (used for design)"), ("known_test", "Known test (reported)"),
            ("hard_unknown_test", "Hard unknowns, test (reported)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (k, t) in zip(axes, keys):
        R, n = np.array(corr[k]["matrix"]), [SIGLAB[x] for x in corr[k]["names"]]
        im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(n)):
            for j in range(len(n)):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(R[i, j]) > 0.6 else INK2)
        ax.set_xticks(range(len(n)), n, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(len(n)), n, fontsize=8)
        ax.set_title(t, fontsize=10); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman ρ (oriented: higher = more unknown)")
    fig.suptitle("Score correlation (Spearman). C- = contrastive representation; * Stage-7 linear head", fontweight="semibold")
    save(fig, out)


def benchmark(res, methods, out: Path):
    apply_style()
    exps = [("B_hard_same_crop", "B hard same-crop"), ("A1_unseen_crops", "A1 unseen crops"),
            ("A2_non_plant_textures", "A2 DTD"), ("C_plantdoc_all_unknown", "C PlantDoc")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.8 / len(methods)
    x = np.arange(len(exps))
    for i, m in enumerate(methods):
        v = [res[m][e]["auroc"] for e, _ in exps]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, v, w, color=SERIES[i % 8], edgecolor="white", label=m,
               hatch="//" if m.startswith("Fusion") else "")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x, [l for _, l in exps]); ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUROC (positive = unknown)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=7, ncols=4, loc="lower left")
    ax.set_title("Final benchmark — same test images, thresholds from known validation (fusions hatched)")
    save(fig, out)


def per_disease(per_cls, methods, out: Path):
    apply_style()
    A = per_cls[[f"{m}::auroc" for m in methods]].to_numpy()
    D = per_cls[[f"{m}::detection" for m in methods]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.8), gridspec_kw={"wspace": 0.05})
    for ax, Z, t, fmt in ((axes[0], A, "AUROC vs all known test", "{:.3f}"), (axes[1], D, "Detection at 95% known-val acceptance", "{:.1%}")):
        ax.imshow(Z, cmap="viridis", vmin=0.0 if Z is D else 0.5, vmax=1.0, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=8, color="white" if Z[i, j] < 0.75 else "black")
        ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(per_cls)), [_short(c) for c in per_cls["unknown_class"]] if ax is axes[0] else [], fontsize=8)
        ax.set_title(t, fontsize=10); ax.grid(False)
    fig.suptitle("Hard unknowns per disease", fontweight="semibold")
    save(fig, out)


def matched(matched_, val_point, methods, out: Path):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    y = np.arange(len(methods))
    ax = axes[0]
    for i, m in enumerate(methods):
        v = val_point[m]
        lo, hi = v.get("ci95_detection", [v["unknown_detection"]] * 2)
        ax.errorbar(v["unknown_detection"], i, xerr=[[v["unknown_detection"] - lo], [hi - v["unknown_detection"]]], fmt="o", color=SERIES[i % 8], capsize=3)
        ax.text(1.0, i, f"known acc. {v['known_acceptance']:.1%}", va="center", ha="right", fontsize=8, color=INK2)
    ax.set_yticks(y, methods, fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("Hard-unknown detection")
    ax.set_title("Threshold from known VALIDATION (95%)", fontsize=10)
    ax = axes[1]
    for i, m in enumerate(methods):
        v = matched_[m]
        ax.errorbar(v["detection"], i, xerr=[[v["detection"] - v["ci95"][0]], [v["ci95"][1] - v["detection"]]], fmt="s", color=SERIES[i % 8], capsize=3)
    ax.set_yticks(y, methods, fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("Hard-unknown detection")
    ax.set_title("Exactly 95% known-TEST acceptance (analysis point)", fontsize=10)
    fig.suptitle("Matched known acceptance — 95% bootstrap intervals", fontweight="semibold")
    save(fig, out)


def deep(P, te, near_lookalike, T, out: Path):
    apply_style()
    sigs = list(P)
    cls, role = te["class_name"].to_numpy(), te["role"].to_numpy()
    fig, axes = plt.subplots(2, len(sigs) + 1, figsize=(19, 6.2), sharey="row")
    from src.osr.fusion import fuse_mean
    fz = fuse_mean(P, sigs)
    for r, (unk, look) in enumerate(near_lookalike.items()):
        for c, k in enumerate(sigs + ["fusion"]):
            ax = axes[r, c]
            s = fz if k == "fusion" else P[k]
            bins = np.linspace(0, 1, 26)
            for m, lab, col in (((role == "known") & (cls != look), "other known", SERIES[0]), ((role == "known") & (cls == look), "look-alike", SERIES[5]),
                                (cls == unk, "UNKNOWN", SERIES[1])):
                ax.hist(s[m], bins=bins, density=True, histtype="step", lw=1.8, color=col, label=lab)
            ax.axvline(T["fusion_mean"] if k == "fusion" else T[f"single::{k}"], color="#52514e", lw=1.2)
            ax.set_title(("Fusion A" if k == "fusion" else SIGLAB[k]), fontsize=9)
            if c == 0:
                ax.set_ylabel(_short(unk), fontsize=9)
            if r == 1:
                ax.set_xlabel("known-val percentile", fontsize=8)
        axes[r, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Normalised scores: unknown vs its look-alike (top: Esca vs Black rot; bottom: NLB vs Gray leaf spot). Line = 95% known-val threshold",
                 fontweight="semibold", fontsize=10)
    save(fig, out)


def disagreement(dis, out: Path):
    apply_style()
    d = dis.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(range(len(d)), d["all_hard_unknowns"], color=SERIES[0])
    for i, v in enumerate(d["all_hard_unknowns"]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(d)), [p.replace("Contrastive Energy*", "C-Energy*").replace("Fusion A (mean, primary)", "Fusion") for p in d["pattern"]], fontsize=8)
    ax.set_xlabel("Hard-unknown test images"); ax.grid(axis="y", visible=False)
    ax.set_title("Which methods reject each hard-unknown image (14 most common patterns; 'none' = accepted by all)")
    save(fig, out)


def examples(ex, out: Path):
    apply_style()
    n = len(ex)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.0))
    fig.subplots_adjust(wspace=0.15, hspace=0.45)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), ex.iterrows()):
        try:
            ax.imshow(Image.open(resolve(r["image_path"])).convert("RGB"))
        except OSError:
            ax.text(0.5, 0.5, "image unavailable", ha="center")
        rej = [m.split("::")[1].replace("Contrastive Energy*", "C-E").replace("Fusion A (mean, primary)", "Fus").replace("Mahalanobis", "Maha")
               for m in r.index if m.startswith("rejected::") and r[m]]
        ax.set_title(f"{r['category']}\n{_short(r['true_class'])} → pred {_short(r['stage3_pred'])}\nrejected by: {', '.join(rej) or 'none'}",
                     fontsize=7)
    fig.suptitle("Qualitative examples chosen by a fixed, seeded rule (not hand-picked)", fontweight="semibold")
    save(fig, out)


def domain_shift(res, methods, out: Path):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(methods))
    dom = [res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]["auroc"] for m in methods]
    dis_ = [res[m]["C_plantdoc_all_unknown"]["auroc"] for m in methods]
    axes[0].bar(x - 0.2, dom, 0.4, color=SERIES[1], label="domain novelty: PlantVillage known vs PlantDoc KNOWN diseases")
    axes[0].bar(x + 0.2, dis_, 0.4, color=SERIES[0], label="disease novelty within PlantDoc: known vs unknown diseases")
    axes[0].axhline(0.5, color=MUTED, ls="--", lw=1); axes[0].set_ylim(0.3, 1.02)
    axes[0].set_xticks(x, methods, rotation=20, ha="right", fontsize=8); axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", visible=False)
    pv = [res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]["plantvillage_known_acceptance"] for m in methods]
    pdk = [res[m]["C_domain_shift_known_pv_vs_known_plantdoc"]["plantdoc_known_acceptance"] for m in methods]
    axes[1].bar(x - 0.2, pv, 0.4, color=SERIES[2], label="PlantVillage known accepted")
    axes[1].bar(x + 0.2, pdk, 0.4, color=SERIES[3], label="PlantDoc known-disease images accepted")
    axes[1].set_ylim(0, 1.02); axes[1].set_xticks(x, methods, rotation=20, ha="right", fontsize=8); axes[1].legend(fontsize=7)
    axes[1].set_ylabel("Share accepted as known (same threshold)"); axes[1].grid(axis="x", visible=False)
    fig.suptitle("PlantDoc: image-domain novelty vs disease novelty", fontweight="semibold")
    save(fig, out)
