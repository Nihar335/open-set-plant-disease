"""Stage-4 MSP figures.

Axis convention: MSP is concentrated within 1e-6 of 1, so confidence is shown as
c = -log10(1 - MSP)  (c = 1 <-> MSP 0.9, 2 <-> 0.99, 3 <-> 0.999 ...), clipped at C_MAX.
Tick labels are written as MSP values.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils.config import PROJECT_ROOT
from src.visualization.style import INK2, MUTED, SERIES, apply_style, save

C_MAX = 12.0
TICKS = [0, 1, 2, 3, 4, 6, 8, 10, 12]
ROLE_COLOR = {"known": SERIES[0], "unknown_near": SERIES[1], "unknown_far_crop": SERIES[2],
              "unknown_far_texture": SERIES[3], "unknown_val": SERIES[4], "known_field": SERIES[0],
              "unknown_near_field": SERIES[1], "unknown_far_field": SERIES[2]}
ROLE_LABEL = {"known": "Known (PlantVillage test)", "unknown_near": "Unknown: new disease, known crop (B)",
              "unknown_far_crop": "Unknown: unseen crop (A1)", "unknown_far_texture": "Unknown: DTD texture (A2)",
              "known_field": "PlantDoc known-field", "unknown_near_field": "PlantDoc unknown, known crop",
              "unknown_far_field": "PlantDoc unknown crop"}


def conf(u):
    return np.clip(-np.log10(np.maximum(np.asarray(u, float), 1e-300)), 0, C_MAX)


def _msp_ticks(ax, axis="x"):
    labels = ["0" if t == 0 else ("0." + "9" * t if t <= 4 else f"1-1e-{t}") for t in TICKS]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(TICKS, labels)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)("MSP (axis: -log10(1 - MSP), clipped)")


def _short(c):
    return (c.replace("Corn_(maize)", "Corn").replace("Pepper,_bell", "Pepper").replace("Cherry_(including_sour)", "Cherry")
             .replace("___", " · ").replace("_", " ").replace("Haunglongbing (Citrus greening)", "HLB")
             .replace("Tomato Yellow Leaf Curl Virus", "YLCV").replace("Spider mites Two-spotted spider mite", "Spider mites")
             .replace("Cercospora leaf spot Gray leaf spot", "Gray leaf spot").replace("Leaf blight (Isariopsis Leaf Spot)", "Leaf blight"))


def _hist(ax, u, color, label, bins):
    ax.hist(conf(u), bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{label} (n={len(u):,})")


def validation_distribution(u_known, correct, u_valunk, rec, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, C_MAX, 49)
    _hist(ax, u_known[correct], SERIES[0], "known val, correctly classified", bins)
    _hist(ax, u_known[~correct], SERIES[1], "known val, misclassified", bins)
    _hist(ax, u_valunk, SERIES[4], "validation-unknown (reported only)", bins)
    tb = rec["strategies"]["B_validation_percentile"]["threshold_u"]
    tc = rec["strategies"]["C_error_aware"]["threshold_u"]
    for t, ls, txt in ((tb, "-", "B: 95% known accepted (selected)"), (tc, ":", "C: error-aware")):
        ax.axvline(conf(t), color="#52514e", ls=ls, lw=1.4)
        ax.text(conf(t) + 0.08, ax.get_ylim()[1] * (0.62 if ls == "-" else 0.50), txt, fontsize=7.5, color=INK2, va="top")
    for m in (0.5, 0.9, 0.99):
        ax.axvline(conf(1 - m), color=MUTED, lw=0.6, ls="--")
    _msp_ticks(ax)
    ax.set_ylabel("Density")
    ax.set_title("Validation MSP distribution (threshold chosen on known validation only)")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, out)


def reliability_diagram(calib, msp, out: Path):
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    bins = calib["bins"]
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a.bar([(x["lo"] + x["hi"]) / 2 for x in bins], [x["accuracy"] for x in bins], width=1 / calib["n_bins"],
          color=SERIES[0], edgecolor="white", alpha=0.85, label="accuracy")
    a.plot([x["confidence"] for x in bins], [x["accuracy"] for x in bins], "o", color=SERIES[1], ms=4, label="mean confidence")
    for x in bins:
        a.text((x["lo"] + x["hi"]) / 2, 0.02, str(x["n"]), rotation=90, fontsize=6, color="white" if x["accuracy"] > .2 else INK2,
               ha="center", va="bottom")
    a.set_xlabel("Confidence (MSP) bin"); a.set_ylabel("Accuracy")
    a.set_title(f"Reliability, known validation (ECE = {calib['ece']:.4f})")
    a.legend(loc="upper left", fontsize=7.5)
    b.hist(np.clip(msp, 0, 1), bins=40, color=SERIES[0], edgecolor="white")
    b.set_yscale("log"); b.set_xlabel("MSP"); b.set_ylabel("Images (log)")
    b.set_title("Confidence histogram (known validation)")
    save(fig, out)


def known_vs_unknown(df, t, out: Path):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    bins = np.linspace(0, C_MAX, 49)
    k = df[df["role"] == "known"]
    for ax, role, title in zip(axes, ["unknown_near", "unknown_far_crop", "unknown_far_texture"],
                               ["B: hard same-crop unknown", "A1: unseen crops", "A2: non-plant textures"]):
        _hist(ax, k["u"], ROLE_COLOR["known"], "known test", bins)
        _hist(ax, df.loc[df["role"] == role, "u"], ROLE_COLOR[role], "unknown test", bins)
        ax.axvline(conf(t), color="#52514e", lw=1.4)
        ax.text(conf(t) - 0.1, ax.get_ylim()[1] * 0.95, "reject  |  accept", ha="center", fontsize=7, color=INK2, va="top")
        _msp_ticks(ax); ax.set_title(title); ax.legend(loc="upper left", fontsize=7)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle("MSP: known vs unknown test images (vertical line = validation-selected threshold)",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)


def unknown_by_class(df, t, out: Path):
    apply_style()
    groups = [("known", "Known test (all 21 classes)", df[df["role"] == "known"])]
    for role in ("unknown_near", "unknown_far_crop"):
        for c, d in df[df["role"] == role].groupby("class_name"):
            groups.append((role, _short(c), d))
    groups.append(("unknown_far_texture", "DTD textures (47)", df[df["role"] == "unknown_far_texture"]))
    groups = [groups[0]] + sorted(groups[1:], key=lambda g: (g[0] != "unknown_near", -np.median(conf(g[2]["u"]))))
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(groups) + 1.4))
    data = [conf(g[2]["u"]) for g in groups][::-1]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, g in zip(bp["boxes"], groups[::-1]):
        patch.set_facecolor(ROLE_COLOR[g[0]]); patch.set_alpha(0.8); patch.set_edgecolor("white")
    for med in bp["medians"]:
        med.set_color("#0b0b0b")
    acc = [(g[2]["u"] <= t).mean() for g in groups][::-1]
    ax.set_yticks(range(1, len(groups) + 1), [f"{g[1]}  (n={len(g[2]):,})" for g in groups[::-1]], fontsize=7.5)
    for y, a in enumerate(acc, 1):
        ax.text(C_MAX + 0.15, y, f"{100 * a:5.1f}%", va="center", fontsize=7, color=INK2)
    ax.text(C_MAX + 0.15, len(groups) + 0.55, "accepted as known", fontsize=7, color=MUTED, va="bottom")
    ax.axvline(conf(t), color="#52514e", lw=1.4)
    _msp_ticks(ax); ax.set_xlim(0, C_MAX + 1.6); ax.grid(axis="y", visible=False)
    ax.set_title("MSP by class (box = IQR, whiskers = 5-95%); right: share accepted at the threshold", pad=18)
    save(fig, out)


def roc_pr(df, t, out_roc: Path, out_pr: Path, results):
    apply_style()
    k = df[df["role"] == "known"]["u"].to_numpy()
    sets = [("B_hard_same_crop", "unknown_near"), ("A1_unseen_crops", "unknown_far_crop"),
            ("A2_non_plant_textures", "unknown_far_texture")]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for name, role in sets:
        u = df.loc[df["role"] == role, "u"].to_numpy()
        y = np.r_[np.zeros(len(k)), np.ones(len(u))]
        fpr, tpr, _ = roc_curve(y, np.r_[k, u])
        ax.plot(fpr, tpr, lw=2, color=ROLE_COLOR[role], label=f"{name.split('_')[0]}: AUROC {results[name]['auroc']:.4f}")
        r = results[name]
        ax.plot(r["false_rejection_rate"], r["unknown_recall"], "o", color=ROLE_COLOR[role], ms=7, mec="white")
    ax.set_xlabel("False rejection rate (known rejected as unknown)")
    ax.set_ylabel("Unknown detection rate (unknown rejected)")
    ax.set_title("ROC — positive class = UNKNOWN, score = 1 - MSP\n(dots = validation-selected threshold)", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out_roc)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, role in sets:
        u = df.loc[df["role"] == role, "u"].to_numpy()
        y = np.r_[np.zeros(len(k)), np.ones(len(u))]
        p, r_, _ = precision_recall_curve(y, np.r_[k, u])
        ax.plot(r_, p, lw=2, color=ROLE_COLOR[role], label=f"{name.split('_')[0]}: AUPR-Out {results[name]['aupr_out']:.4f}")
        ax.axhline(y.mean(), color=ROLE_COLOR[role], lw=0.8, ls=":")
    ax.set_xlabel("Unknown recall"); ax.set_ylabel("Unknown precision")
    ax.set_title("Precision-recall — positive class = UNKNOWN\n(dotted = chance level = unknown share of the pool)", fontsize=9.5)
    ax.set_ylim(0, 1.02); ax.legend(loc="lower left", fontsize=8)
    save(fig, out_pr)


def threshold_analysis(val, df, t, out: Path):
    apply_style()
    grid = np.linspace(0, C_MAX, 241)
    tu = 10.0 ** (-grid)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    vk, vu = val[val["role"] == "known"], val[val["role"] == "unknown_val"]
    ok = vk["pred"] == vk["class_id"]
    a.plot(grid, [(vk["u"] <= x).mean() for x in tu], color=SERIES[0], lw=2, label="known acceptance")
    a.plot(grid, [(vk.loc[ok, "u"] > x).mean() for x in tu], color=SERIES[5], lw=2, label="correct known rejected")
    a.plot(grid, [(vk.loc[~ok, "u"] > x).mean() for x in tu], color=SERIES[1], lw=2, label="misclassified known rejected")
    a.plot(grid, [(vu["u"] > x).mean() for x in tu], color=SERIES[4], lw=2, ls="--", label="validation-unknown rejected (reported only)")
    a.set_title("VALIDATION (where the threshold is chosen)")
    k, u = df[df["role"] == "known"], df[df["role"] == "unknown_near"]
    kok = k["pred"] == k["label"]
    acc_k = np.array([(k["u"] <= x).mean() for x in tu])
    rec = np.array([(u["u"] > x).mean() for x in tu])
    rk = np.array([(k["u"] > x).sum() for x in tu])
    ru = np.array([(u["u"] > x).sum() for x in tu])
    prec = np.where(rk + ru > 0, ru / np.maximum(rk + ru, 1), np.nan)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), np.nan)
    b.plot(grid, acc_k, color=SERIES[0], lw=2, label="known acceptance")
    b.plot(grid, 1 - acc_k, color=SERIES[5], lw=2, label="false rejection rate")
    b.plot(grid, rec, color=SERIES[1], lw=2, label="unknown detection rate")
    b.plot(grid, 1 - rec, color=SERIES[3], lw=2, label="false known rate")
    b.plot(grid, f1, color=SERIES[6], lw=2, ls="--", label="unknown F1")
    b.set_title("TEST, Experiment B (shown for transparency; not used to choose)")
    for ax in (a, b):
        ax.axvline(conf(t), color="#52514e", lw=1.4)
        _msp_ticks(ax); ax.legend(loc="center left", fontsize=7.2); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    a.set_ylabel("Rate")
    fig.suptitle("Threshold analysis: accept as known iff MSP >= threshold (vertical line = selected)",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)


def examples_figure(ex: pd.DataFrame, out: Path):
    apply_style()
    n = len(ex)
    fig, axes = plt.subplots(1, n, figsize=(1.95 * n, 3.4))
    for ax, r in zip(np.atleast_1d(axes), ex.itertuples()):
        with Image.open(PROJECT_ROOT / r.image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((256, 256))
            ax.imshow(im)
        ax.axis("off")
        ok = r.msp_decision == "UNKNOWN"
        ax.set_title("\n".join(textwrap.wrap(r.rule, 24)), fontsize=6.3, color=MUTED)
        ax.text(0.5, -0.04, f"true: {_short(r.true_unknown_disease)}\npred: {_short(r.classifier_prediction)}\n"
                f"MSP {r.classifier_confidence_msp:.4f} -> {r.msp_decision}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.3,
                color="#006300" if ok else "#d03b3b")
    fig.suptitle("Hard unknowns (Experiment B): reproducible selection rule, not cherry-picked",
                 fontweight="semibold", fontsize=10)
    save(fig, out)


def plantdoc_figure(df, t, out: Path, results):
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, C_MAX, 49)
    for role in ("known", "known_field", "unknown_near_field", "unknown_far_field"):
        d = df[df["role"] == role]
        lab = ROLE_LABEL[role]
        a.hist(conf(d["u"]), bins=bins, density=True, histtype="step", lw=2,
               color=SERIES[6] if role == "known" else ROLE_COLOR[role], ls=":" if role == "known" else "-",
               label=f"{lab} (n={len(d):,})")
    a.axvline(conf(t), color="#52514e", lw=1.4)
    _msp_ticks(a); a.set_ylabel("Density"); a.legend(fontsize=7, loc="upper left")
    a.tick_params(axis="x", labelrotation=45, labelsize=7)
    a.set_title("Experiment C: MSP on PlantDoc (PlantVillage known test dotted)")
    k = df.loc[df["role"] == "known_field", "u"].to_numpy()
    b.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for name, roles, col in (("C_plantdoc_near_unknown", ["unknown_near_field"], SERIES[1]),
                             ("C_plantdoc_far_unknown", ["unknown_far_field"], SERIES[2])):
        u = df.loc[df["role"].isin(roles), "u"].to_numpy()
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(k)), np.ones(len(u))], np.r_[k, u])
        b.plot(fpr, tpr, lw=2, color=col, label=f"{name.replace('C_plantdoc_', '')}: AUROC {results[name]['auroc']:.4f}")
    b.set_xlabel("False rejection rate (PlantDoc known-field)"); b.set_ylabel("Unknown detection rate")
    b.set_title("Experiment C ROC (positive = unknown)"); b.legend(loc="lower right", fontsize=8)
    save(fig, out)
