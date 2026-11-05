"""Stage-5 figures: energy distributions and MSP-vs-Energy comparisons (same images, same logits)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from src.visualization.osr_plots import C_MAX, ROLE_COLOR, _msp_ticks, _short, conf
from src.visualization.style import INK2, MUTED, SERIES, apply_style, save


def _ehist(ax, e, color, label, bins, ls="-"):
    ax.hist(np.clip(e, bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=2, color=color, ls=ls,
            label=f"{label} (n={len(e):,})")


def validation_distribution(e_known, correct, e_valunk, tau, T, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lo, hi = np.percentile(np.r_[e_known, e_valunk], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    _ehist(ax, e_known[correct], SERIES[0], "known val, correctly classified", bins)
    _ehist(ax, e_known[~correct], SERIES[1], "known val, misclassified", bins)
    _ehist(ax, e_valunk, SERIES[4], "validation-unknown (reported only)", bins)
    ax.axvline(tau, color="#52514e", lw=1.4)
    ax.text(tau, ax.get_ylim()[1] * 0.6, "  accept ← | → reject\n  (95% known-val accepted)", fontsize=7.5, color=INK2)
    ax.set_xlabel(f"Energy  E = -T·logsumexp(logits/T),  T = {T}   (lower = more known)")
    ax.set_ylabel("Density")
    ax.set_title("Validation energy distribution (threshold chosen on known validation only)")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, out)


def temperature_curve(per_t, msp_auroc, out: Path):
    apply_style()
    T = [float(k) for k in per_t]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(T, [per_t[k]["auroc_known_val_vs_validation_unknown"] for k in per_t], "o-", color=SERIES[0], lw=2,
            label="Energy: known val vs validation-unknown")
    ax.plot(T, [per_t[k]["auroc_misclassified_vs_correct_known_val"] for k in per_t], "s--", color=SERIES[1], lw=2,
            label="Energy: misclassified vs correct (known val)")
    ax.axhline(msp_auroc, color=SERIES[2], lw=1.5, ls=":", label="MSP: known val vs validation-unknown")
    ax.set_xscale("log"); ax.set_xticks(T, [str(t) for t in T])
    ax.set_xlabel("Temperature T"); ax.set_ylabel("AUROC (validation)")
    ax.set_title("Temperature sweep — VALIDATION data only")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, out)


def known_vs_unknown(df, tau, out: Path):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    k = df[df["role"] == "known"]["energy"]
    lo, hi = np.percentile(df.loc[df["role"].isin(["known", "unknown_near", "unknown_far_crop", "unknown_far_texture"]),
                                  "energy"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    for ax, role, title in zip(axes, ["unknown_near", "unknown_far_crop", "unknown_far_texture"],
                               ["B: hard same-crop unknown", "A1: unseen crops", "A2: non-plant textures"]):
        _ehist(ax, k, ROLE_COLOR["known"], "known test", bins)
        _ehist(ax, df.loc[df["role"] == role, "energy"], ROLE_COLOR[role], "unknown test", bins)
        ax.axvline(tau, color="#52514e", lw=1.4)
        ax.set_title(title); ax.set_xlabel("Energy (lower = more known)"); ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle("Energy: known vs unknown test images (vertical line = validation-selected threshold; left of it = accepted)",
                 fontweight="semibold", fontsize=10.5)
    save(fig, out)


def msp_vs_energy(df, msp_res, en_res, per_cls, boot, out: Path):
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.35]})
    k = df[df["role"] == "known"]
    u = df[df["role"] == "unknown_near"]
    y = np.r_[np.zeros(len(k)), np.ones(len(u))]
    for col, name, colr, res in (("u_msp", "MSP", SERIES[0], msp_res), ("energy", "Energy (T=1)", SERIES[1], en_res)):
        fpr, tpr, _ = roc_curve(y, np.r_[k[col], u[col]])
        bb = boot["B_hard_same_crop"]["auroc"]
        ci = bb["A_ci95"] if name == "MSP" else bb["B_ci95"]
        a.plot(fpr, tpr, lw=2, color=colr, label=f"{name}: AUROC {res['B_hard_same_crop']['auroc']:.4f} "
                                                 f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        r = res["B_hard_same_crop"]
        a.plot(r["false_rejection_rate"], r["unknown_recall"], "o", color=colr, ms=7, mec="white")
    a.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    a.set_xlabel("False rejection rate (known)"); a.set_ylabel("Unknown detection rate")
    a.set_title("Experiment B ROC, same images & logits\n(dots = validation thresholds; [ ] = bootstrap 95% CI)", fontsize=9.5)
    a.legend(loc="lower right", fontsize=8)
    d = per_cls[per_cls["experiment"].isin(["B_hard_same_crop", "A1_unseen_crops"])].copy()
    d["label"] = [("B · " if e.startswith("B") else "A1 · ") + _short(c) for e, c in zip(d["experiment"], d["unknown_class"])]
    d = d.sort_values(["experiment", "msp_auroc"], ascending=[False, True]).reset_index(drop=True)
    yy = np.arange(len(d))
    b.barh(yy - 0.2, d["msp_auroc"], height=0.38, color=SERIES[0], label="MSP", edgecolor="white")
    b.barh(yy + 0.2, d["energy_auroc"], height=0.38, color=SERIES[1], label="Energy (T=1)", edgecolor="white")
    for i, r in d.iterrows():
        b.text(max(r["msp_auroc"], r["energy_auroc"]) + 0.005, i, f"{r['energy_auroc'] - r['msp_auroc']:+.3f}",
               va="center", fontsize=7, color=INK2)
    b.axvline(0.5, color=MUTED, ls="--", lw=1)
    b.set_yticks(yy, d["label"], fontsize=7.5); b.set_xlim(0.4, 1.06)
    b.set_xlabel("AUROC vs all known test images (right: Energy − MSP)")
    b.set_title("Per unknown class: does Energy help where MSP fails?"); b.legend(loc="lower right", fontsize=8)
    b.grid(axis="y", visible=False)
    save(fig, out)


def deep_dive(df, near_cfg, t_msp, tau, deep, out: Path):
    apply_style()
    k = df[df["role"] == "known"]
    cases = list(deep)
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 4.4 * len(cases)))
    fig.subplots_adjust(hspace=0.62, wspace=0.28)
    for row, c in zip(np.atleast_2d(axes), cases):
        u = df[df["class_name"] == c]
        look = k[k["class_name"].isin(near_cfg[c]["resembles"])]
        other = k[~k["class_name"].isin(near_cfg[c]["resembles"])]
        cb = np.linspace(0, C_MAX, 49)
        for d, colr, lab, ls in ((other, SERIES[0], "other known", "-"), (look, SERIES[5], "look-alike known: "
                                 + ", ".join(_short(x) for x in near_cfg[c]["resembles"]), "-"), (u, SERIES[1], _short(c) + " (UNKNOWN)", "-")):
            row[0].hist(conf(d["u_msp"]), bins=cb, density=True, histtype="step", lw=2, color=colr, ls=ls, label=f"{lab} (n={len(d):,})")
        row[0].axvline(conf(t_msp), color="#52514e", lw=1.4); _msp_ticks(row[0])
        row[0].tick_params(axis="x", labelrotation=45, labelsize=7)
        row[0].set_title(f"MSP — AUROC vs all known {deep[c]['msp']['auroc_vs_all_known']:.4f}, "
                         f"detected {100 * deep[c]['msp']['detection']:.2f}%", fontsize=9)
        lo, hi = np.percentile(np.r_[k["energy"], u["energy"]], [0.5, 99.5])
        eb = np.linspace(lo, hi, 50)
        for d, colr, lab in ((other, SERIES[0], "other known"), (look, SERIES[5], "look-alike known"), (u, SERIES[1], "unknown")):
            _ehist(row[1], d["energy"].to_numpy(), colr, lab, eb)
        row[1].axvline(tau, color="#52514e", lw=1.4)
        row[1].set_xlabel("Energy (lower = more known)")
        row[1].set_title(f"Energy — AUROC vs all known {deep[c]['energy']['auroc_vs_all_known']:.4f}, "
                         f"detected {100 * deep[c]['energy']['detection']:.2f}%", fontsize=9)
        row[0].legend(fontsize=6.5, loc="upper left"); row[1].legend(fontsize=6.5, loc="upper right")
        for d, colr, s in ((k.sample(min(3000, len(k)), random_state=0), SERIES[0], 4), (look, SERIES[5], 6), (u, SERIES[1], 6)):
            row[2].scatter(conf(d["u_msp"]), d["energy"], s=s, alpha=0.35, color=colr, edgecolors="none")
        row[2].axvline(conf(t_msp), color="#52514e", lw=1, ls=":"); row[2].axhline(tau, color="#52514e", lw=1, ls=":")
        _msp_ticks(row[2]); row[2].set_ylabel("Energy"); row[2].tick_params(axis="x", labelrotation=45, labelsize=7)
        row[2].set_title("Same images: MSP vs Energy (dotted = thresholds)", fontsize=9)
        row[0].set_ylabel(_short(c), fontsize=10, fontweight="semibold")
    fig.suptitle("Deep dive: the two hardest Stage-4 unknowns under MSP and Energy (identical logits)",
                 fontweight="semibold", fontsize=11)
    save(fig, out)


def plantdoc(df, tau, en, msp, out: Path):
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4))
    roles = ("known", "known_field", "unknown_near_field", "unknown_far_field")
    lo, hi = np.percentile(df.loc[df["role"].isin(roles), "energy"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    for role in roles:
        d = df[df["role"] == role]["energy"].to_numpy()
        _ehist(a, d, SERIES[6] if role == "known" else ROLE_COLOR[role],
               {"known": "PlantVillage known test", "known_field": "PlantDoc known-field",
                "unknown_near_field": "PlantDoc unknown, known crop", "unknown_far_field": "PlantDoc unknown crop"}[role],
               bins, ls=":" if role == "known" else "-")
    a.axvline(tau, color="#52514e", lw=1.4); a.set_xlabel("Energy (lower = more known)"); a.set_ylabel("Density")
    a.set_title("Experiment C: energy on PlantDoc"); a.legend(fontsize=7)
    names = ["C_plantdoc_all_unknown", "C_plantdoc_near_unknown", "C_plantdoc_far_unknown"]
    x = np.arange(len(names))
    b.bar(x - 0.2, [msp[n]["auroc"] for n in names], 0.38, color=SERIES[0], label="MSP")
    b.bar(x + 0.2, [en[n]["auroc"] for n in names], 0.38, color=SERIES[1], label="Energy (T=1)")
    b.axhline(0.5, color=MUTED, ls="--", lw=1)
    b.set_xticks(x, ["all unknown", "unknown disease,\nknown crop", "unknown crop"]); b.set_ylim(0, 1)
    b.set_ylabel("AUROC (PlantDoc known-field vs unknown-field)"); b.set_title("Experiment C: MSP vs Energy")
    b.legend(fontsize=8)
    save(fig, out)
