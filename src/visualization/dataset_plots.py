"""Phase-2 dataset figures."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.utils.config import PROJECT_ROOT
from src.visualization.style import INK2, MUTED, SERIES, save


def _short(cls: str) -> str:
    return (cls.replace("Corn_(maize)", "Corn").replace("Cherry_(including_sour)", "Cherry")
               .replace("Pepper,_bell", "Pepper").replace("___", " · ").replace("_", " "))


def class_distribution(pv: pd.DataFrame, out: Path) -> None:
    counts = (pv.groupby(["crop", "class_name", "is_healthy"]).size().rename("n").reset_index()
                .sort_values(["crop", "n"], ascending=[False, True]))
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    y = np.arange(len(counts))
    colors = [SERIES[1] if h else SERIES[0] for h in counts["is_healthy"]]
    ax.barh(y, counts["n"], color=colors, height=0.72, edgecolor="white", linewidth=0.5)
    for yi, n in zip(y, counts["n"]):
        ax.text(n + 40, yi, f"{n:,}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [_short(c) for c in counts["class_name"]], fontsize=7.5)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"PlantVillage (color): {len(pv):,} images, {len(counts)} classes")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])],
              labels=["Disease", "Healthy"], loc="lower right")
    ax.set_xlim(0, counts["n"].max() * 1.12)
    save(fig, out)


def crop_distribution(df: pd.DataFrame, out: Path, title: str) -> None:
    t = df.groupby(["crop", "is_healthy"]).size().unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)
    t = t.loc[t.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(t) + 1.2))
    ax.barh(t.index, t[False], color=SERIES[0], height=0.7, label="Disease", edgecolor="white", linewidth=0.8)
    ax.barh(t.index, t[True], left=t[False], color=SERIES[1], height=0.7, label="Healthy",
            edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right")
    save(fig, out)


def plantdoc_distribution(pd_df: pd.DataFrame, out: Path) -> None:
    t = pd_df.groupby(["class_name", "source_split"]).size().unstack(fill_value=0).reindex(
        columns=["train", "test"], fill_value=0)
    t = t.loc[t.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(t.index, t["train"], color=SERIES[0], height=0.72, label="PlantDoc 'train' folder",
            edgecolor="white", linewidth=0.8)
    ax.barh(t.index, t["test"], left=t["train"], color=SERIES[1], height=0.72, label="PlantDoc 'test' folder",
            edgecolor="white", linewidth=0.8)
    for yi, n in enumerate(t.sum(1)):
        ax.text(n + 2, yi, str(n), va="center", fontsize=7, color=MUTED)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"PlantDoc: {len(pd_df):,} images, {len(t)} class folders")
    ax.legend(loc="lower right")
    save(fig, out)


def image_sizes(inv: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for k, name in enumerate(["plantvillage", "plantdoc", "dtd"]):
        d = inv[(inv["dataset"] == name) & inv["ok"]]
        if len(d):
            ax.scatter(d["width"], d["height"], s=9, alpha=0.45, color=SERIES[k], label=f"{name} (n={len(d):,})",
                       edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Width (px, log)"); ax.set_ylabel("Height (px, log)")
    ax.set_title("Image dimensions")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, out)


def leaf_group_sizes(pv: pd.DataFrame, out: Path) -> None:
    sizes = pv.groupby("leaf_group").size()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.arange(1, min(sizes.max(), 40) + 2) - 0.5
    ax.hist(sizes.clip(upper=40), bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Images per physical-leaf group (clipped at 40)")
    ax.set_ylabel("Groups (log)")
    ax.set_title(f"PlantVillage: {sizes.size:,} leaf groups for {len(pv):,} images")
    save(fig, out)


def background_stats(inv: pd.DataFrame, out: Path) -> None:
    """Border-pixel variation per dataset — a proxy for background uniformity (capture bias)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, name in enumerate(["plantvillage", "plantdoc", "dtd"]):
        d = inv[(inv["dataset"] == name) & inv["ok"]]
        if not len(d):
            continue
        for ax, col, lab in [(axes[0], "border_std", "Std. of 8-px border intensity"),
                             (axes[1], "border_mean", "Mean 8-px border intensity")]:
            ax.hist(d[col], bins=60, range=(0, 255 if col == "border_mean" else 90), density=True,
                    histtype="step", linewidth=2, color=SERIES[k], label=name)
            ax.set_xlabel(lab)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Background statistics (image border)", fontweight="semibold", fontsize=11)
    save(fig, out)


def sample_grid(df: pd.DataFrame, out: Path, title: str, per_class: int = 1, seed: int = 0, ncols: int = 8) -> None:
    classes = sorted(df["class_name"].unique())
    picks = [(c, r) for c in classes
             for r in df[(df["class_name"] == c) & df["ok"]].sample(
                 min(per_class, int(((df["class_name"] == c) & df["ok"]).sum())), random_state=seed).itertuples()]
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.95))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (c, r) in zip(np.ravel(axes), picks):
        with Image.open(PROJECT_ROOT / r.rel_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((256, 256))
            ax.imshow(im)
        ax.set_title(textwrap.fill(_short(c), 22), fontsize=5.8, color=INK2)
    fig.suptitle(title, fontweight="semibold", fontsize=11)
    save(fig, out)


def pair_examples(inv: pd.DataFrame, pairs: pd.DataFrame, out: Path, title: str, n: int = 6,
                  split_of: dict | None = None) -> None:
    """Pairs side by side. Tiles show our split when `split_of` (path -> split) is given,
    otherwise the dataset's own source split (e.g. the maintainer's train/test)."""
    pairs = pairs.head(n)
    if pairs.empty:
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.9 * len(pairs), 4.2), squeeze=False)
    for k, r in enumerate(pairs.itertuples()):
        for row, idx in enumerate((r.i, r.j)):
            rec = inv.iloc[idx]
            with Image.open(PROJECT_ROOT / rec.rel_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((256, 256))
                axes[row, k].imshow(im)
            axes[row, k].axis("off")
            where = f"split={split_of.get(rec.rel_path, '-')}" if split_of else f"{rec.dataset[:5]}/{rec.source_split}"
            axes[row, k].set_title(f"{where}\n{_short(rec.class_name)[:24]}",
                                    fontsize=5.8, color=INK2)
        axes[1, k].text(0.5, -0.08, f"score={r.hamming:g}", transform=axes[1, k].transAxes,
                        ha="center", va="top", fontsize=7, color=MUTED)
    fig.suptitle(title, fontweight="semibold", fontsize=10)
    save(fig, out)


# ----------------------------------------------------------------------------- Stage 2 (protocol) figures
ROLE_ORDER = ["known", "unknown_near", "unknown_val", "unknown_far_crop", "unknown_far_texture",
              "known_field", "unknown_near_field", "unknown_far_field"]
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}


def known_unknown_distribution(used: pd.DataFrame, out: Path) -> None:
    """Images per role, stacked by split (only images that enter the protocol)."""
    t = used.groupby(["role", "split"]).size().unstack(fill_value=0).reindex(
        index=[r for r in ROLE_ORDER if r in set(used["role"])], columns=["train", "val", "test"], fill_value=0)
    labels = {"known": "Known (PlantVillage)", "unknown_near": "Unknown: new disease, known crop (B)",
              "unknown_val": "Validation-unknown", "unknown_far_crop": "Unknown: unseen crop (A1)",
              "unknown_far_texture": "Unknown: non-plant texture, DTD (A2)", "known_field": "PlantDoc known-field (C)",
              "unknown_near_field": "PlantDoc unknown, known crop (C)", "unknown_far_field": "PlantDoc unknown crop (C)"}
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(t))
    for s in ("train", "val", "test"):
        ax.barh([labels[r] for r in t.index], t[s], left=left, color=SPLIT_COLORS[s], label=s, height=0.68,
                edgecolor="white", linewidth=0.8)
        left += t[s].to_numpy()
    for yi, n in enumerate(left):
        ax.text(n + left.max() * 0.01, yi, f"{int(n):,}", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.set_title("Open-set protocol: images per role and split")
    ax.legend(loc="lower right", ncols=3)
    save(fig, out)


def split_distribution(known: pd.DataFrame, out: Path) -> None:
    t = known.groupby(["class_name", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    t = t.loc[t.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    left = np.zeros(len(t))
    for s in ("train", "val", "test"):
        ax.barh([_short(c) for c in t.index], t[s], left=left, color=SPLIT_COLORS[s], label=s, height=0.72,
                edgecolor="white", linewidth=0.8)
        left += t[s].to_numpy()
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("Images")
    ax.grid(axis="y", visible=False)
    ax.set_title("Known classes: leaf-grouped train / val / test")
    ax.legend(loc="lower right")
    save(fig, out)


def manifest_grid(df: pd.DataFrame, out: Path, title: str, ncols: int = 7, per_class: int = 1, seed: int = 0) -> None:
    sample_grid(df.rename(columns={"image_path": "rel_path"}).assign(ok=True), out, title,
                per_class=per_class, seed=seed, ncols=ncols)


def domain_comparison(used: pd.DataFrame, out: Path, seed: int = 0) -> None:
    """Same disease, two domains: PlantVillage (lab) vs PlantDoc (field)."""
    pdoc = used[(used["dataset"] == "plantdoc")]
    classes = [c for c in pdoc["pv_equivalent"].value_counts().index][:8]
    fig, axes = plt.subplots(2, len(classes), figsize=(1.7 * len(classes), 4.3), squeeze=False)
    for k, c in enumerate(classes):
        for row, d in enumerate((used[(used["dataset"] == "plantvillage") & (used["class_name"] == c)],
                                 pdoc[pdoc["pv_equivalent"] == c])):
            r = d.sample(1, random_state=seed).iloc[0]
            with Image.open(PROJECT_ROOT / r.image_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((256, 256))
                axes[row, k].imshow(im)
            axes[row, k].axis("off")
        axes[0, k].set_title(_short(c)[:26], fontsize=6.2, color=INK2)
    axes[0, 0].text(-0.08, 0.5, "PlantVillage\n(lab)", transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=8)
    axes[1, 0].text(-0.08, 0.5, "PlantDoc\n(field/web)", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Domain shift: the same class in both datasets", fontweight="semibold", fontsize=10)
    save(fig, out)
