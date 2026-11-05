"""Leakage-safe, reproducible splitting of the KNOWN PlantVillage classes.

Unit of assignment = "split group": the union (connected component) of
  * physical-leaf groups  (class_name || maintainer leaf_id), and
  * strong duplicate clusters (identical bytes / pixels / orientation-invariant hashes).
A split group is assigned to exactly one of train/val/test, so no physical leaf and no
strong duplicate can straddle two splits.

Assignment: per class, groups are visited largest-first (ties broken by a seeded shuffle)
and each goes to the split with the largest remaining deficit relative to its target
share of that class's images. Deterministic given the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.duplicates import UnionFind

SPLITS = ("train", "val", "test")


def split_groups(df: pd.DataFrame, extra_links: pd.DataFrame | None = None) -> pd.Series:
    """Connected components over leaf_group + dup_cluster (+ optional extra (i, j) links).
    `df` index must be the inventory row index used by `extra_links`."""
    pos = {ix: k for k, ix in enumerate(df.index)}
    uf = UnionFind(len(df))
    for key in ("leaf_group", "dup_cluster"):
        for members in df.groupby(key).indices.values():
            for m in members[1:]:
                uf.union(int(members[0]), int(m))
    if extra_links is not None:
        for i, j in zip(extra_links["i"], extra_links["j"]):
            if i in pos and j in pos:
                uf.union(pos[i], pos[j])
    labels = uf.labels()
    return pd.Series([f"sg{l:06d}" for l in labels], index=df.index, name="split_group_id")


def assign_splits(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Return a Series (index = df.index) with train/val/test per image."""
    rng = np.random.default_rng(seed)
    target = np.array([fractions[s] for s in SPLITS], dtype=float)
    target /= target.sum()
    # a split group spanning classes (rare) is assigned under its majority class
    g = df.groupby("split_group_id").agg(n=("class_name", "size"),
                                         cls=("class_name", lambda s: s.value_counts().index[0]))
    assignment = {}
    for cls in sorted(g["cls"].unique()):
        groups = g[g["cls"] == cls]
        order = groups.index.to_numpy()[rng.permutation(len(groups))]
        order = sorted(order, key=lambda k: -groups.at[k, "n"])          # stable: largest first
        total, filled = groups["n"].sum(), np.zeros(3)
        for k in order:
            deficit = target * total - filled
            s = int(np.argmax(deficit))
            assignment[k] = SPLITS[s]
            filled[s] += groups.at[k, "n"]
    return df["split_group_id"].map(assignment).rename("split")
