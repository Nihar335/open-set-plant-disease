"""Exact and near-duplicate detection (pHash Hamming distance, GPU if available)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hex_to_bits(hex_hashes: pd.Series) -> np.ndarray:
    ints = np.array([int(h, 16) for h in hex_hashes], dtype=">u8")
    return np.unpackbits(ints.view(np.uint8).reshape(-1, 8), axis=1)  # (N, 64) uint8


def near_duplicate_pairs(bits: np.ndarray, threshold: int, chunk: int = 2048) -> pd.DataFrame:
    """All pairs (i<j) with Hamming distance <= threshold.

    Hamming(a,b) = a·(1-b) + (1-a)·b, computed as two matmuls on ±bit matrices —
    exact for 0/1 values and ~100x faster on GPU than pairwise XOR in numpy.
    """
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32  # integers <= 64 are exact in fp16
    B = torch.from_numpy(bits).to(dev, dtype)
    nB = 1 - B
    rows, cols, dists = [], [], []
    n = B.shape[0]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = B[s:e] @ nB.T + nB[s:e] @ B.T                       # (chunk, N)
        idx = torch.nonzero(d <= threshold + 0.5)
        i, j = idx[:, 0] + s, idx[:, 1]
        keep = i < j
        i, j = i[keep], j[keep]
        rows.append(i.cpu()); cols.append(j.cpu()); dists.append(d[i - s, j].round().to(torch.int16).cpu())
    return pd.DataFrame({"i": torch.cat(rows).numpy(), "j": torch.cat(cols).numpy(),
                         "hamming": torch.cat(dists).numpy()})


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def pair_category(a, b) -> str:
    ds = tuple(sorted((a.dataset, b.dataset)))
    if ds == ("plantvillage", "plantvillage"):
        if a.class_name != b.class_name:
            return "PV: different class"
        if a.leaf_group == b.leaf_group:
            return "PV: same class, same leaf group"
        return "PV: same class, different leaf group"
    if ds == ("plantdoc", "plantdoc"):
        if a.class_name != b.class_name:
            return "PlantDoc: different class"
        if a.source_split != b.source_split:
            return "PlantDoc: same class, train<->test folder"
        return "PlantDoc: same class, same folder"
    if ds == ("plantdoc", "plantvillage"):
        return "PlantVillage <-> PlantDoc"
    return f"{ds[0]} <-> {ds[1]}" if ds[0] != ds[1] else f"{ds[0]} internal"


def describe_pairs(pairs: pd.DataFrame, inv: pd.DataFrame, method: str, strength: str) -> pd.DataFrame:
    cols = ["method", "strength", "score", "category", "i", "j", "path_i", "path_j",
            "class_i", "class_j", "dataset_i", "dataset_j", "leaf_group_i", "leaf_group_j"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    a = inv.iloc[pairs["i"].to_numpy()].reset_index(drop=True)
    b = inv.iloc[pairs["j"].to_numpy()].reset_index(drop=True)
    return pd.DataFrame({
        "method": method, "strength": strength, "score": pairs["score"].to_numpy(),
        "category": [pair_category(x, y) for x, y in zip(a.itertuples(), b.itertuples())],
        "i": pairs["i"].to_numpy(), "j": pairs["j"].to_numpy(),
        "path_i": a.rel_path.to_numpy(), "path_j": b.rel_path.to_numpy(),
        "class_i": a.class_name.to_numpy(), "class_j": b.class_name.to_numpy(),
        "dataset_i": a.dataset.to_numpy(), "dataset_j": b.dataset.to_numpy(),
        "leaf_group_i": a.leaf_group.to_numpy(), "leaf_group_j": b.leaf_group.to_numpy()})[cols]
