"""Stage-8 class-agnostic score fusion: orientation, known-validation ECDF normalisation, redundancy clusters, fusion rules.

Convention everywhere: larger (normalised) score = more likely unknown; accept as known iff score <= threshold.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

DIRECTIONS = {"higher_is_unknown": 1.0, "lower_is_unknown": -1.0}


def orient(raw, direction: str) -> np.ndarray:
    """Map a raw score to the 'higher = more unknown' convention (e.g. MSP -> -MSP)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("scores must be finite")
    return DIRECTIONS[direction] * x


class ECDFNormalizer:
    """Mid-rank empirical CDF of ORIENTED scores of known-validation images.

    t(s) = (#{v < s} + 0.5 * #{v == s}) / n  in [0, 1]; monotone non-decreasing, so a single score keeps
    its ranking (and AUROC). Scores beyond the reference range saturate at 0 / 1. No distributional
    assumption; fitted on known images only.
    """

    def fit(self, oriented_known):
        v = np.sort(np.asarray(oriented_known, dtype=np.float64))
        if v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all():
            raise ValueError("need a non-empty finite 1-D reference sample")
        self.ref_ = v
        return self

    def transform(self, oriented):
        s = np.asarray(oriented, dtype=np.float64)
        lo = np.searchsorted(self.ref_, s, side="left")
        hi = np.searchsorted(self.ref_, s, side="right")
        return (lo + 0.5 * (hi - lo)) / len(self.ref_)


def spearman_matrix(cols: dict) -> tuple[list, np.ndarray]:
    names = list(cols)
    R = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        R[i, j] = R[j, i] = spearmanr(cols[names[i]], cols[names[j]]).statistic
    return names, R


def complete_linkage_clusters(names, R, threshold: float) -> list[list[str]]:
    """Agglomerative complete linkage on |rho|: merge while every cross pair has |rho| >= threshold."""
    A = np.abs(R)
    clusters = [[i] for i in range(len(names))]
    while True:
        best, pair = -1.0, None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            link = min(A[i, j] for i in clusters[a] for j in clusters[b])
            if link >= threshold and link > best:
                best, pair = link, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return [sorted((names[i] for i in c), key=names.index) for c in clusters]


def representatives(names, R, clusters) -> dict:
    """Per cluster, the member with the lowest mean |rho| to signals OUTSIDE the cluster (ties -> first listed)."""
    A, idx, out = np.abs(R), {n: i for i, n in enumerate(names)}, {}
    for c in clusters:
        outside = [idx[n] for n in names if n not in c]
        if not outside or len(c) == 1:
            out[tuple(c)] = c[0]
            continue
        out[tuple(c)] = min(c, key=lambda n: (A[idx[n], outside].mean(), names.index(n)))
    return out


def cluster_weights(clusters) -> dict:
    w = {}
    for c in clusters:
        for n in c:
            w[n] = 1.0 / len(clusters) / len(c)
    return w


def fuse_mean(P: dict, names) -> np.ndarray:
    return np.mean([P[n] for n in names], axis=0)


def fuse_max(P: dict, names) -> np.ndarray:
    return np.max([P[n] for n in names], axis=0)


def fuse_weighted(P: dict, weights: dict) -> np.ndarray:
    tot = sum(weights.values())
    if not np.isclose(tot, 1.0):
        raise ValueError("weights must sum to 1")
    return np.sum([w * P[n] for n, w in weights.items()], axis=0)


def known_quantile_threshold(u_known, q: float) -> float:
    """Smallest observed score that accepts >= q of the known reference images (accept iff u <= t)."""
    return float(np.quantile(np.asarray(u_known, dtype=np.float64), q, method="higher"))


def simplex_grid(k: int, step: float):
    n = int(round(1 / step))
    for c in itertools.combinations(range(n + k - 1), k - 1):
        parts, prev = [], -1
        for x in c:
            parts.append(x - prev - 1)
            prev = x
        parts.append(n + k - 2 - prev)
        yield np.array(parts, dtype=float) / n
