"""Paired, stratified bootstrap for comparing two OSR scores on the SAME images.

Each resample draws the known set and the unknown set independently with replacement
(stratified, so the known:unknown ratio is fixed) and evaluates BOTH scores on the same draw.
Thresholds are held fixed at their validation-selected values (not re-estimated), so the
intervals describe test-sampling uncertainty of the deployed decision rules.
Reported: 2.5/97.5 percentiles for each method and for the difference (B - A), plus the share
of resamples where B > A. Overlapping marginal intervals are NOT a significance test; the
difference interval is the relevant quantity.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _fpr95(uk, uu):
    y = np.r_[np.ones(len(uk)), np.zeros(len(uu))]                 # known positive, score -u
    fpr, tpr, _ = roc_curve(y, -np.r_[uk, uu])
    return float(fpr[min(int(np.searchsorted(tpr, 0.95)), len(fpr) - 1)])


def _point(uk, uu, tk):
    y = np.r_[np.zeros(len(uk)), np.ones(len(uu))]
    rej_u, rej_k = (uu > tk).sum(), (uk > tk).sum()
    rec = rej_u / len(uu)
    prec = rej_u / (rej_u + rej_k) if rej_u + rej_k else 0.0
    return {"auroc": float(roc_auc_score(y, np.r_[uk, uu])), "fpr_at_95tpr": _fpr95(uk, uu),
            "unknown_recall": float(rec), "unknown_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "known_acceptance_rate": float((uk <= tk).mean())}


def paired_bootstrap(uk_a, uu_a, t_a, uk_b, uu_b, t_b, n_boot=1000, seed=42) -> dict:
    """A and B are two scores for the same known (uk_*) and unknown (uu_*) images, aligned by index."""
    rng = np.random.default_rng(seed)
    nk, nu = len(uk_a), len(uu_a)
    assert len(uk_b) == nk and len(uu_b) == nu
    keys = list(_point(uk_a, uu_a, t_a))
    A, B = {k: [] for k in keys}, {k: [] for k in keys}
    for _ in range(n_boot):
        ik, iu = rng.integers(0, nk, nk), rng.integers(0, nu, nu)
        pa, pb = _point(uk_a[ik], uu_a[iu], t_a), _point(uk_b[ik], uu_b[iu], t_b)
        for k in keys:
            A[k].append(pa[k]); B[k].append(pb[k])
    out = {}
    pa, pb = _point(uk_a, uu_a, t_a), _point(uk_b, uu_b, t_b)
    for k in keys:
        a, b = np.array(A[k]), np.array(B[k])
        d = b - a
        out[k] = {"A": pa[k], "B": pb[k], "A_ci95": [float(np.quantile(a, .025)), float(np.quantile(a, .975))],
                  "B_ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                  "diff_B_minus_A": pb[k] - pa[k],
                  "diff_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                  "share_B_gt_A": float((d > 0).mean())}
    out["_meta"] = {"n_boot": n_boot, "seed": seed, "n_known": nk, "n_unknown": nu,
                    "method": "paired stratified bootstrap; thresholds fixed at validation values"}
    return out
