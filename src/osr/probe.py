"""Diagnostic representation probe — NOT an OSR method.

Asks: does a LINEAR direction in a frozen embedding separate two image sets? It is trained
WITH the labels of the sets being compared (including unseen-disease labels), so it can only
bound what a representation encodes; it must never select thresholds or be reported as OSR.

Cross-validation is leaf-grouped (StratifiedGroupKFold): every photo of one physical leaf
(and every strong duplicate) stays in one fold, so the probe cannot score well by
recognising a leaf it was trained on.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "Diagnostic representation probe — not an OSR method"


def _oof(X, y, groups, C, seed, grouped=True):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped else StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups if grouped else None):
        if grouped:
            assert not set(groups[tr]) & set(groups[te]), "leaf group in both train and eval fold"
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def probe(Xa, Xb, ga, gb, C=0.1, repeats=5, n_boot=1000, seed=42) -> dict:
    """Xa = positive set, Xb = negative set; ga/gb = leaf-group ids (must not be shared between sets)."""
    X = np.r_[Xa, Xb]
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))].astype(int)
    g = np.r_[np.asarray(ga, dtype=object), np.asarray(gb, dtype=object)]
    rng = np.random.default_rng(seed)
    au, ac, f1 = [], [], []
    p0 = None
    for r in range(repeats):
        p = _oof(X, y, g, C, seed + r)
        p0 = p if p0 is None else p0
        au.append(roc_auc_score(y, p)); ac.append(accuracy_score(y, p >= 0.5)); f1.append(f1_score(y, p >= 0.5))
    # bootstrap over leaf GROUPS of the first repeat's out-of-fold predictions
    ug = np.unique(g)
    idx_by_g = {k: np.flatnonzero(g == k) for k in ug}
    boot = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_g[k] for k in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[pick])) == 2:
            boot.append(roc_auc_score(y[pick], p0[pick]))
    ungrouped = roc_auc_score(y, _oof(X, y, g, C, seed, grouped=False))
    return {"label": LABEL, "n_pos": int(len(Xa)), "n_neg": int(len(Xb)), "n_groups": int(len(ug)),
            "auroc_mean": float(np.mean(au)), "auroc_min_over_repeats": float(np.min(au)),
            "auroc_max_over_repeats": float(np.max(au)),
            "auroc_group_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "accuracy_mean": float(np.mean(ac)), "f1_mean": float(np.mean(f1)),
            "accuracy_range": [float(np.min(ac)), float(np.max(ac))], "f1_range": [float(np.min(f1)), float(np.max(f1))],
            "auroc_ungrouped_cv_for_reference": float(ungrouped), "repeats": repeats, "C": C,
            "cv": "StratifiedGroupKFold(5), groups = leaf/duplicate groups"}
