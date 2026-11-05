"""Confidence calibration: reliability bins and Expected Calibration Error (Guo et al., 2017)."""
from __future__ import annotations

import numpy as np


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            acc, cf = float(correct[m].mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - cf)
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                         "accuracy": acc, "confidence": cf})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins,
            "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean())}
