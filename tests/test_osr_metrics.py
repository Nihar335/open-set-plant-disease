"""Synthetic sanity checks for OSR metric DIRECTION and definitions (no real data)."""
from __future__ import annotations

import numpy as np

from src.evaluation.calibration import reliability
from src.evaluation.osr_metrics import oscr, ranking_metrics, threshold_metrics
from src.osr.scores import msp_scores


def test_direction_perfect_and_inverted():
    known, unknown = np.array([0.01, 0.02, 0.03]), np.array([0.5, 0.6, 0.7])   # u: unknowns higher
    m = ranking_metrics(known, unknown)
    assert m["auroc"] == 1.0 and m["aupr_out"] == 1.0 and m["fpr_at_95tpr"] == 0.0
    inv = ranking_metrics(unknown, known)                                      # swapped -> inverted
    assert inv["auroc"] == 0.0 and inv["fpr_at_95tpr"] == 1.0


def test_chance_level():
    rng = np.random.default_rng(0)
    m = ranking_metrics(rng.random(20000), rng.random(20000))
    assert abs(m["auroc"] - 0.5) < 0.02 and abs(m["fpr_at_95tpr"] - 0.95) < 0.02


def test_oscr_bounds():
    k, u = np.array([0.0, 0.1, 0.2, 0.3]), np.array([0.8, 0.9])
    correct = np.array([1, 1, 1, 0], bool)
    assert abs(oscr(k, correct, u) - 0.75) < 1e-12          # perfect separation -> closed-set accuracy
    assert oscr(u[:1].repeat(4) * 0 + 0.95, correct, u) < 0.75   # knowns less "known" than unknowns


def test_threshold_metrics_counts():
    uk, uu = np.array([0.0, 0.0, 0.5, 0.0]), np.array([0.9, 0.0])
    m = threshold_metrics(uk, np.array([0, 1, 1, 2]), np.array([0, 1, 1, 0]), uu, t=0.4, n_classes=3)
    assert m["known_acceptance_rate"] == 0.75 and m["unknown_recall"] == 0.5
    assert m["unknown_precision"] == 0.5 and m["rejected_known"] == 1
    assert abs(m["open_set_accuracy"] - (2 + 1) / 6) < 1e-12   # known 0,1 accepted+correct; 1 unknown rejected


def test_msp_no_cancellation():
    s = msp_scores(np.array([[30.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert s["u"][0] > 0 and abs(s["u"][0] - 2 * np.exp(-30) / (1 + 2 * np.exp(-30))) < 1e-20
    assert abs(s["msp"][1] - np.e / (np.e + 2)) < 1e-12 and list(s["pred"]) == [0, 0]


def test_ece_perfect():
    r = reliability(np.array([1.0, 1.0, 0.5, 0.5]), np.array([1, 1, 1, 0], bool))
    assert abs(r["ece"]) < 1e-12
