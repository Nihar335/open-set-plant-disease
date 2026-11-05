"""Stage-6 Mahalanobis: stability (singular / near-singular covariance, extremes), direction, thresholds, schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.osr_metrics import ranking_metrics, threshold_metrics
from src.osr.mahalanobis import TiedMahalanobis
from src.utils.config import PROJECT_ROOT


def _data(n=300, d=16, c=3, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 10, (c, d))
    y = np.repeat(np.arange(c), n // c)
    return means[y] + rng.normal(0, scale, (len(y), d)), y, means


def test_direction_known_closer_than_unknown():
    X, y, means = _data()
    m = TiedMahalanobis().fit(X, y, 3)
    known, _ = m.score(means + np.random.default_rng(1).normal(0, 1, means.shape))
    far, _ = m.score(np.full((3, 16), 100.0))
    assert known.max() < far.min()
    assert ranking_metrics(known, far)["auroc"] == 1.0


def test_matches_explicit_inverse_when_well_conditioned():
    X, y, _ = _data(n=3000, d=8)
    m = TiedMahalanobis().fit(X, y, 3)
    S = np.linalg.inv(m.W_ @ m.W_.T)                  # recovered (floored) covariance
    Q = np.random.default_rng(2).normal(0, 5, (20, 8))
    ref = np.array([[(q - mu) @ np.linalg.inv(S) @ (q - mu) for mu in m.means_] for q in Q])
    assert np.allclose(m.distances(Q), ref, rtol=1e-8)


def test_singular_covariance_n_less_than_d():
    X, y, _ = _data(n=30, d=200)                       # 30 samples, 200 dims -> sample covariance singular
    m = TiedMahalanobis().fit(X, y, 3)
    assert 0 < m.shrinkage_ <= 1 and np.isfinite(m.condition_number_)
    s, _ = m.score(np.random.default_rng(3).normal(0, 1, (10, 200)))
    assert np.isfinite(s).all() and (s >= 0).all()


def test_near_zero_eigenvalue_is_floored():
    X, y, _ = _data(n=600, d=10)
    X[:, 9] = X[:, 0] * 2.0                             # exactly collinear feature -> near-zero eigenvalue
    m = TiedMahalanobis(eig_floor_rel=1e-6).fit(X, y, 3)
    assert m.condition_number_ <= 1e6 * (1 + 1e-9)
    assert np.isfinite(m.score(X)[0]).all()


def test_extreme_embeddings_finite():
    X, y, _ = _data()
    m = TiedMahalanobis().fit(X, y, 3)
    s, _ = m.score(np.array([[1e8] * 16, [-1e8] * 16, [0.0] * 16]))
    assert np.isfinite(s).all() and s[0] > s[2]


def test_nonfinite_inputs_raise():
    X, y, _ = _data()
    m = TiedMahalanobis().fit(X, y, 3)
    with pytest.raises(ValueError):
        m.score(np.array([[np.nan] * 16]))
    with pytest.raises(ValueError):
        m.score(np.array([[np.inf] * 16]))
    bad = X.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        TiedMahalanobis().fit(bad, y, 3)
    with pytest.raises(ValueError):
        TiedMahalanobis().fit(X, np.zeros_like(y), 3)   # a class with no samples


def test_threshold_direction_and_labels():
    d_known, d_unknown, tau = np.array([5.0, 10.0, 50.0]), np.array([20.0, 300.0]), 30.0   # accept iff d <= tau
    m = threshold_metrics(d_known, np.array([0, 1, 2]), np.array([0, 1, 2]), d_unknown, tau, 21)
    assert m["known_acceptance_rate"] == 2 / 3 and m["unknown_recall"] == 0.5


PRED = PROJECT_ROOT / "results" / "predictions" / "osr_mahalanobis_predictions.csv"


@pytest.mark.skipif(not PRED.exists(), reason="Stage-6 evaluation not run yet")
def test_prediction_schema():
    p = pd.read_csv(PRED, low_memory=False)
    need = ["image_path", "dataset", "true_class", "true_known_unknown", "predicted_class", "msp", "energy",
            "mahalanobis", "mahalanobis_threshold", "predicted_known_unknown", "correct_osr_decision"]
    assert set(need) <= set(p.columns) and p["image_path"].is_unique
    d, t = p["mahalanobis"].astype(float), p["mahalanobis_threshold"].astype(float)
    assert np.isfinite(d).all() and (d >= 0).all() and t.nunique() == 1
    assert ((d <= t) == (p["predicted_known_unknown"] == "known")).all()
    assert ((p["predicted_known_unknown"] == p["true_known_unknown"]) == p["correct_osr_decision"]).all()
    assert set(p["true_known_unknown"]) == {"known", "unknown"}
