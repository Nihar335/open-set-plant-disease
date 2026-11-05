"""Stage-5 energy score: numerical stability, direction, thresholds, labels, prediction schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.osr_metrics import ranking_metrics, threshold_metrics
from src.osr.scores import energy_score, msp_scores
from src.utils.config import PROJECT_ROOT

K = 21


def test_extreme_logits_finite():
    logits = np.array([[1e4] + [0.0] * (K - 1), [-1e4] * K, [1e4] * K, [3e38 / 1e6] + [-3e38 / 1e6] * (K - 1),
                       np.linspace(-500, 500, K)])
    for T in (0.5, 1.0, 5.0):
        e = energy_score(logits, T)
        assert np.isfinite(e).all(), (T, e)
    # naive torch exp-sum overflows on the same input -> the stable form is required
    assert not torch.isfinite(torch.log(torch.exp(torch.tensor(logits[0])).sum()))


def test_matches_reference_logsumexp():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 8, (200, K))
    for T in (0.5, 1.0, 2.0):
        ref = -T * torch.logsumexp(torch.tensor(x / T), dim=1).numpy()
        assert np.allclose(energy_score(x, T), ref, atol=1e-10)


def test_direction_confident_lower_than_uniform():
    confident = np.array([[10.0] + [0.0] * (K - 1)])
    uniform = np.zeros((1, K))
    ec, eu = energy_score(confident)[0], energy_score(uniform)[0]
    assert ec < eu                                            # lower energy = more in-distribution
    assert abs(eu - (-np.log(K))) < 1e-12
    assert abs(ec - (-(10 + np.log(1 + (K - 1) * np.exp(-10))))) < 1e-12


def test_not_shift_invariant_unlike_msp():
    a, b = np.zeros((1, K)), np.full((1, K), 5.0)            # identical softmax, different logit magnitude
    assert np.allclose(msp_scores(a)["msp"], msp_scores(b)["msp"])
    assert abs((energy_score(b) - energy_score(a))[0] + 5.0) < 1e-12


def test_log_msp_identity_at_T1():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 5, (50, K))
    s = msp_scores(x)
    assert np.allclose(np.log(s["msp"]), x.max(1) + energy_score(x, 1.0), atol=1e-9)


def test_threshold_direction_and_label_handling():
    e_known = np.array([-12.0, -11.0, -10.0, -3.0])          # last known is uncertain
    e_unknown = np.array([-2.0, -9.5])
    tau = -5.0                                               # accept as known iff E <= tau
    m = threshold_metrics(e_known, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), e_unknown, tau, K)
    assert m["known_acceptance_rate"] == 0.75 and m["rejected_known"] == 1
    assert m["unknown_recall"] == 0.5 and m["rejected_unknown"] == 1


def test_metric_direction_with_energy():
    rng = np.random.default_rng(2)
    known, unknown = rng.normal(-12, 1, 500), rng.normal(-4, 1, 500)   # unknowns have higher energy
    assert ranking_metrics(known, unknown)["auroc"] > 0.99
    assert ranking_metrics(unknown, known)["auroc"] < 0.01


PRED = PROJECT_ROOT / "results" / "predictions" / "osr_energy_predictions.csv"


@pytest.mark.skipif(not PRED.exists(), reason="Stage-5 evaluation not run yet")
def test_prediction_file_schema():
    p = pd.read_csv(PRED, low_memory=False)
    need = ["image_path", "dataset", "true_class", "true_known_unknown", "predicted_class", "msp", "energy",
            "energy_threshold", "predicted_known_unknown", "correct_osr_decision"]
    assert set(need) <= set(p.columns)
    assert p["image_path"].is_unique and set(p["true_known_unknown"]) == {"known", "unknown"}
    e, t = p["energy"].astype(float), p["energy_threshold"].astype(float)
    assert np.isfinite(e).all() and t.nunique() == 1
    assert ((e <= t) == (p["predicted_known_unknown"] == "known")).all()          # accept iff E <= tau
    assert (p.loc[p["predicted_known_unknown"] == "unknown", "predicted_class"] == "UNKNOWN").all()
    assert ((p["predicted_known_unknown"] == p["true_known_unknown"]) == p["correct_osr_decision"]).all()
