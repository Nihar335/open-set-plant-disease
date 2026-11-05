"""Stage-8 tests: score orientation (larger = more unknown) and known-validation ECDF normalisation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from src.osr.fusion import ECDFNormalizer, orient
from src.osr.scores import energy_score, msp_scores


def test_msp_direction_lower_msp_is_more_unknown():
    logits = np.array([[10.0, 0, 0], [1.0, 0.9, 0.8]])            # confident vs uncertain
    msp = msp_scores(logits)["msp"]
    u = orient(msp, "lower_is_unknown")
    assert msp[0] > msp[1] and u[1] > u[0]


def test_energy_direction_flat_logits_are_more_unknown():
    e = energy_score(np.array([[10.0, 0, 0], [0.1, 0.1, 0.1]]), 1.0)
    assert orient(e, "higher_is_unknown")[1] > orient(e, "higher_is_unknown")[0]


def test_distance_direction_farther_is_more_unknown():
    d = np.array([0.1, 5.0])                                       # Mahalanobis / prototype distance
    u = orient(d, "higher_is_unknown")
    assert u[1] > u[0]


def test_orient_rejects_bad_input():
    with pytest.raises(ValueError):
        orient([1.0], "sideways")
    with pytest.raises(ValueError):
        orient([np.nan], "higher_is_unknown")


def test_ecdf_midrank_values_and_range():
    n = ECDFNormalizer().fit([1, 2, 3, 4])
    np.testing.assert_allclose(n.transform([0, 1, 2.5, 4, 9]), [0, 0.125, 0.5, 0.875, 1.0])
    np.testing.assert_allclose(n.transform([2, 2]), [0.375, 0.375])  # ties get the mid-rank


def test_ecdf_is_monotone_and_preserves_ranking():
    rng = np.random.default_rng(0)
    ref, x = rng.normal(size=500), rng.normal(size=200) * 2
    t = ECDFNormalizer().fit(ref).transform(x)
    order = np.argsort(x)
    assert (np.diff(t[order]) >= 0).all()
    inside = (x > ref.min()) & (x < ref.max())
    assert (rankdata(t[inside]) == rankdata(x[inside])).all() or np.corrcoef(rankdata(t[inside]), rankdata(x[inside]))[0, 1] > 0.999


def test_ecdf_of_reference_is_uniformish_and_fitted_on_known_only():
    ref = np.arange(1000.0)
    t = ECDFNormalizer().fit(ref).transform(ref)
    assert abs(t.mean() - 0.5) < 1e-9 and t.min() > 0 and t.max() < 1
    with pytest.raises(ValueError):
        ECDFNormalizer().fit([])
