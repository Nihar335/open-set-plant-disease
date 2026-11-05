"""Stage-8 tests: redundancy clusters, representatives, fusion rules, thresholds, oracle grid, protocol."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.osr_metrics import ranking_metrics
from src.osr.fusion import (cluster_weights, complete_linkage_clusters, fuse_max, fuse_mean, fuse_weighted,
                            known_quantile_threshold, representatives, simplex_grid, spearman_matrix)


def test_clusters_group_redundant_signals_only():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    cols = {"msp": a + 0.05 * rng.normal(size=2000), "energy": a, "maha": rng.normal(size=2000)}
    names, R = spearman_matrix(cols)
    cl = complete_linkage_clusters(names, R, 0.8)
    assert sorted(map(sorted, cl)) == [["energy", "msp"], ["maha"]]
    rep = representatives(names, R, cl)
    assert rep[("maha",)] == "maha" and rep[("msp", "energy")] in {"msp", "energy"}


def test_complete_linkage_requires_all_pairs():
    names = ["a", "b", "c"]
    R = np.array([[1, .9, .9], [.9, 1, .5], [.9, .5, 1]])        # a~b, a~c but b !~ c
    cl = complete_linkage_clusters(names, R, 0.8)
    assert len(cl) == 2 and all(len(c) <= 2 for c in cl)


def test_cluster_weights_sum_to_one_and_split_within_cluster():
    w = cluster_weights([["msp", "energy"], ["maha"]])
    assert w == {"msp": 0.25, "energy": 0.25, "maha": 0.5}


def test_fusion_rules():
    P = {"a": np.array([0.1, 0.9]), "b": np.array([0.5, 0.2])}
    np.testing.assert_allclose(fuse_mean(P, ["a", "b"]), [0.3, 0.55])
    np.testing.assert_allclose(fuse_max(P, ["a", "b"]), [0.5, 0.9])
    np.testing.assert_allclose(fuse_weighted(P, {"a": 0.75, "b": 0.25}), [0.2, 0.725])
    with pytest.raises(ValueError):
        fuse_weighted(P, {"a": 0.5, "b": 0.6})


def test_fusion_combines_complementary_signals():
    """Two signals that each see a different half of the unknowns: their mean beats either alone."""
    rng = np.random.default_rng(0)
    nk, nu = 3000, 2000
    k1, k2 = rng.uniform(size=nk), rng.uniform(size=nk)
    u1 = np.r_[rng.uniform(0.8, 1, nu // 2), rng.uniform(size=nu // 2)]
    u2 = np.r_[rng.uniform(size=nu // 2), rng.uniform(0.8, 1, nu // 2)]
    a1, a2 = ranking_metrics(k1, u1)["auroc"], ranking_metrics(k2, u2)["auroc"]
    af = ranking_metrics((k1 + k2) / 2, (u1 + u2) / 2)["auroc"]
    assert af > max(a1, a2)


def test_threshold_accepts_target_share_of_known():
    u = np.arange(1, 1001) / 1000
    t = known_quantile_threshold(u, 0.95)
    assert (u <= t).mean() >= 0.95 and (u <= t).mean() < 0.952


def test_simplex_grid_is_complete_and_normalised():
    g = list(simplex_grid(3, 0.5))
    assert len(g) == 6 and all(np.isclose(w.sum(), 1) for w in g)
    assert len(list(simplex_grid(6, 0.1))) == 3003


def test_fusion_config_uses_known_data_only():
    from src.utils.config import load_config
    cfg = load_config()
    f = cfg["osr_fusion"]
    assert f["primary"] in f["fusions"] and f["normalisation"].startswith("known_validation")
    assert f["redundancy"]["statistic"].endswith("known_validation")
    assert all(v[2] in {"higher_is_unknown", "lower_is_unknown"} for v in f["candidate_signals"].values())
    assert f["candidate_signals"]["orig_msp"][2] == "lower_is_unknown"
