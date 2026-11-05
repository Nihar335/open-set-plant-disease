"""Stage-9 tests: the frozen demo predictor reproduces the cached Stage-4/5/7 scores on VALIDATION images (no test data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import resolve

pytestmark = pytest.mark.skipif(not resolve("models/contrastive/linear_head.npz").exists(), reason="linear head not exported")


@pytest.fixture(scope="module")
def predictor():
    from src.inference.predictor import OSRPredictor
    return OSRPredictor()


def _val_rows():
    z = np.load(resolve("results/predictions/cache/msp_logits_val.npz"))
    meta = pd.read_csv(resolve("results/embeddings_contrastive/val_meta.csv"), low_memory=False)
    E = np.load(resolve("results/embeddings_contrastive/val_embeddings.npz"))["embeddings"].astype(np.float64)
    assert list(z["paths"]) == meta["image_path"].tolist()
    idx = [int(np.flatnonzero(meta["role"] == "known")[i]) for i in (0, 500, 2000)] + [int(np.flatnonzero(meta["role"] == "unknown_val")[0])]
    return z["logits"], meta, E, idx


def test_predictor_reproduces_cached_scores(predictor):
    from src.osr.scores import energy_score, msp_scores
    logits, meta, E, idx = _val_rows()
    for i in idx:
        out = predictor.predict(resolve(meta["image_path"].iat[i]))
        ms = msp_scores(logits[i:i + 1].astype(np.float64))
        assert out["predicted_known_class"] == predictor.classes[int(ms["pred"][0])]
        assert abs(out["energy"] - energy_score(logits[i:i + 1], 1.0)[0]) < 1e-2
        f = E[i] / np.linalg.norm(E[i])
        ce = energy_score((f @ predictor.coef.T + predictor.intercept)[None], 1.0)[0]
        assert abs(out["contrastive_energy"] - ce) < 1e-2


def test_predictor_decision_rule_and_wording(predictor):
    _, meta, _, idx = _val_rows()
    out = predictor.predict(resolve(meta["image_path"].iat[idx[0]]))
    assert (out["decision"] == "UNKNOWN / NOVEL") == (out["contrastive_energy"] > out["contrastive_energy_threshold"])
    assert out["message"] in {"Potentially unseen / out-of-distribution input", "Known disease (within the 21 trained classes)"}
    assert 0 <= out["contrastive_energy_percentile_known_val"] <= 1 and len(out["top3"]) == 3
