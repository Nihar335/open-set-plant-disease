"""OSR scores computed from classifier logits."""
from __future__ import annotations

import numpy as np


def msp_scores(logits: np.ndarray) -> dict:
    """Maximum softmax probability, computed in float64.

    Returns msp, u = 1 - msp (unknown-ness; computed as the sum of the NON-max softmax
    probabilities, so it keeps precision when msp is within 1e-7 of 1), predicted class, top-3.
    """
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)             # max logit -> 0
    e = np.exp(z)
    denom = e.sum(axis=1)
    pred = z.argmax(axis=1)
    not_max = np.ones_like(e, dtype=bool)
    not_max[np.arange(len(e)), pred] = False
    rest = np.where(not_max, e, 0.0).sum(axis=1)      # sum over NON-max classes directly (no 1 - x cancellation)
    u = rest / denom
    probs = e / denom[:, None]
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {"probs": probs, "msp": 1.0 - u, "u": u, "pred": pred, "top3": top3,
            "top3_probs": np.take_along_axis(probs, top3, axis=1)}


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free energy E(x) = -T * logsumexp(f(x) / T)   (Liu et al., NeurIPS 2020).

    Sign convention: LOWER energy = more in-distribution (known); HIGHER energy = stronger OOD
    signal, so E itself is an unknown-ness score (same direction as u = 1 - MSP).
    Computed in float64 with the max-shift log-sum-exp trick: no overflow for any finite logits.
    Note: unlike MSP, E is not invariant to adding a constant to all logits (E shifts by -c):
    it keeps the logit *magnitude* that softmax normalises away.  At T = 1,
    log MSP = max_c f_c + E, i.e. MSP and energy differ exactly by the max-logit term.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=1)
    return -temperature * (m + np.log(np.exp(z - m[:, None]).sum(axis=1)))
