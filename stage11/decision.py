"""Stage 11 decision rule, shared by evaluate.py (tuning / testing) and engine.py (the app).

For a photo, the user's crop and the weather:
  p_full   = softmax(logits / T)                        image model, all classes
  in_mass  = sum of p_full over the user's crop's classes  (how much the photo looks like this crop's classes)
  p_crop   = p_full restricted to the crop, renormalised
  match_d  = cosine(detected-symptom probabilities, symptom profile of disease d)
  fused_d  ∝ p_crop_d · match_d^beta · weather_d          renormalised over the crop's classes
  fit      = min(1, in_mass / m0)                         photo-fit
  q_d      ∝ exp(6 · match_d) · weather_d                 ranking by detected symptoms alone
  conf_d   = (fit·fused_d + (1-fit)·q_d) · (fit + (1-fit)·0.75)
Decision (`may_name`): conf_top >= THRESHOLD, the crop is "diagnosable" by the image model (KB.model_coverage) and
no other crop takes more image probability than the stated crop -> that disease; otherwise "Unknown disease" with
the top three.

T, beta and m0 are fitted on validation photos only (evaluate.py) and stored in models/stage11/calibration.json.
m0 is the 5th percentile of in-crop mass over all field validation photos; that pool is ~90% apple-dataset photos,
so on independent web/field photos (PlantDoc val) about 39% fall below m0 (audit 2026-10-23, stage11/audit/).
It acts mainly as the guard against photos of other plants / non-plants; see stage11/audit/AUDIT_REPORT.md.
Weather is not used in the score: the 0.8..1.2 factor was hand-set and never validated (no weather-labelled
photos), so the app reports weather risk next to each disease instead.
"""
from __future__ import annotations

import numpy as np

THRESHOLD = 0.60   # lowered from 0.80 at the user's request (2026-10-23)
MATCH_FLOOR = 0.05


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def symptom_match(sym: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    """(N,S) x (C,S) -> (N,C) cosine similarity, floored."""
    a = sym / (np.linalg.norm(sym, axis=-1, keepdims=True) + 1e-8)
    b = profiles / (np.linalg.norm(profiles, axis=-1, keepdims=True) + 1e-8)
    return np.clip(a @ b.T, MATCH_FLOOR, 1.0)


SYMPTOM_SHARPNESS = 6.0   # softmax temperature for ranking diseases by symptom match (same as database-only path)
SYMPTOM_CAP = 0.75        # confidence ceiling for evidence that comes from symptom matching alone


def fuse(logits, sym, crop_mask, profiles, T=1.0, beta=0.0, m0=0.0, weather=None):
    """Returns dict of arrays: conf (N,C), fused (N,C), p_crop (N,C), in_mass (N,), match (N,C), fit (N,).

    When the photo fits the crop's training classes (fit = 1) the result is the image-based `fused`.
    When it does not (fit < 1: the image model puts much of its probability on other crops), the ranking is blended
    towards q = symptom-match ranking over the crop's diseases (detected symptoms vs database profiles, x weather):
        final = fit * fused + (1 - fit) * q,   conf = final * (fit + (1 - fit) * SYMPTOM_CAP)
    so a poorly fitting photo is pulled towards the symptom ranking and its top three follow the visible symptoms.
    """
    logits, sym, crop_mask = np.atleast_2d(logits), np.atleast_2d(sym), np.atleast_2d(crop_mask)
    p_full = softmax(logits / T)
    in_mass = (p_full * crop_mask).sum(1)
    p_crop = p_full * crop_mask
    p_crop = p_crop / (p_crop.sum(1, keepdims=True) + 1e-12)
    match = symptom_match(sym, profiles)
    w = np.atleast_2d(weather) if weather is not None else 1.0
    s = p_crop * match ** beta * w
    fused = s / (s.sum(1, keepdims=True) + 1e-12)
    q = np.exp(SYMPTOM_SHARPNESS * match) * crop_mask * w
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    fit = np.minimum(1.0, in_mass / m0) if m0 > 0 else np.ones_like(in_mass)
    final = fit[:, None] * fused + (1 - fit[:, None]) * q
    conf = final * (fit + (1 - fit) * SYMPTOM_CAP)[:, None]
    return {"conf": conf, "final": final, "fused": fused, "symptom_rank": q, "p_crop": p_crop, "p_full": p_full,
            "in_mass": in_mass, "match": match, "fit": fit}


def crop_check(p_full: np.ndarray, class_crops: np.ndarray, stated_mask: np.ndarray) -> dict:
    """Image probability per crop for one photo: does another crop look more likely than the stated one?"""
    mass = {c: float(p_full[class_crops == c].sum()) for c in dict.fromkeys(class_crops)}
    stated = float(p_full[stated_mask].sum())
    best = max(mass, key=mass.get)
    return {"stated_crop_mass": stated, "most_likely_crop": best, "most_likely_mass": mass[best],
            "conflict": bool(not stated_mask[class_crops == best].any() and mass[best] > stated)}


def may_name(top_conf: float, coverage_level: str, conflict: bool) -> bool:
    """The single rule that allows a named disease / healthy verdict (shared by engine.py and evaluate_multicrop.py)."""
    return top_conf >= THRESHOLD and coverage_level == "diagnosable" and not conflict
