"""SYNTHETIC inputs for report-engine unit tests.

`synthetic_assessment` builds an object with exactly the shape FrozenOSRAdapter.assess() returns (model_prediction +
osr_status), with invented scores. No image is opened and no final-test or held-out data is used. The model identity and
frozen threshold are the real frozen values (read-only), because the engine rejects anything else.
"""
from __future__ import annotations

from stage10.osr.osr_adapter import MSG_KNOWN, MSG_UNKNOWN, NOTE, RULE, crop_of_class
from stage10.report_engine.engine import frozen_model_versions, frozen_threshold
from stage10.report_engine.inputs import ImageInfo, ReportRequest

TS = "2026-10-14T12:00:00+00:00"
IMG = ImageInfo(sha256="ab" * 32, width=256, height=256, format="JPEG")


def synthetic_assessment(cls: str, known: bool, top3=None) -> dict:
    t = frozen_threshold()
    score = t - 1.0 if known else t + 1.0
    return {
        "model_prediction": {"ran": True, "predicted_class": cls, "predicted_crop": crop_of_class(cls),
                             "top3": [{"class": c, "softmax": p} for c, p in (top3 or [(cls, 0.9)])],
                             "softmax_is_calibrated": False, "model_versions": dict(frozen_model_versions())},
        "osr_status": {"decision": "known" if known else "unknown", "score_name": "contrastive_energy", "score": score,
                       "frozen_threshold": t, "rule": RULE, "percentile_known_validation": 0.5 if known else 0.99,
                       "borderline_display": False,
                       "other_scores": {"msp": 0.9, "msp_says_unknown": False, "energy": -8.0, "energy_says_unknown": False},
                       "message": dict(MSG_KNOWN if known else MSG_UNKNOWN), "interpretation_note": dict(NOTE)},
    }


def request(**kw) -> ReportRequest:
    kw.setdefault("image", IMG)
    kw.setdefault("generated_at", TS)
    return ReportRequest(**kw)


EB_FEATURES = {"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}


def known_eb(**kw):
    """Tomato, user-confirmed; model says Tomato___Early_blight (known); user reports matching symptoms."""
    base = dict(plant_name="Tomato", plant_confirmation="confirmed", reported_features=dict(EB_FEATURES),
                model_assessment=synthetic_assessment("Tomato___Early_blight", True))
    base.update(kw)
    return request(**base)


def unknown_tomato(**kw):
    base = dict(plant_name="Tomato", plant_confirmation="confirmed", reported_features=dict(EB_FEATURES),
                model_assessment=synthetic_assessment("Tomato___Early_blight", False))
    base.update(kw)
    return request(**base)
