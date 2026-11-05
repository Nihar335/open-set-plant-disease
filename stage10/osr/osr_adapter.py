"""Thin, read-only wrapper around the frozen Stage-9 `OSRPredictor`.

* Imports and calls `src.inference.predictor.OSRPredictor` unchanged; it never re-implements scores or thresholds.
* Before loading, verifies that the protected Stage-9 files still have their baseline hashes (stage10/ASSET_MANIFEST.json).
* Returns the raw predictor dict untouched (`raw`) plus the two report sub-objects (`model_prediction`, `osr_status`)
  in the Stage-10 schema shape. The known/unknown decision is copied verbatim from the predictor; language and any
  future LLM have no influence on it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stage10.guards import assert_not_final_test

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "stage10" / "ASSET_MANIFEST.json"

CROP_OF_CLASS_PREFIX = {"Apple": "Apple", "Corn_(maize)": "Corn", "Grape": "Grape", "Pepper,_bell": "Pepper bell",
                        "Potato": "Potato", "Tomato": "Tomato"}
RULE = "accept as known iff score <= frozen_threshold (95% known-validation acceptance; Stage 7)"
MSG_KNOWN = {"en": "The image meets the frozen model's acceptance criterion for one of its 21 known classes."}
MSG_UNKNOWN = {"en": "Potentially unfamiliar to this model; further investigation is recommended. The image does not meet the current "
                     "model's acceptance criteria for a known class. This may reflect an unseen disease, image-quality problems, a "
                     "different plant, environmental stress, or domain shift. The OSR output does not establish the actual cause."}
NOTE = {"en": "Research-model output (Stage 7 contrastive Energy, threshold frozen on known validation images). It is not a laboratory "
              "confirmation, and the softmax values are not calibrated probabilities of correctness."}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def crop_of_class(cls: str | None) -> str | None:
    if cls is None:
        return None
    return CROP_OF_CLASS_PREFIX.get(cls.split("___")[0])


def verify_protected_files() -> dict:
    """Raise if any protected Stage-9 file differs from its baseline hash. Returns the protected-hash table."""
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = {p: h for p, h in man["protected_stage9_files"].items() if _sha256(ROOT / p) != h}
    if bad:
        raise RuntimeError(f"protected Stage-9 files changed: {sorted(bad)}")
    return man["protected_stage9_files"]


class FrozenOSRAdapter:
    def __init__(self, predictor=None, device: str | None = None):
        prot = verify_protected_files()
        if predictor is None:
            from src.inference.predictor import OSRPredictor
            predictor = OSRPredictor(device=device)
        self.p = predictor
        cfg = predictor.cfg
        self.versions = {"stage3_checkpoint_sha256": cfg["osr_msp"]["checkpoint_sha256"],
                         "contrastive_checkpoint_sha256": json.loads((ROOT / "results/metrics/contrastive_environment.json").read_text())["checkpoint_sha256"],
                         "linear_head_sha256": prot["models/contrastive/linear_head.npz"]}

    @property
    def frozen_threshold(self) -> float:
        return float(self.p.t_cenergy)

    def assess(self, image) -> dict:
        if isinstance(image, (str, Path)):
            assert_not_final_test(image)
        raw = self.p.predict(image)
        known = raw["decision"] == "KNOWN DISEASE"
        assert known == (not raw["contrastive_energy_says_unknown"])
        model_prediction = {
            "ran": True,
            "predicted_class": raw["predicted_known_class"],
            "predicted_crop": crop_of_class(raw["predicted_known_class"]),
            "top3": [{"class": c, "softmax": float(p)} for c, p in raw["top3"]],
            "softmax_is_calibrated": False,
            "model_versions": dict(self.versions),
        }
        osr_status = {
            "decision": "known" if known else "unknown",
            "score_name": "contrastive_energy",
            "score": float(raw["contrastive_energy"]),
            "frozen_threshold": float(raw["contrastive_energy_threshold"]),
            "rule": RULE,
            "percentile_known_validation": float(raw["contrastive_energy_percentile_known_val"]),
            "borderline_display": False,
            "other_scores": {"msp": float(raw["msp"]), "msp_says_unknown": bool(raw["msp_says_unknown"]),
                             "energy": float(raw["energy"]), "energy_says_unknown": bool(raw["energy_says_unknown"])},
            "message": dict(MSG_KNOWN if known else MSG_UNKNOWN),
            "interpretation_note": dict(NOTE),
        }
        return {"raw": raw, "model_prediction": model_prediction, "osr_status": osr_status}
