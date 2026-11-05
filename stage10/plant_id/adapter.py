"""Plant-identification adapter with abstention (M4-06). Library code only: NOT wired into any farmer-facing application
(integration stays behind its explicit owner approval gate).

Responsibilities kept separate (PROPOSAL.md section 1):
  * this adapter only SUGGESTS candidate taxa or refuses; it never diagnoses, never sets the user-confirmed crop and never
    makes the frozen six-crop disease model applicable to a new crop;
  * it answers only when the top-1 score reaches the threshold frozen on DEVELOPMENT data (protocol v2 operating point);
    with no resolved operating point it does not run at all ('not_run');
  * scores are softmax match scores over the 40-taxon vocabulary, not calibrated probabilities, and are never shown as
    percentages;
  * nothing leaves the device: no external identification service exists in this module.

The model backend is injected. BioCLIPBackend loads the pinned local BioCLIP 2 weights and must run in the isolated
environment (C:/Users/nihar/venvs/agrovaidya-plantid); tests use a fake backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from stage10.plant_id.pilot import common as PC

HERE = Path(__file__).resolve().parent
OPERATING_POINT = HERE / "d2" / "protocol_v2_operating_point.json"
VOCAB = HERE / "pilot" / "vocabulary_v1.json"
MIN_SIDE = 224
MAX_CANDIDATES = 5
SEMANTICS = "softmax over the vocabulary taxa; ordinal match score, not a calibrated probability"
THRESHOLD_SOURCE = "development data only (protocol v2); never test data"

# Language-independent message keys. English is authoritative; no other language has a reviewed translation yet, so
# none is emitted (localisation boundary: add reviewed translations by key, never machine-translate here).
MESSAGES = {
    "plant_id.candidates": "Possible plant identity suggested by the image model. This is a suggestion only: please confirm "
                           "the plant name yourself. It says nothing about plant health or disease.",
    "plant_id.unidentified": "Potentially unfamiliar to this model; further investigation is recommended. The image did not "
                             "match any plant the model supports closely enough. Please enter the plant name yourself, or ask "
                             "your local Krishi Vigyan Kendra (KVK).",
    "plant_id.insufficient_image": "The photo could not be used for plant identification. Please take a clear, well-lit photo "
                                   "of the whole plant or a leaf, at least 224 pixels on its shorter side.",
    "plant_id.not_run": "Plant identification is not available: no operating threshold has been validated on development "
                        "data. Please enter the plant name yourself.",
}
LIMITATIONS = [
    "Suggestion only: the plant you confirm is always used, never this suggestion.",
    "Plant identification is not a disease assessment.",
    "Only the 40 crops of vocabulary v1 can be suggested; other plants should be refused but can occasionally be wrongly matched.",
    "Validated identification performance has not been established for this model in India.",
]


class Backend(Protocol):
    name: str
    version: str
    weights_sha256: str
    licence: str

    def scores(self, image_bytes: bytes) -> list[tuple[str, float]]:
        """All vocabulary taxa with their softmax scores, highest first."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_operating_point(path: Path = OPERATING_POINT) -> dict:
    if not path.exists():
        return {"status": "absent", "threshold": None, "threshold_source": THRESHOLD_SOURCE, "operating_point_sha256": None}
    op = json.loads(path.read_text(encoding="utf-8"))
    resolved = op.get("status") == "resolved" and isinstance(op.get("tau"), (int, float))
    return {"status": "resolved" if resolved else "unresolved", "threshold": float(op["tau"]) if resolved else None,
            "threshold_source": THRESHOLD_SOURCE, "operating_point_sha256": _sha(path)}


class PlantIdAdapter:
    def __init__(self, backend: Backend, operating_point: dict | None = None, vocab_path: Path = VOCAB, kb=None):
        self.backend = backend
        self.op = operating_point if operating_point is not None else load_operating_point()
        v = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.vocab = {t["scientific_name"]: t for t in v["taxa"]}
        self.vocab_meta = {"id": "vocabulary_v1", "version": str(v["version"]), "n_taxa": len(v["taxa"]), "sha256": _sha(vocab_path)}
        self.kb_ids = {}
        if kb is not None:
            for pid, p in kb.plants.items():
                self.kb_ids[" ".join(p["scientific_name"].split()[:2])] = pid

    def _result(self, status, reason, candidates=()):
        return {"status": status, "message_key": f"plant_id.{status}", "message": {"en": MESSAGES[f"plant_id.{status}"]},
                "model": {"name": self.backend.name, "version": self.backend.version, "weights_sha256": self.backend.weights_sha256,
                          "licence": self.backend.licence, "runs_locally": True, "external_service": None},
                "vocabulary": self.vocab_meta, "operating_point": dict(self.op), "candidates": list(candidates),
                "abstention": {"abstained": status != "candidates", "reason": reason},
                "requires_user_confirmation": True, "sets_confirmed_crop": False, "disease_assessment_performed": False,
                "evaluated_scope": {"evaluation_id": None, "validated": False},
                "privacy": {"image_left_device": False, "consent_reference": None},
                "limitations": [{"en": x} for x in LIMITATIONS]}

    def identify(self, image_bytes: bytes) -> dict:
        if self.op["status"] != "resolved":
            return self._result("not_run", "no_validated_operating_point")
        im, _, why = PC.load_image(image_bytes, MIN_SIDE)
        if im is None:
            return self._result("insufficient_image", "insufficient_image")
        scores = sorted(self.backend.scores(image_bytes), key=lambda x: -x[1])
        tau = self.op["threshold"]
        if not scores or scores[0][1] < tau:
            return self._result("unidentified", "below_threshold")
        cands = []
        for i, (name, s) in enumerate(scores[:MAX_CANDIDATES], 1):
            t = self.vocab[name]
            cands.append({"rank": i, "taxon_rank": "species", "scientific_name": name,
                          "common_names": {"en": t["common_name_en"]}, "kb_plant_id": self.kb_ids.get(name),
                          "score": round(float(s), 6), "score_semantics": SEMANTICS, "score_is_calibrated_probability": False,
                          "above_threshold": float(s) >= tau,
                          "limitations": [] if i == 1 else [{"en": "Lower-ranked alternative; shown for comparison only."}]})
        return self._result("candidates", None, cands)


class BioCLIPBackend:
    """Pinned local BioCLIP 2 (template T1, vocabulary v1). Run only in the isolated environment."""
    name, version, licence = "BioCLIP 2", "2957b322090f9cb17ae72c71981c7218a28d81e0", "MIT (weights); see LICENCE_AND_OVERLAP_REVIEW.md"

    def __init__(self, device: str = "cpu"):
        from stage10.plant_id.pilot import evaluate_pilot as E
        self.weights_sha256 = E.WEIGHTS_SHA
        self._scorer = E.Scorer("T1", device)

    def scores(self, image_bytes: bytes) -> list[tuple[str, float]]:
        import io
        from PIL import Image
        s = self._scorer
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = s.pre(im).unsqueeze(0).to(s.device)
        with s.torch.no_grad():
            f = s.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (s.scale * f @ s.tf.T).softmax(dim=-1)[0].float().cpu().tolist()
        return sorted(((s.vocab["taxa"][i]["scientific_name"], v) for i, v in enumerate(p)), key=lambda x: -x[1])
