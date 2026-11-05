"""Loads and validates the Stage 11 datasets (symptoms, diseases, diagnostic tests, plants, states)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"

# Crops whose diseases are also relevant to a related crop key (same plant family, same pathogen group).
CROP_GROUPS = {
    "cucurbits": ["cucurbits", "squash"],
    "squash": ["squash", "cucurbits"],
    "chilli": ["chilli", "bell pepper"],
    "bell pepper": ["bell pepper", "chilli"],
}


def _load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


@dataclass
class KB:
    symptoms: list
    diseases: list
    tests: dict
    plants: list
    states: list
    solutions: dict
    resources: dict
    symptom_ids: list = field(init=False)
    by_key: dict = field(init=False)
    model_classes: list = field(init=False)

    def __post_init__(self):
        self.symptom_ids = [s["id"] for s in self.symptoms]
        self.by_key = {d["key"]: d for d in self.diseases}
        # Output order of the disease head = order of image-trained entries in diseases.yaml.
        self.model_classes = [d for d in self.diseases if d.get("model_class")]

    # ---------------------------------------------------------------- symptom profiles
    def profile(self, key: str) -> np.ndarray:
        vec = np.zeros(len(self.symptom_ids), dtype=np.float32)
        for sid, w in self.by_key[key]["symptoms"].items():
            vec[self.symptom_ids.index(sid)] = w
        return vec

    def symptom_targets(self) -> np.ndarray:
        """Weak symptom labels for every model class, shape (n_classes, n_symptoms)."""
        return np.stack([self.profile(d["key"]) for d in self.model_classes])

    # ---------------------------------------------------------------- plants
    def resolve_plant(self, text: str):
        """Return (plant_record, match_type) for a user-typed plant name, or (None, None)."""
        q = _norm(text)
        if not q:
            return None, None
        names = {}
        for p in self.plants:
            for n in [p["common_name"], p["scientific_name"], *p.get("aliases", [])]:
                names.setdefault(_norm(n), p)
            # genus + species without the author, e.g. "solanum lycopersicum"
            names.setdefault(" ".join(_norm(p["scientific_name"]).split()[:2]), p)
        if q in names:
            return names[q], "exact"
        if q.endswith("s") and q[:-1] in names:
            return names[q[:-1]], "exact"
        close = difflib.get_close_matches(q, list(names), n=1, cutoff=0.8)
        if close:
            return names[close[0]], "fuzzy"
        return None, None

    def crop_keys(self, model_crop: str | None) -> list:
        if not model_crop:
            return []
        return CROP_GROUPS.get(model_crop, [model_crop])

    # ---------------------------------------------------------------- image-model coverage (data-driven)
    def model_coverage(self, model_crop: str | None) -> dict:
        """How far the image model covers a crop, derived from the trained classes only.

        level  diagnosable : the crop's own photos trained a healthy class AND at least one disease class, so the
                             model has learned to separate them; a named result (>= 80%) is allowed.
               limited     : the crop's own classes are only healthy, or only disease(s). Renormalising within the
                             crop would always give 100% to what exists (e.g. any soybean leaf -> "healthy"),
                             so results are leads only.
               related     : no photos of this crop; classes of a related crop (CROP_GROUPS) are used as leads only.
               none        : no trained classes; symptom matching against the database only.
        """
        own = [d for d in self.model_classes if d["crop"] == model_crop] if model_crop else []
        rel = [d for d in self.model_classes if model_crop and d["crop"] in self.crop_keys(model_crop)
               and d["crop"] != model_crop]
        healthy = [d["key"] for d in own if d["pathogen_type"] == "none"]
        diseases = [d["key"] for d in own if d["pathogen_type"] != "none"]
        if healthy and diseases:
            level = "diagnosable"
        elif own:
            level = "limited"
        elif rel:
            level = "related"
        else:
            level = "none"
        return {"level": level, "classes": [d["key"] for d in own], "healthy": healthy, "diseases": diseases,
                "related_classes": [d["key"] for d in rel],
                "related_crops": sorted({d["crop"] for d in rel})}

    # ---------------------------------------------------------------- states
    def state(self, name: str):
        q = _norm(name)
        if not q:
            return None
        for s in self.states:
            if _norm(s["name"]) == q:
                return s
        # Partial names such as "Jammu" -> "Jammu and Kashmir", or a capital such as "Hyderabad".
        hits = [s for s in self.states if f" {q} " in f" {_norm(s['name'])} " or _norm(s["capital"]) == q]
        if len(hits) == 1:
            return hits[0]
        close = difflib.get_close_matches(q, [_norm(s["name"]) for s in self.states], n=1, cutoff=0.85)
        return next((s for s in self.states if _norm(s["name"]) == close[0]), None) if close else None

    # ---------------------------------------------------------------- validation
    def validate(self) -> list:
        errors = []
        keys = [d["key"] for d in self.diseases]
        if len(keys) != len(set(keys)):
            errors.append("duplicate disease keys")
        for d in self.diseases:
            for sid in d["symptoms"]:
                if sid not in self.symptom_ids:
                    errors.append(f"{d['key']}: unknown symptom {sid}")
            for t in d["tests"]:
                if t not in self.tests:
                    errors.append(f"{d['key']}: unknown test {t}")
            for la in d.get("look_alikes", []):
                if la not in self.by_key:
                    errors.append(f"{d['key']}: unknown look-alike {la}")
            if d.get("pathogen_type") not in (None, "none") and not d["tests"]:
                errors.append(f"{d['key']}: disease without tests")
            if d.get("pathogen_type") not in (None, "none") and d["key"] not in self.solutions:
                errors.append(f"{d['key']}: no entry in solutions.yaml")
        for s in self.states:
            if s["name"] not in self.resources["states"]:
                errors.append(f"state {s['name']}: no entry in state_resources.yaml")
        crops = {d["crop"] for d in self.diseases}
        for p in self.plants:
            if p["model_crop"] and p["model_crop"] not in crops:
                errors.append(f"plant {p['common_name']}: model_crop {p['model_crop']} has no diseases")
        return errors


@lru_cache(maxsize=1)
def load_kb() -> KB:
    return KB(
        symptoms=_load("symptoms.yaml"),
        diseases=_load("diseases.yaml"),
        tests={t["id"]: t for t in _load("diagnostic_tests.yaml")},
        plants=_load("plants.yaml"),
        states=_load("india_states.yaml"),
        solutions=_load("solutions.yaml"),
        resources=_load("state_resources.yaml"),
    )


if __name__ == "__main__":
    kb = load_kb()
    errs = kb.validate()
    print(f"symptoms={len(kb.symptoms)} diseases={len(kb.diseases)} (image-trained {len(kb.model_classes)}) "
          f"tests={len(kb.tests)} plants={len(kb.plants)} states={len(kb.states)}")
    print("\n".join(errs) if errs else "validation PASS")
