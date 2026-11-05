"""Structural (JSON Schema 2020-12) + semantic validation of Stage-10 report objects.

Structural rules live in the *.schema.json files. Semantic rules that JSON Schema cannot express are checked here:
  * every referenced source_id / symptom_id / recommendation_id / candidate_id exists in the report;
  * no dose may appear while the Stage-10 config forbids pesticide doses / fertilizer prescriptions;
  * a dose must cite one of the recommendation's own sources;
  * test-fixture sources are rejected unless explicitly allowed (tests only);
  * non-English reports carry the translation for every safety warning / disclaimer and a translation-review flag;
  * candidate count respects config.candidates.max_candidates;
  * when a frozen threshold is supplied, osr_status.frozen_threshold must equal it exactly.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent
ROOT = SCHEMA_DIR.parents[1]
BASE = "https://agrovaidya.invalid/stage10/schemas/"
NAMES = ["common", "plant_identity", "symptom_observation", "disease_candidate", "agricultural_recommendation", "plant_health_report",
         "plant_identification"]


class ReportValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:10]) + (f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""))
        self.errors = errors


@lru_cache(maxsize=1)
def stage10_config() -> dict:
    return yaml.safe_load((ROOT / "stage10" / "config.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _registry() -> Registry:
    reg = Registry()
    for n in NAMES:
        s = json.loads((SCHEMA_DIR / f"{n}.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(s)
        reg = reg.with_resource(BASE + f"{n}.schema.json", Resource.from_contents(s))
    return reg


def validator(name: str) -> Draft202012Validator:
    s = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(s, registry=_registry(), format_checker=FormatChecker())


def structural_errors(obj: dict, name: str = "plant_health_report") -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator(name).iter_errors(obj)]


def _walk_localized(obj, path=""):
    """Yield (path, localized_text) for every dict that looks like a localized_text."""
    if isinstance(obj, dict):
        if "en" in obj and all(isinstance(v, str) for v in obj.values()):
            yield path, obj
        for k, v in obj.items():
            yield from _walk_localized(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_localized(v, f"{path}/{i}")


def semantic_errors(r: dict, *, allow_fixture_sources: bool = False, frozen_threshold: float | None = None) -> list[str]:
    cfg = stage10_config()
    err: list[str] = []
    src = [s["source_id"] for s in r.get("evidence_sources", [])]
    if len(src) != len(set(src)):
        err.append("evidence_sources: duplicate source_id")
    src_set = set(src)
    if not allow_fixture_sources and any(s.get("source_type") == "test_fixture" for s in r.get("evidence_sources", [])):
        err.append("evidence_sources: test_fixture sources are not allowed in real reports")

    def check_ids(ids, where):
        for i in ids:
            if i not in src_set:
                err.append(f"{where}: source_id {i} not in evidence_sources")

    recs = [(sec, x) for sec in ("recommended_tests", "treatment_guidance", "nutrition_guidance", "soil_guidance", "immediate_actions")
            for x in r.get(sec, [])]
    rec_ids = [x["recommendation_id"] for _, x in recs]
    if len(rec_ids) != len(set(rec_ids)):
        err.append("recommendations: duplicate recommendation_id")
    test_ids = {x["recommendation_id"] for x in r.get("recommended_tests", [])}
    for sec, x in recs:
        where = f"{sec}/{x['recommendation_id']}"
        check_ids(x.get("source_ids", []), where)
        d = x.get("dose")
        if d is not None:
            if x["subcategory"] == "chemical" and not cfg["knowledge_base"]["allow_pesticide_dose"]:
                err.append(f"{where}: pesticide dose present but config forbids doses")
            if x["subcategory"] in ("fertilizer", "nutrition") and not cfg["knowledge_base"]["allow_fertilizer_prescription"]:
                err.append(f"{where}: fertilizer prescription present but config forbids prescriptions")
            if d["source_id"] not in x.get("source_ids", []):
                err.append(f"{where}: dose.source_id is not one of the recommendation's sources")
    sym_ids = [s["symptom_id"] for s in r.get("visible_symptoms", [])]
    if len(sym_ids) != len(set(sym_ids)):
        err.append("visible_symptoms: duplicate symptom_id")
    cands = r.get("candidate_diagnoses", [])
    if len(cands) > cfg["candidates"]["max_candidates"]:
        err.append(f"candidate_diagnoses: {len(cands)} > max_candidates")
    cand_ids = {c["candidate_id"] for c in cands}
    if len(cand_ids) != len(cands):
        err.append("candidate_diagnoses: duplicate candidate_id")
    for c in cands:
        w = f"candidate_diagnoses/{c['candidate_id']}"
        check_ids(c["source_ids"], w)
        check_ids(c["scientific_cause"]["source_ids"], w + "/scientific_cause")
        for s in c["supporting_symptoms"]:
            if s not in sym_ids:
                err.append(f"{w}: supporting symptom {s} not in visible_symptoms")
        for t in c["recommended_confirmatory_tests"]:
            if t not in test_ids:
                err.append(f"{w}: confirmatory test {t} not in recommended_tests")
    for row in r.get("differential_diagnosis", {}).get("rows", []):
        for cid, cell in row["per_candidate"].items():
            if cid not in cand_ids:
                err.append(f"differential_diagnosis: unknown candidate {cid}")
            check_ids(cell["source_ids"], f"differential_diagnosis/{row['finding']}/{cid}")
    tax = r.get("plant_identity", {}).get("taxonomy", {})
    for rank, v in tax.items():
        check_ids(v.get("source_ids", []), f"plant_identity/taxonomy/{rank}")
    # language
    lang = r.get("language", "en")
    if lang not in cfg["language"]["supported"]:
        err.append(f"language {lang} not enabled in stage10/config.yaml")
    if lang != "en":
        for p, t in list(_walk_localized(r.get("safety_warnings", []), "/safety_warnings")) + [("/disclaimer", r.get("disclaimer", {}))]:
            if lang not in t:
                err.append(f"{p}: missing '{lang}' translation of a safety-critical text")
        tr = r.get("review_status", {}).get("translation_review", {})
        if lang not in tr:
            err.append(f"review_status.translation_review: missing entry for '{lang}'")
        elif lang in cfg["language"]["requires_fluent_review"] and tr[lang] == "not_applicable":
            err.append(f"review_status.translation_review.{lang} cannot be not_applicable")
    # frozen threshold integrity
    if frozen_threshold is not None and r.get("osr_status", {}).get("decision") != "not_run":
        if r["osr_status"].get("frozen_threshold") != frozen_threshold:
            err.append("osr_status.frozen_threshold differs from the frozen Stage-7 threshold")
    try:
        datetime.fromisoformat(r.get("created_at", "").replace("Z", "+00:00"))
    except ValueError:
        err.append("created_at: not an ISO-8601 date-time")
    return err


def validate_report(r: dict, **kw) -> None:
    errs = structural_errors(r) + semantic_errors(r, **kw)
    if errs:
        raise ReportValidationError(errs)
