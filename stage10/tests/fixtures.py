"""SYNTHETIC test fixtures for the Stage-10 report schema.

These objects exist only to exercise the schema and validator. They are NOT example outputs of the application,
contain no agricultural advice, and their "sources" are explicitly typed `test_fixture` at example.invalid URLs.
"""
from __future__ import annotations

import copy

FIX_SRC = {"source_id": "SRC-FIXTURE-1", "title": "TEST FIXTURE — not a real source", "publisher_or_author": "test suite",
           "url_or_doi": "https://example.invalid/fixture", "published_or_updated": None, "accessed_on": "2026-10-14",
           "claims_supported": ["placeholder claim used only by schema tests"], "geographic_relevance": "not_stated",
           "source_type": "test_fixture", "limitations": "synthetic", "license_or_terms": None}
T = lambda s: {"en": s}  # noqa: E731
SHA = "0" * 64


def _unk_taxon():
    return {"value": None, "status": "not_assessable", "source_ids": []}


def _base(report_type: str, decision: str) -> dict:
    return {
        "schema_version": "1.0.0", "report_id": "AV-20261014-FIXTURE", "created_at": "2026-10-14T10:00:00+00:00",
        "report_type": report_type, "language": "en", "region": {"country": "IN", "state": None, "district": None},
        "image_metadata": {"sha256": SHA, "width": 256, "height": 256, "format": "JPEG", "quality_gate": {"passed": True, "issues": []}},
        "plant_identity": {
            "user_confirmation": "confirmed", "crop_confirmed_by_user": "Tomato",
            "model_supporting_evidence": {"model_crop": "Tomato", "model_class": "Tomato___Early_blight", "agrees_with_user": True,
                                          "note": T("fixture")},
            "in_model_scope": True, "identification_level": "crop_common_name", "common_name": T("Tomato"),
            "taxonomy": {"kingdom": _unk_taxon(), "family": _unk_taxon(), "genus": _unk_taxon(), "species": _unk_taxon()},
            "species_level_claimed": False, "variety_or_cultivar": None, "growth_stage": None, "alternatives": [],
            "uncertainty": {"level": "moderate", "statements": [T("fixture")]}},
        "model_prediction": {"ran": True, "predicted_class": "Tomato___Early_blight", "predicted_crop": "Tomato",
                             "top3": [{"class": "Tomato___Early_blight", "softmax": 0.9}], "softmax_is_calibrated": False,
                             "model_versions": {"stage3_checkpoint_sha256": SHA, "contrastive_checkpoint_sha256": SHA,
                                                "linear_head_sha256": SHA}},
        "osr_status": {"decision": decision, "score_name": "contrastive_energy", "score": -6.0, "frozen_threshold": -5.123311662698785,
                       "rule": "accept as known iff score <= frozen_threshold (95% known-validation acceptance; Stage 7)",
                       "percentile_known_validation": 0.5, "borderline_display": False, "message": T("fixture"),
                       "interpretation_note": T("fixture")},
        "visible_symptoms": [{
            "symptom_id": "SYM-1", "symptom_type": "lesion_circular", "label": T("circular brown lesions"), "evidence_status": "observed",
            "origin": "image_analysis", "input_channel": "image_analysis", "location": {"plant_part": "leaf_upper", "bbox": [0.1, 0.1, 0.4, 0.4]},
            "affected_area_fraction": 0.05, "measurement_method": "fixture", "visual_evidence": T("fixture"), "confidence": "moderate",
            "image_quality_limitations": []}],
        "candidate_diagnoses": [], "differential_diagnosis": {"rows": []},
        "recommended_tests": [{
            "recommendation_id": "REC-1", "category": "diagnostic_test", "subcategory": "laboratory_test", "text": T("fixture test"),
            "consequential": False, "source_ids": ["SRC-FIXTURE-1"], "applies_to": {"crop": "Tomato", "disease_id": None, "region": None},
            "geographic_applicability": "general", "dose": None, "what_it_can_establish": T("fixture"),
            "what_it_cannot_establish": T("fixture"), "expert_review_status": "pending"}],
        "treatment_guidance": [], "nutrition_guidance": [], "soil_guidance": [], "immediate_actions": [],
        "follow_up_plan": {"enabled": False, "suggested_interval_days": None, "what_to_record": []},
        "evidence_sources": [copy.deepcopy(FIX_SRC)],
        "uncertainty": {"level": "moderate", "statements": [T("fixture")]},
        "safety_warnings": [T("fixture safety warning")], "disclaimer": T("fixture disclaimer"),
        "review_status": {"expert_review": "pending", "translation_review": {"en": "not_applicable"}},
        "content_keys": [],
        "generator": {"name": "fixture", "version": "0", "mode": "deterministic_template", "llm_used": False,
                      "knowledge_base_version": "0.0.0"},
    }


def known_report() -> dict:
    return _base("known_disease", "known")


def candidate(i: int = 1, **over) -> dict:
    c = {"candidate_id": f"CAND-{i}", "status": "candidate", "kb_disease_id": None, "candidate_name": T(f"fixture candidate {i}"),
         "cause_category": "fungal", "scientific_cause": {"value": None, "status": "unverified", "source_ids": []},
         "affected_crop": "Tomato", "crop_compatible": True, "supporting_symptoms": ["SYM-1"], "conflicting_or_missing": [],
         "reference_image_matches": [], "important_alternatives": [], "recommended_confirmatory_tests": ["REC-1"],
         "source_ids": ["SRC-FIXTURE-1"], "ranking_method": "kb_symptom_match", "ranking_score": 0.5,
         "ranking_score_is_probability": False, "validation_status": "unvalidated", "uncertainty_statement": T("fixture")}
    c.update(over)
    return c


def unknown_report(n_candidates: int = 2) -> dict:
    r = _base("unknown_investigation", "unknown")
    r["candidate_diagnoses"] = [candidate(i + 1) for i in range(n_candidates)]
    if n_candidates:
        r["differential_diagnosis"]["rows"] = [{"finding": "crop_compatibility", "per_candidate": {
            c["candidate_id"]: {"assessment": "supports", "note": T("fixture"), "source_ids": ["SRC-FIXTURE-1"]}
            for c in r["candidate_diagnoses"]}}]
    else:
        r["insufficient_evidence_statement"] = T("Insufficient evidence to rank a reliable disease candidate. Additional observations "
                                                 "or laboratory testing are recommended.")
    return r


def hindi(r: dict) -> dict:
    r = copy.deepcopy(r)
    r["language"] = "hi"
    for w in r["safety_warnings"]:
        w["hi"] = "परीक्षण चेतावनी"
    r["disclaimer"]["hi"] = "परीक्षण अस्वीकरण"
    r["review_status"]["translation_review"]["hi"] = "pending_fluent_speaker_review"
    return r


def rec(**over) -> dict:
    x = {"recommendation_id": "REC-9", "category": "disease_management", "subcategory": "cultural", "text": T("fixture"),
         "consequential": False, "source_ids": [], "applies_to": {"crop": "Tomato", "disease_id": None, "region": None},
         "geographic_applicability": "general", "dose": None, "expert_review_status": "pending"}
    x.update(over)
    return x
