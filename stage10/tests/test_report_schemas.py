"""Stage-10 schema + validator tests: each safety rule must REJECT the violating object (not just accept good ones)."""
from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from stage10.schemas.validate import NAMES, SCHEMA_DIR, semantic_errors, structural_errors, validate_report
from stage10.tests import fixtures as fx

FROZEN = -5.123311662698785


def ok(r, **kw):
    validate_report(r, allow_fixture_sources=True, **kw)


def bad_struct(r):
    assert structural_errors(r), "expected a structural (schema) error"


def bad_sem(r, **kw):
    assert semantic_errors(r, allow_fixture_sources=True, **kw), "expected a semantic error"


@pytest.mark.parametrize("name", NAMES)
def test_schemas_are_valid_2020_12(name):
    Draft202012Validator.check_schema(json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8")))


def test_valid_fixtures_pass():
    ok(fx.known_report(), frozen_threshold=FROZEN)
    ok(fx.unknown_report(2))
    ok(fx.unknown_report(3))
    ok(fx.unknown_report(0))
    ok(fx.hindi(fx.unknown_report(1)))


def test_report_type_must_match_osr_decision_and_scope():
    r = fx.known_report(); r["osr_status"]["decision"] = "unknown"; bad_struct(r)
    r = fx.known_report(); r["plant_identity"]["in_model_scope"] = False; r["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] = None
    bad_struct(r)
    r = fx.unknown_report(1); r["osr_status"]["decision"] = "known"; bad_struct(r)


def test_at_most_three_candidates_and_never_padded_or_confirmed():
    bad_struct(fx.unknown_report(4))
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["status"] = "confirmed"; bad_struct(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["ranking_score_is_probability"] = True; bad_struct(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["probability"] = 0.8; bad_struct(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["validation_status"] = "validated"; bad_struct(r)   # needs evaluation_reference


def test_zero_candidates_requires_insufficient_evidence_statement():
    r = fx.unknown_report(0); del r["insufficient_evidence_statement"]; bad_struct(r)


def test_candidate_needs_source_and_crop_mismatch_note():
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["source_ids"] = []; bad_struct(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["crop_compatible"] = False; bad_struct(r)
    r["candidate_diagnoses"][0]["crop_mismatch_note"] = {"en": "crop differs"}; ok(r)


def test_symptom_provenance_rules():
    r = fx.known_report(); s = r["visible_symptoms"][0]
    s.update(origin="knowledge_base_typical", evidence_status="observed"); bad_struct(r)        # typical != observed
    s.update(evidence_status="unverified"); bad_struct(r)                                        # typical cannot carry bbox/area
    s["location"]["bbox"] = None; s["affected_area_fraction"] = None; s["measurement_method"] = None; bad_struct(r)  # channel != origin (D-5)
    s["input_channel"] = "knowledge_base"; ok(r)
    r = fx.known_report(); r["visible_symptoms"][0].update(origin="model_inference", evidence_status="observed"); bad_struct(r)
    r = fx.known_report(); r["visible_symptoms"][0]["measurement_method"] = None; bad_struct(r)   # area without method


def test_recommendation_safety_rules():
    def with_rec(x):
        r = fx.known_report(); r["treatment_guidance"] = [x]; return r
    bad_struct(with_rec(fx.rec(consequential=True)))                                   # consequential without source
    ok(with_rec(fx.rec(consequential=True, source_ids=["SRC-FIXTURE-1"])))
    bad_struct(with_rec(fx.rec(subcategory="chemical", consequential=True, source_ids=["SRC-FIXTURE-1"])))   # no safety notes
    bad_struct(with_rec(fx.rec(subcategory="chemical", consequential=False, source_ids=["SRC-FIXTURE-1"],
                               safety_notes=[{"en": "wear PPE"}])))                     # chemical must be consequential
    chem = fx.rec(subcategory="chemical", consequential=True, source_ids=["SRC-FIXTURE-1"], safety_notes=[{"en": "follow the label"}])
    ok(with_rec(chem))
    dosed = dict(chem, dose={"amount": "2", "unit": "g/L", "basis": "label", "source_id": "SRC-FIXTURE-1",
                             "official_label_or_recommendation": True})
    assert not structural_errors(with_rec(dosed))
    bad_sem(with_rec(dosed))                                                             # config forbids pesticide doses
    unofficial = copy.deepcopy(dosed); unofficial["dose"]["official_label_or_recommendation"] = False
    bad_struct(with_rec(unofficial))


def test_nutrient_suspected_vs_confirmed_and_test_descriptions():
    def with_nut(x):
        r = fx.known_report(); r["nutrition_guidance"] = [x]; return r
    bad_struct(with_nut(fx.rec(category="suspected_nutrient_issue", subcategory="nutrition")))           # needs verification_test
    ok(with_nut(fx.rec(category="suspected_nutrient_issue", subcategory="nutrition", verification_test={"en": "soil test"})))
    bad_struct(with_nut(fx.rec(category="confirmed_nutrient_deficiency", subcategory="nutrition")))      # needs confirmation evidence
    r = fx.known_report(); del r["recommended_tests"][0]["what_it_cannot_establish"]; bad_struct(r)


def test_identity_rules():
    r = fx.known_report(); r["plant_identity"]["species_level_claimed"] = True; bad_struct(r)          # no species value/source
    r = fx.known_report(); r["model_prediction"]["softmax_is_calibrated"] = True; bad_struct(r)
    r = fx.known_report(); r["generator"]["llm_used"] = True; bad_struct(r)                            # deterministic mode


def test_semantic_reference_integrity():
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["source_ids"] = ["SRC-MISSING"]; bad_sem(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["supporting_symptoms"] = ["SYM-99"]; bad_sem(r)
    r = fx.unknown_report(1); r["candidate_diagnoses"][0]["recommended_confirmatory_tests"] = ["REC-77"]; bad_sem(r)
    r = fx.unknown_report(2); r["differential_diagnosis"]["rows"][0]["per_candidate"]["CAND-9"] = \
        {"assessment": "supports", "note": {"en": "x"}, "source_ids": []}; bad_sem(r)


def test_fixture_sources_rejected_in_real_reports():
    assert semantic_errors(fx.known_report())   # allow_fixture_sources defaults to False


def test_hindi_requires_safety_translation_and_review_flag():
    r = fx.hindi(fx.known_report()); del r["safety_warnings"][0]["hi"]; bad_sem(r)
    r = fx.hindi(fx.known_report()); r["review_status"]["translation_review"]["hi"] = "not_applicable"; bad_sem(r)
    r = fx.hindi(fx.known_report()); del r["review_status"]["translation_review"]["hi"]; bad_sem(r)
    r = fx.known_report(); r["language"] = "ta"; bad_sem(r)                                            # not enabled yet


def test_frozen_threshold_must_be_unaltered():
    r = fx.known_report(); r["osr_status"]["frozen_threshold"] = -5.0
    bad_sem(r, frozen_threshold=FROZEN)
