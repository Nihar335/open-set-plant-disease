"""Focused tests for report-schema defects found in Milestone 3 (written BEFORE the schema was changed).

D-1  The report had no field for the knowledge-base version, which Milestone 3 requires in every report.
     Fix: `generator.knowledge_base_version` (required, semantic version).
D-2  `source_record.source_type` could not express two source types that the Milestone-2 knowledge base actually uses
     (international university extension pages; taxonomic databases such as GBIF). Mapping them to "other" or
     "botanical_authority" would misdescribe the source. Fix: two enum values added; nothing else changes.

Both changes are additive except that D-1 makes a new field required. No existing constraint was relaxed.
See stage10/report_engine/README.md, "Schema findings".
"""
from __future__ import annotations

import copy

import pytest

from stage10.schemas.validate import structural_errors, validate_report
from stage10.tests import fixtures as fx


def test_d1_generator_requires_knowledge_base_version():
    r = fx.known_report()
    assert r["generator"]["knowledge_base_version"]
    validate_report(r, allow_fixture_sources=True)
    del r["generator"]["knowledge_base_version"]
    assert any("knowledge_base_version" in e for e in structural_errors(r))


@pytest.mark.parametrize("v", ["", "latest", "0.1", "v0.1.0"])
def test_d1_knowledge_base_version_must_be_semver(v):
    r = fx.known_report()
    r["generator"]["knowledge_base_version"] = v
    assert structural_errors(r)


@pytest.mark.parametrize("t", ["university_extension_international", "taxonomic_database"])
def test_d2_kb_source_types_are_representable(t):
    r = fx.known_report()
    r["evidence_sources"].append(dict(copy.deepcopy(fx.FIX_SRC), source_id="SRC-FIXTURE-2", source_type=t))
    validate_report(r, allow_fixture_sources=True)


def test_d2_unknown_source_type_still_rejected():
    r = fx.known_report()
    r["evidence_sources"][0]["source_type"] = "blog"
    assert structural_errors(r)


# ---------------------------------------------------------------------------------------------------------------------
# D-5  An observation did not record HOW it was obtained (checklist tap, guided answer, confirmed typed text, confirmed
#      voice transcript, image analysis). `origin: user_report` conflated them, which multimodal input (M5) cannot use.
#      Fix: required `input_channel`, consistent with `origin`; there is deliberately no value for an UNCONFIRMED
#      transcript or text interpretation, so such input cannot be represented as an observation.
# D-6  Fixed report texts carried no language-independent identity, so a report could not be re-rendered in another
#      language (or spoken) from keys. Fix: required report-level `content_keys` [{path, key, params}].
def test_d5_observation_requires_input_channel():
    r = fx.known_report()
    assert r["visible_symptoms"][0]["input_channel"] == "image_analysis"
    validate_report(r, allow_fixture_sources=True)
    del r["visible_symptoms"][0]["input_channel"]
    assert any("input_channel" in e for e in structural_errors(r))


@pytest.mark.parametrize("origin,channel,ok", [
    ("image_analysis", "image_analysis", True), ("image_analysis", "typed_text_confirmed", False),
    ("user_report", "voice_transcript_confirmed", True), ("user_report", "symptom_checklist", True),
    ("user_report", "guided_question_answer", True), ("user_report", "image_analysis", False),
    ("user_report", "voice_transcript_unconfirmed", False), ("knowledge_base_typical", "knowledge_base", True),
    ("knowledge_base_typical", "symptom_checklist", False)])
def test_d5_channel_consistent_with_origin(origin, channel, ok):
    r = fx.known_report()
    s = r["visible_symptoms"][0]
    s.update(origin=origin, input_channel=channel)
    if origin == "knowledge_base_typical":
        s.update(evidence_status="unverified", affected_area_fraction=None, measurement_method=None)
        s["location"]["bbox"] = None
    assert (structural_errors(r) == []) is ok


def test_d6_report_requires_content_keys():
    r = fx.known_report()
    assert r["content_keys"] == []
    validate_report(r, allow_fixture_sources=True)
    r["content_keys"] = [{"path": "/disclaimer", "key": "disclaimer", "params": {}}]
    validate_report(r, allow_fixture_sources=True)
    for bad in ({"path": "disclaimer", "key": "disclaimer", "params": {}}, {"path": "/disclaimer", "key": "Bad Key", "params": {}},
                {"path": "/disclaimer", "key": "disclaimer"}):
        r["content_keys"] = [bad]
        assert structural_errors(r)
    del r["content_keys"]
    assert any("content_keys" in e for e in structural_errors(r))
