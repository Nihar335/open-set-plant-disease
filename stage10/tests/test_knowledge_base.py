"""Knowledge-base integrity: real records must pass; every injected fault must be rejected."""
from __future__ import annotations

import copy
import re

import pytest

from stage10.knowledge_base.kb import DOSE, integrity_errors, load_kb, load_raw

RAW = load_raw()


def errs(mut):
    r = copy.deepcopy(RAW)
    mut(r)
    return integrity_errors(r)


def _cond(r, cid):
    return next(c for c in r["conditions"] if c["condition_id"] == cid)


def _mg(r, mid):
    return next(m for m in r["management"] if m["management_id"] == mid)


def test_real_knowledge_base_has_no_integrity_errors():
    assert integrity_errors(RAW) == []
    kb = load_kb()
    assert len(kb.sources) >= 20 and len(kb.sourced_conditions()) >= 15


FAULTS = {
    "altered_passage": lambda r: _cond(r, "COND-GRAPE-ESCA")["features"][0]["claim"]["evidence"][0].__setitem__("passage", "stripes start out bright blue in all cultivars"),
    "duplicate_condition_id": lambda r: r["conditions"].append(copy.deepcopy(r["conditions"][0])),
    "duplicate_source_id": lambda r: r["sources"].append(copy.deepcopy(r["sources"][0])),
    "broken_feature_ref": lambda r: _cond(r, "COND-MAIZE-GRAY-LEAF-SPOT")["features"][0].__setitem__("feature_id", "FEAT-DOES-NOT-EXIST"),
    "broken_plant_ref": lambda r: _cond(r, "COND-RICE-BLAST")["affected_plants"][0].__setitem__("plant_id", "PLANT-UNKNOWN"),
    "broken_lookalike_ref": lambda r: _cond(r, "COND-GRAPE-ESCA")["look_alikes"].append({"condition_id": "COND-NOPE", "basis": "shared_kb_features"}),
    "self_lookalike": lambda r: _cond(r, "COND-GRAPE-ESCA")["look_alikes"].append({"condition_id": "COND-GRAPE-ESCA", "basis": "shared_kb_features"}),
    "broken_diagnostic_ref": lambda r: _cond(r, "COND-GRAPE-ESCA")["diagnostic_ids"].append("DIAG-NOPE"),
    "unknown_evidence_source": lambda r: _cond(r, "COND-RICE-BLAST")["features"][0]["claim"]["evidence"][0].__setitem__("source_id", "SRC-NOT-REGISTERED"),
    "claim_without_evidence": lambda r: _cond(r, "COND-RICE-BLAST")["features"][0]["claim"].__setitem__("evidence", []),
    "placeholder_source_url": lambda r: r["sources"][0].__setitem__("url", "https://example.invalid/fake"),
    "test_fixture_source_type": lambda r: r["sources"][0].__setitem__("source_type", "test_fixture"),
    "non_https_source": lambda r: r["sources"][0].__setitem__("url", "http://agritech.tnau.ac.in/x"),
    "cache_hash_mismatch": lambda r: r["sources"][0]["verification"].__setitem__("bytes_sha256", "0" * 64),
    "dose_in_management_summary": lambda r: _mg(r, "MGMT-RUST-CHEMICAL").__setitem__("summary", "Spray kresoxim-methyl @ 1 ml/l at 35 DAS"),
    "dose_in_claim_text": lambda r: _mg(r, "MGMT-SEPTORIA-CHEMICAL")["claim"].__setitem__("text", "Spray 200 ml/ha"),
    "percent_concentration_in_summary": lambda r: _mg(r, "MGMT-MG-CORRECTION").__setitem__("summary", "Spray 2% MgSO4"),
    "chemical_without_safety_notes": lambda r: _mg(r, "MGMT-NLB-CHEMICAL").pop("safety_notes"),
    "chemical_not_consequential": lambda r: _mg(r, "MGMT-NLB-CHEMICAL").__setitem__("consequential", False),
    "nutrient_without_confirmation_requirement": lambda r: _mg(r, "MGMT-K-CORRECTION").__setitem__("requires_confirmation_first", False),
    "one_way_management_link": lambda r: _cond(r, "COND-MAIZE-COMMON-RUST")["management_ids"].remove("MGMT-RUST-CHEMICAL"),
    "question_without_not_sure": lambda r: r["questions"][0]["options"].pop(),
    "question_unknown_feature": lambda r: r["questions"][0]["options"][0]["sets"].__setitem__("FEAT-BOGUS", True),
    "stub_with_claims": lambda r: _cond(r, "COND-TOMATO-LEAF-MOLD").update(curation_status="stub_unsourced", review_status="draft",
                                                                          last_verified=None),
    "sourced_without_features": lambda r: _cond(r, "COND-RICE-BLAST").__setitem__("features", []),
    "model_class_missing": lambda r: r["model_classes"].pop(),
    "model_class_role_mismatch": lambda r: next(m for m in r["model_classes"] if m["class_name"] == "Tomato___Target_Spot").__setitem__("research_role", "frozen_model_known_class"),
    "unverified_review_value": lambda r: r["sources"][0].__setitem__("review_status", "trust_me"),
}


@pytest.mark.parametrize("name", sorted(FAULTS))
def test_injected_fault_is_rejected(name):
    assert errs(FAULTS[name]), f"fault '{name}' was not detected"


def test_consequential_recommendations_have_sources_and_safety():
    for m in RAW["management"]:
        if m["consequential"]:
            assert m["claim"]["evidence"], m["management_id"]
            assert m.get("safety_notes"), m["management_id"]
        if m["category"] == "nutrient":
            assert m["requires_confirmation_first"] is True
            assert any("test" in s.lower() for s in m["safety_notes"])


def test_no_dose_in_any_of_our_wording():
    from stage10.knowledge_base.kb import _claims
    for key in ("conditions", "diagnostics", "management", "plants"):
        for r in RAW[key]:
            for c in _claims(r):
                assert not DOSE.search(c["text"]), c["text"]
    for m in RAW["management"]:
        assert not DOSE.search(m["summary"]), m["summary"]
        assert m["dose_policy"] in ("no_dose_in_source", "withheld_by_policy")


def test_sources_are_real_verified_and_not_placeholders():
    for s in RAW["sources"]:
        assert s["url"].startswith("https://") and "example" not in s["url"]
        assert s["source_type"] != "test_fixture" and s["review_status"] == "source_verified"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["accessed_on"])
        assert s["verification"]["method"] == "fetched_and_read_full_text"


def test_non_indian_sources_are_flagged():
    src = {s["source_id"]: s for s in RAW["sources"]}
    for m in RAW["management"]:
        geos = {src[e["source_id"]]["geographic_relevance"] for e in m["claim"]["evidence"]}
        if geos == {"international_not_verified_for_India"}:
            assert m["geographic_applicability"] == "international_not_verified_for_India", m["management_id"]


def test_nothing_is_marked_expert_reviewed_yet():
    for key in ("sources", "plants", "conditions", "diagnostics", "management"):
        assert all(r["review_status"] != "expert_reviewed" for r in RAW[key])


def test_unsourced_diagnostics_are_flagged():
    for d in RAW["diagnostics"]:
        if not d["claims"]:
            assert d["curation_status"] == "needs_source" and d["review_status"] == "draft"


def test_synthetic_schema_fixtures_are_test_only():
    from stage10.tests import fixtures as fx
    assert fx.FIX_SRC["source_type"] == "test_fixture" and "example.invalid" in fx.FIX_SRC["url_or_doi"]
    kb_ids = {s["source_id"] for s in RAW["sources"]}
    assert fx.FIX_SRC["source_id"] not in kb_ids
