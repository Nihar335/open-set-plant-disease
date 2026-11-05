"""Report engine (Milestone 3): generation, validation, traceability, safety rules, coverage, reproducibility, Hindi.

All inputs are SYNTHETIC (stage10/tests/report_fixtures.py). No image is opened; no final-test or held-out data is used.
"""
from __future__ import annotations

import copy
import json
import random

import pytest

from stage10.guards import ForbiddenImageError
from stage10.knowledge_base.kb import KnowledgeBase, load_kb
from stage10.osr.osr_adapter import MSG_KNOWN, MSG_UNKNOWN, NOTE
from stage10.report_engine import policy
from stage10.report_engine.checks import UNFAMILIAR, engine_errors
from stage10.report_engine.engine import ReportEngineError, build_report, messages
from stage10.report_engine.inputs import ImageInfo, ImageObservation, Region, ReportInputError, image_info_from_file, validate_request
from stage10.schemas.validate import validate_report
from stage10.tests import report_fixtures as rf

KB = load_kb()
SECTIONS = ("recommended_tests", "treatment_guidance", "nutrition_guidance", "soil_guidance", "immediate_actions")


def gen(req, **kw):
    return build_report(req, KB, **kw)


def recs(r):
    return [x for s in SECTIONS for x in r[s]]


def texts(r):
    return json.dumps(r, ensure_ascii=False)


# ---------------------------------------------------------------- known-disease reports
def test_known_disease_report_structure():
    res = gen(rf.known_eb())
    r = res.report
    validate_report(r, frozen_threshold=rf.frozen_threshold())
    assert r["report_type"] == "known_disease" and r["osr_status"]["decision"] == "known"
    c1 = r["candidate_diagnoses"][0]
    assert c1["kb_disease_id"] == "COND-TOMATO-EARLY-BLIGHT" and c1["ranking_method"] == "model_class_evidence"
    assert c1["ranking_score"] is None and c1["status"] == "candidate" and c1["crop_compatible"]
    assert c1["scientific_cause"] == {"value": "Alternaria solani", "status": "unverified", "source_ids": ["SRC-TNAU-TOM-EARLYBLIGHT"]}
    assert r["plant_identity"]["taxonomy"]["family"]["value"] == "Solanaceae" and not r["plant_identity"]["species_level_claimed"]
    st = " ".join(s["en"] for s in r["uncertainty"]["statements"])
    assert "Why Early blight is listed" in st and "cannot establish" in st and "not a confirmed diagnosis" in st
    assert r["treatment_guidance"] and all(x["text"]["en"].startswith("If ") for x in r["treatment_guidance"])
    assert r["generator"]["knowledge_base_version"] == KB.version and r["generator"]["llm_used"] is False
    assert r["model_prediction"]["model_versions"] == rf.frozen_model_versions()


def test_known_report_separates_observed_from_source_typical_symptoms():
    r = gen(rf.known_eb(reported_features={"FEAT-YELLOW-HALO": True, "FEAT-FRUIT-LESIONS": True})).report
    obs = [s for s in r["visible_symptoms"] if s["origin"] == "user_report"]
    typ = [s for s in r["visible_symptoms"] if s["origin"] == "knowledge_base_typical"]
    assert {s["evidence_status"] for s in obs} == {"observed"}
    assert typ and all(s["evidence_status"] == "unverified" and s["location"]["bbox"] is None for s in typ)
    assert any("concentric rings" in s["label"]["en"] for s in typ)
    typ_ids = {s["symptom_id"] for s in typ}
    assert not typ_ids & {x for c in r["candidate_diagnoses"] for x in c["supporting_symptoms"]}


def test_model_output_is_preserved_unchanged():
    req = rf.known_eb()
    r = gen(req).report
    assert r["model_prediction"] == req.model_assessment["model_prediction"]
    assert {k: v for k, v in r["osr_status"].items()} == req.model_assessment["osr_status"]


def test_healthy_and_stub_known_classes_give_no_disease_claims():
    r = gen(rf.known_eb(reported_features={}, model_assessment=rf.synthetic_assessment("Tomato___healthy", True))).report
    assert r["candidate_diagnoses"] == [] and "does not establish that the plant is free" in texts(r)
    assert r["insufficient_evidence_statement"]["en"].startswith("Insufficient evidence")
    # a model class whose KB record is an unsourced stub gives no disease claims (synthetic stub: all real records are sourced now)
    import copy
    from stage10.knowledge_base.kb import KnowledgeBase, load_raw
    raw = copy.deepcopy(load_raw())
    stub = next(c for c in raw["conditions"] if c["condition_id"] == "COND-TOMATO-LEAF-MOLD")
    stub.update(curation_status="stub_unsourced", features=[], favourable_conditions=[], management_ids=[], diagnostic_ids=[])
    raw["management"] = [m for m in raw["management"] if "COND-TOMATO-LEAF-MOLD" not in m["applies_to_conditions"]]
    req = rf.known_eb(reported_features={}, model_assessment=rf.synthetic_assessment("Tomato___Leaf_Mold", True))
    r = build_report(req, KnowledgeBase(raw)).report
    assert r["candidate_diagnoses"] == [] and "not yet source-verified" in texts(r) and not r["treatment_guidance"]
    # with the real (now sourced) record the same model class gives a sourced candidate with management
    r = gen(req).report
    assert r["candidate_diagnoses"][0]["kb_disease_id"] == "COND-TOMATO-LEAF-MOLD" and r["treatment_guidance"]


# ---------------------------------------------------------------- unknown-investigation reports
def test_unknown_investigation_report():
    r = gen(rf.unknown_tomato()).report
    assert r["report_type"] == "unknown_investigation"
    assert r["osr_status"]["message"]["en"].startswith(UNFAMILIAR)
    assert any(w["en"].startswith("An 'unfamiliar to this model' result does not mean a new disease") for w in r["safety_warnings"])
    names = [c["kb_disease_id"] for c in r["candidate_diagnoses"]]
    # SC-M5-02: the model's closest class (photo not accepted) is combined with the symptom match
    assert names[0] == "COND-TOMATO-EARLY-BLIGHT" and "COND-TOMATO-TARGET-SPOT" in names
    for c in r["candidate_diagnoses"]:
        assert c["ranking_method"] in ("hybrid", "kb_symptom_match") and isinstance(c["ranking_score"], float)
        assert c["ranking_score_is_probability"] is False and c["validation_status"] == "unvalidated"
        assert c["recommended_confirmatory_tests"]
    assert "new disease has been discovered" not in texts(r).replace("does not mean a new disease has been discovered", "")


def test_unknown_report_rows_and_questions():
    r = gen(rf.unknown_tomato(reported_features={"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True})).report
    findings = [row["finding"] for row in r["differential_diagnosis"]["rows"]]
    assert findings[:5] == ["crop_compatibility", "observed_symptoms_explained", "missing_expected_signs",
                            "environmental_compatibility", "distinguishing_test"]
    assert "other" in findings and "Unanswered questions" in texts(r)


def test_non_infectious_candidate_stays_a_hypothesis():
    r = gen(rf.unknown_tomato(reported_features={"FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": True,
                                                 "FEAT-OLDER-LOWER-LEAVES-FIRST": True})).report
    # the nutrient problem the user's symptoms point to stays listed next to the model's closest disease (SC-M5-02)
    assert "nutrient_deficiency" in [c["cause_category"] for c in r["candidate_diagnoses"]]
    (n,) = r["nutrition_guidance"]
    assert n["category"] == "suspected_nutrient_issue" and n["text"]["en"].startswith("Suspected only")
    assert n["verification_test"] and n["depends_on"] and n["safety_notes"] and n["subcategory"] == "laboratory_test"
    assert "KVK" in n["text"]["en"] and "names no corrective product or rate" in n["text"]["en"]
    assert any("only a hypothesis" in w["en"] for w in r["safety_warnings"])
    assert all(x["category"] != "confirmed_nutrient_deficiency" for x in recs(r))


@pytest.mark.parametrize("feats", [{"FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True},
                                   {"FEAT-MARGIN-SCORCH": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}])
def test_named_nutrient_products_are_withheld(feats):
    """Owner decision 2026-10-14: no named corrective fertiliser/nutrient products in reports."""
    res = gen(rf.unknown_tomato(reported_features=feats))
    t = texts(res.report).lower()
    for prod in ("sulphate", "sulfate", "dolomite", "muriate", "foliar spray", "magnesium sulphate", "potassium sulphate"):
        assert prod not in t
    w = [x for x in res.evidence_trace["withheld_management"] if x["category"] == "nutrient"]
    assert w and all("owner decision" in x["reason"] for x in w)
    assert res.report["nutrition_guidance"] and all(x["source_ids"] == [] and not x["consequential"] for x in res.report["nutrition_guidance"])


def test_image_observations_are_inferred_and_used():
    obs = (ImageObservation("FEAT-CONCENTRIC-RINGS", "leaf_upper", (0.1, 0.1, 0.5, 0.5), "moderate", 0.12, "synthetic mask",
                            ("slight blur",), "synthetic-test"),)
    r = gen(rf.unknown_tomato(image_observations=obs, reported_features={"FEAT-YELLOW-HALO": True})).report
    s1 = r["visible_symptoms"][0]
    assert s1["origin"] == "image_analysis" and s1["evidence_status"] == "inferred" and s1["image_quality_limitations"] == ["slight blur"]
    assert any("SYM-1" in c["supporting_symptoms"] for c in r["candidate_diagnoses"])


def test_user_report_overrides_conflicting_image_observation():
    obs = (ImageObservation("FEAT-CONCENTRIC-RINGS"),)
    r = gen(rf.unknown_tomato(image_observations=obs, reported_features={"FEAT-CONCENTRIC-RINGS": False, "FEAT-YELLOW-HALO": True,
                                                                         "FEAT-OLDER-LOWER-LEAVES-FIRST": True})).report
    assert "disagree" in texts(r)


# ---------------------------------------------------------------- contradictions and abstention
def test_contradictory_symptoms_are_reported():
    feats = {"FEAT-CONCENTRIC-RINGS": False, "FEAT-YELLOW-HALO": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}
    known = gen(rf.known_eb(reported_features=feats)).report
    c1 = known["candidate_diagnoses"][0]
    assert any(x["en"].startswith("Conflicting") and "concentric rings" in x["en"] for x in c1["conflicting_or_missing"])
    assert known["uncertainty"]["level"] == "low"
    unk = gen(rf.unknown_tomato(reported_features=feats)).report
    assert "COND-TOMATO-EARLY-BLIGHT" not in [c["kb_disease_id"] for c in unk["candidate_diagnoses"]]


@pytest.mark.parametrize("feats", [{}, {"FEAT-YELLOW-HALO": True}, {"FEAT-INTERVEINAL-STRIPING": True, "FEAT-DARK-SPOTS-WOOD-SECTION": True}])
def test_insufficient_evidence_abstains(feats):
    # without a model result (e.g. the photo model could not run) weak or unrelated symptoms still give no candidate
    r = gen(rf.unknown_tomato(reported_features=feats, model_assessment=None)).report
    assert r["candidate_diagnoses"] == [] and r["insufficient_evidence_statement"]["en"].startswith("Insufficient evidence")
    assert not r["treatment_guidance"] and not r["nutrition_guidance"] and r["uncertainty"]["level"] == "not_assessable"


# ---------------------------------------------------------------- plant identity
def test_missing_plant_identity_makes_everything_preliminary():
    feats = {"FEAT-BLACK-SPECKS-IN-LESION": True, "FEAT-FRUIT-MUMMY": True, "FEAT-FRUIT-LESIONS": True}   # grape AND apple black rot
    r = gen(rf.request(reported_features=feats)).report
    pi = r["plant_identity"]
    assert pi["identification_level"] == "not_assessable" and pi["crop_confirmed_by_user"] is None
    assert all(v["status"] == "not_assessable" for v in pi["taxonomy"].values())
    assert len({c["affected_crop"] for c in r["candidate_diagnoses"]}) > 1
    assert all(not c["crop_compatible"] and c["crop_mismatch_note"]["en"].startswith("Preliminary") for c in r["candidate_diagnoses"])
    assert not r["treatment_guidance"] and "different host plants" in texts(r)
    assert any(x["text"]["en"].startswith("Confirm the name of the crop") for x in r["immediate_actions"])


def test_unsure_plant_is_not_treated_as_confirmed():
    r = gen(rf.unknown_tomato(plant_confirmation="unsure")).report
    assert r["plant_identity"]["crop_confirmed_by_user"] is None and "were not sure" in texts(r)
    assert all(not c["crop_compatible"] for c in r["candidate_diagnoses"]) and not r["treatment_guidance"]


def test_known_class_without_plant_confirmation_is_preliminary():
    r = gen(rf.known_eb(plant_name=None, plant_confirmation="not_provided")).report
    assert r["report_type"] == "known_disease" and r["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] is None
    assert not r["candidate_diagnoses"][0]["crop_compatible"] and not r["treatment_guidance"]


def test_incompatible_plant_identity():
    r = gen(rf.known_eb(plant_name="Grape")).report
    assert r["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] is False
    assert "not applicable" in r["plant_identity"]["model_supporting_evidence"]["note"]["en"]
    assert all(c["kb_disease_id"] != "COND-TOMATO-EARLY-BLIGHT" for c in r["candidate_diagnoses"])
    assert all(c["crop_compatible"] for c in r["candidate_diagnoses"])


# ---------------------------------------------------------------- candidate limits
def test_candidate_limit_and_scores_over_random_inputs():
    rng = random.Random(0)
    feats = [f for f, x in KB.features.items()]
    plants = [None, "Tomato", "Grape", "Maize", "Apple", "Mango", "Rice"]
    for _ in range(60):
        f = {x: rng.random() < 0.7 for x in rng.sample(feats, rng.randint(0, 7))}
        p = rng.choice(plants)
        a = rf.synthetic_assessment("Tomato___Early_blight", rng.random() < 0.5)
        r = gen(rf.request(plant_name=p, plant_confirmation="confirmed" if p else "not_provided", reported_features=f,
                           model_assessment=a)).report
        assert len(r["candidate_diagnoses"]) <= 3
        for c in r["candidate_diagnoses"]:
            assert c["ranking_score_is_probability"] is False and c["validation_status"] == "unvalidated"
            assert c["ranking_score"] is None or c["ranking_method"] in ("kb_symptom_match", "hybrid")
        if not r["candidate_diagnoses"] and r["report_type"] != "known_disease":
            assert "insufficient_evidence_statement" in r


# ---------------------------------------------------------------- treatment and regional safety
def test_no_chemical_or_biological_advice_and_withheld_listed():
    r = gen(rf.unknown_tomato()).report
    res = gen(rf.unknown_tomato())
    assert not [x for x in recs(r) if x["subcategory"] in ("chemical", "biological", "fertilizer")]
    w = {x["management_id"]: x["reason"] for x in res.evidence_trace["withheld_management"]}
    assert w["MGMT-TARGETSPOT-CHEMICAL"].startswith("foreign source")
    assert "MGMT-EARLYBLIGHT-CHEMICAL" in w
    for prod in ("mancozeb", "chlorothalonil", "copper oxychloride", "azoxystrobin", "iprodione"):
        assert prod not in texts(r).lower()
    assert any(x["text"]["en"].startswith("If treatment is needed after confirmation") for x in r["immediate_actions"])


def test_foreign_guidance_is_labelled_not_indian():
    r = gen(rf.unknown_tomato()).report
    foreign = [x for x in recs(r) if x["geographic_applicability"] == "international_not_verified_for_India"]
    assert foreign and all("not verified for India" in x["text"]["en"] for x in foreign if x["category"] != "diagnostic_test")
    assert all(any("not verified for India" in n["en"] for n in x["risk_notes"]) for x in foreign)
    assert any("outside India" in w["en"] for w in r["safety_warnings"])


def test_region_notes():
    tn = gen(rf.known_eb(region=Region("tamil nadu"))).report
    mh = gen(rf.known_eb(region=Region("Maharashtra", "Pune"))).report
    assert tn["region"] == {"country": "IN", "state": "Tamil Nadu", "district": None}
    eb = lambda r: [x for x in r["treatment_guidance"] if x["applies_to"]["disease_id"] == "COND-TOMATO-EARLY-BLIGHT"]  # noqa: E731
    assert all("risk_notes" not in x for x in eb(tn))
    assert all(any("Maharashtra has not been verified" in n["en"] for n in x["risk_notes"]) for x in eb(mh))


def test_chemical_gate_rules():
    m_foreign = KB.management["MGMT-TARGETSPOT-CHEMICAL"]
    m_tnau = KB.management["MGMT-EARLYBLIGHT-CHEMICAL"]
    full = {"management_id": "MGMT-EARLYBLIGHT-CHEMICAL", "crop": "Tomato", "use": "early blight", "registration_source_id": "SRC-GOV-X",
            "label_source_id": "SRC-TNAU-TOM-EARLYBLIGHT", "verified_on": "2026-10-14", "expert_reviewed": True}
    assert policy.chemical_decision(m_foreign, KB, [dict(full, management_id="MGMT-TARGETSPOT-CHEMICAL")])[0] is False
    assert policy.chemical_decision(m_tnau, KB, [])[0] is False
    assert policy.chemical_decision(m_tnau, KB, [dict(full, expert_reviewed=False)])[0] is False
    assert policy.chemical_decision(m_tnau, KB, [full])[0] is False            # SRC-GOV-X does not exist
    raw = copy.deepcopy(KB.raw)
    raw["sources"].append(dict(copy.deepcopy(KB.sources["SRC-TNAU-TOM-EARLYBLIGHT"]), source_id="SRC-GOV-X", source_type="government_india"))
    kb2 = KnowledgeBase(raw)
    assert policy.chemical_decision(kb2.management["MGMT-EARLYBLIGHT-CHEMICAL"], kb2, [full])[0] is True
    with pytest.raises(ReportEngineError):            # even then this milestone refuses to emit actionable chemical advice
        build_report(rf.known_eb(), kb2, verifications=[full])


def test_registration_verification_file_is_empty():
    assert policy.load_verifications() == []


# ---------------------------------------------------------------- coverage without image capability
@pytest.mark.parametrize("plant,feats,cond", [
    ("Mango", {"FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True, "FEAT-FRUIT-LESIONS": True}, "COND-MANGO-ANTHRACNOSE"),
    ("paddy", {"FEAT-SPINDLE-GREY-CENTRE-BROWN-MARGIN": True, "FEAT-NODE-OR-NECK-BLACK-LESIONS": True}, "COND-RICE-BLAST")])
def test_knowledge_coverage_without_image_capability(plant, feats, cond):
    for a in (None, rf.synthetic_assessment("Tomato___Early_blight", True)):
        r = gen(rf.request(plant_name=plant, plant_confirmation="confirmed", reported_features=feats, model_assessment=a)).report
        assert r["report_type"] == "out_of_model_scope" and r["plant_identity"]["in_model_scope"] is False
        assert r["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] is None
        assert [c["kb_disease_id"] for c in r["candidate_diagnoses"]] == [cond]
        assert "not_covered_by_frozen_model" in texts(r)
        assert any(w["en"].startswith("The image model was never trained on this plant") for w in r["safety_warnings"])
        assert r["plant_identity"]["taxonomy"]["genus"]["source_ids"]
        if a:
            assert "not applicable to this plant" in texts(r)


def test_plant_outside_knowledge_base():
    r = gen(rf.request(plant_name="Banana", plant_confirmation="confirmed",
                       reported_features={"FEAT-YELLOW-HALO": True, "FEAT-CONCENTRIC-RINGS": True})).report
    assert r["report_type"] == "out_of_model_scope" and r["candidate_diagnoses"] == []
    assert all(v["value"] is None for v in r["plant_identity"]["taxonomy"].values())
    assert r["insufficient_evidence_statement"]["en"].startswith("This plant is not in the AgroVaidya knowledge base")
    assert not r["treatment_guidance"] and r["evidence_sources"] == []
    assert any(x["subcategory"] == "consult_expert" for x in r["immediate_actions"])


def test_failed_quality_gate():
    r = gen(rf.known_eb(image=ImageInfo("ab" * 32, 100, 100, "JPEG", ("shorter side 100 px is below 224 px",)))).report
    assert r["report_type"] == "insufficient_image" and not r["candidate_diagnoses"] and not r["image_metadata"]["quality_gate"]["passed"]
    assert any(x["subcategory"] == "photography" for x in r["immediate_actions"])


# ---------------------------------------------------------------- traceability
def test_every_source_is_traced_to_verbatim_passages():
    res = gen(rf.unknown_tomato(region=Region("Karnataka")))
    r, t = res.report, res.evidence_trace
    assert engine_errors(r, t, KB, validate_request(rf.unknown_tomato(region=Region("Karnataka")), KB)) == []
    traced = {e["source_id"] for x in t["entries"] for e in x["evidence"]}
    assert {s["source_id"] for s in r["evidence_sources"]} <= traced | {s for w in t["withheld_management"] for s in w["source_ids"]}
    for s in r["evidence_sources"]:
        assert s["claims_supported"] and s["url_or_doi"].startswith("https://")
    for x in recs(r):
        if x["source_ids"]:
            assert any(e["report_path"] == next(f"{sec}/{x['recommendation_id']}" for sec in SECTIONS if x in r[sec]) for e in t["entries"])


def _tamper(fn):
    req = rf.unknown_tomato()
    res = gen(req)
    r, t = copy.deepcopy(res.report), copy.deepcopy(res.evidence_trace)
    fn(r, t)
    return engine_errors(r, t, KB, validate_request(req, KB))


def test_broken_or_missing_sources_are_detected():
    assert any("not in evidence_sources" in e for e in _tamper(lambda r, t: r["evidence_sources"].pop(0)))
    assert any("not found verbatim" in e or "not a knowledge-base" in e
               for e in _tamper(lambda r, t: t["entries"][0]["evidence"][0].update(passage="an invented passage text")))
    assert any("unknown source" in e for e in _tamper(lambda r, t: t["entries"][0]["evidence"][0].update(source_id="SRC-NOPE")))
    assert any("no traced claim" in e for e in _tamper(lambda r, t: t.update(entries=[], withheld_management=[])))


def test_engine_checks_catch_unsafe_edits():
    def add_dose(r, t):
        r["treatment_guidance"][0]["text"]["en"] += " Spray 2 g/l."
    assert any("dose-like" in e for e in _tamper(add_dose))
    assert any("overclaiming" in e for e in _tamper(lambda r, t: r["uncertainty"]["statements"].append({"en": "This will cure the plant."})))
    assert any("not an observation" in e for e in _tamper(lambda r, t: r["visible_symptoms"][0].update(origin="knowledge_base_typical", evidence_status="unverified", visual_evidence=None)))
    assert any("frozen adapter" in e for e in _tamper(lambda r, t: r["model_prediction"].update(predicted_class="Tomato___healthy")))
    assert any("established" in e for e in _tamper(lambda r, t: r["osr_status"]["message"].update(en="Unknown disease detected.")))
    assert any("chemical" in e for e in _tamper(lambda r, t: r["treatment_guidance"][0].update(subcategory="chemical", safety_notes=[{"en": "x"}], consequential=True)))


def test_kb_used_by_default_is_integrity_checked(monkeypatch):
    import stage10.report_engine.engine as eng
    called = []
    monkeypatch.setattr(eng, "load_kb", lambda: called.append(1) or KB)
    build_report(rf.known_eb())
    assert called


# ---------------------------------------------------------------- inputs
@pytest.mark.parametrize("kw", [
    {"reported_features": {"FEAT-NOT-REAL": True}}, {"reported_features": {"FEAT-YELLOW-HALO": "yes"}},
    {"answers": {"Q-NOPE": "x"}}, {"region": Region("Atlantis")}, {"region": Region(None, "Pune")},
    {"region": Region("Karnataka", "12.9716, 77.5946")}, {"language": "fr"}, {"plant_confirmation": "maybe"},
    {"plant_name": "<script>alert(1)</script>"}, {"generated_at": "yesterday"}, {"generated_at": "2026-10-14T12:00:00"},
    {"image": ImageInfo("xyz", 10, 10, "JPEG")}, {"image": ImageInfo("ab" * 32, 10, 10, "GIF")},
    {"image_observations": (ImageObservation("FEAT-ENV-COOL-WET"),)},
    {"image_observations": (ImageObservation("FEAT-YELLOW-HALO", affected_area_fraction=0.2),)},
])
def test_invalid_inputs_rejected(kw):
    with pytest.raises(ReportInputError):
        gen(rf.known_eb(**kw))


def test_tampered_model_assessment_rejected():
    for mutate in (lambda a: a["osr_status"].update(frozen_threshold=-4.0),
                   lambda a: a["osr_status"].update(decision="unknown"),
                   lambda a: a["model_prediction"].update(predicted_crop="Apple"),
                   lambda a: a["model_prediction"]["model_versions"].update(linear_head_sha256="0" * 64)):
        a = rf.synthetic_assessment("Tomato___Early_blight", True)
        mutate(a)
        with pytest.raises(ReportEngineError):
            gen(rf.known_eb(model_assessment=a))


def test_final_test_image_refused():
    import csv
    from src.utils.config import resolve
    with open(resolve("data/splits/test.csv"), newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))["image_path"]
    with pytest.raises(ForbiddenImageError):
        image_info_from_file(resolve(first))


# ---------------------------------------------------------------- reproducibility
def test_identical_inputs_give_identical_reports():
    a, b = gen(rf.unknown_tomato()), gen(rf.unknown_tomato())
    assert a.to_json() == b.to_json() and a.evidence_trace == b.evidence_trace
    c = gen(rf.unknown_tomato(generated_at="2026-10-15T08:00:00+00:00"))
    assert c.report["report_id"] != a.report["report_id"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("report_id", "created_at")}  # noqa: E731
    assert strip(c.report) == strip(a.report)
    d = gen(rf.unknown_tomato(reported_features={**rf.EB_FEATURES, "FEAT-ENV-WARM-HUMID": True}))
    assert d.report["report_id"] != a.report["report_id"]


# ---------------------------------------------------------------- Hindi
def test_hindi_report_preserves_safety_labels_and_review_status():
    en = gen(rf.unknown_tomato()).report
    hi = gen(rf.unknown_tomato(language="hi")).report
    assert hi["review_status"]["translation_review"] == {"en": "not_applicable", "hi": "pending_fluent_speaker_review"}
    for w in hi["safety_warnings"] + [hi["disclaimer"], hi["osr_status"]["message"], hi["osr_status"]["interpretation_note"]]:
        assert w.get("hi")
    for c, e in zip(hi["candidate_diagnoses"], en["candidate_diagnoses"]):
        assert c["uncertainty_statement"]["hi"] and c["uncertainty_statement"]["en"] == e["uncertainty_statement"]["en"]
    assert all(x["text"].get("hi") for x in hi["immediate_actions"])
    assert [w["en"] for w in hi["safety_warnings"]] == [w["en"] for w in en["safety_warnings"]] + [messages()["messages"]["translation_note"]["en"]]
    assert hi["osr_status"]["message"]["hi"].startswith("इस मॉडल के लिए संभवतः अपरिचित")
    assert {k: v for k, v in hi["model_prediction"].items()} == en["model_prediction"]


def test_message_catalogue_complete_and_consistent():
    cat = messages()
    assert cat["meta"]["translation_status"]["hi"] == "pending_fluent_speaker_review"
    for k, m in cat["messages"].items():
        assert m.get("en") and m.get("hi"), k
    assert cat["messages"]["osr_unknown"]["en"] == MSG_UNKNOWN["en"] and cat["messages"]["osr_unknown"]["en"].startswith(UNFAMILIAR)
    assert cat["messages"]["osr_known"]["en"] == MSG_KNOWN["en"] and cat["messages"]["osr_note"]["en"] == NOTE["en"]
    from stage10.knowledge_base.kb import DOSE
    from stage10.report_engine.checks import BANNED
    for m in cat["messages"].values():
        assert not DOSE.search(m["en"]) and not DOSE.search(m["hi"]) and not BANNED.search(m["en"])


# ---------------------------------------------------------------- modality provenance (D-5) and language independence (D-6)
def test_input_channels_recorded():
    req = rf.unknown_tomato(reported_features={"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True},
                            reported_channels={"FEAT-CONCENTRIC-RINGS": "voice_transcript_confirmed"},
                            answers={"Q-SPOT-PATTERN": "rings"} if "Q-SPOT-PATTERN" in KB.questions else {},
                            image_observations=(ImageObservation("FEAT-YELLOW-HALO"),))
    r = gen(req).report
    ch = {(s["label"]["en"], s["origin"]): s["input_channel"] for s in r["visible_symptoms"]}
    assert ch[(KB.features["FEAT-CONCENTRIC-RINGS"]["label"], "user_report")] == "voice_transcript_confirmed"
    assert ch[(KB.features["FEAT-YELLOW-HALO"]["label"], "user_report")] == "symptom_checklist"
    assert ch[(KB.features["FEAT-YELLOW-HALO"]["label"], "image_analysis")] == "image_analysis"


def test_guided_answers_are_marked_as_answers():
    q = next(iter(KB.questions.values()))
    opt = next(o for o in q["options"] if any(o["sets"].values()))
    r = gen(rf.unknown_tomato(reported_features={}, answers={q["question_id"]: opt["option_id"]})).report
    assert r["visible_symptoms"] and {s["input_channel"] for s in r["visible_symptoms"]} == {"guided_question_answer"}


@pytest.mark.parametrize("ch", ["voice_transcript_unconfirmed", "typed_text", "image_analysis", "llm"])
def test_unconfirmed_or_invalid_channels_rejected(ch):
    with pytest.raises(ReportInputError):
        gen(rf.unknown_tomato(reported_channels={"FEAT-CONCENTRIC-RINGS": ch}))
    with pytest.raises(ReportInputError):
        gen(rf.unknown_tomato(reported_channels={"FEAT-WILTING": "typed_text_confirmed"}))      # feature not reported


@pytest.mark.parametrize("req", [rf.known_eb(), rf.unknown_tomato(), rf.request(plant_name="Banana", plant_confirmation="confirmed"),
                                 rf.unknown_tomato(reported_features={"FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": True,
                                                                      "FEAT-OLDER-LOWER-LEAVES-FIRST": True})])
def test_localize_from_keys_equals_direct_generation(req):
    import dataclasses
    from stage10.report_engine.localize import localize
    en = gen(req).report
    hi = gen(dataclasses.replace(req, language="hi")).report
    assert en["report_id"] == hi["report_id"]
    assert localize(en, "hi") == hi and localize(hi, "en") == en
    assert en["content_keys"] and all(ck["key"] in messages()["messages"] for ck in en["content_keys"])


def test_language_changes_no_diagnostic_field():
    en, hi = gen(rf.unknown_tomato()).report, gen(rf.unknown_tomato(language="hi")).report

    def strip(o):
        if isinstance(o, dict):
            if "en" in o and all(isinstance(v, str) for v in o.values()):
                return o["en"]
            return {k: strip(v) for k, v in o.items() if k not in ("language", "review_status", "safety_warnings", "content_keys")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    assert strip(en) == strip(hi)


def test_engine_checks_catch_key_and_withheld_tampering():
    assert any("differs from catalogue" in e for e in _tamper(lambda r, t: r["disclaimer"].update(en="Edited disclaimer.")))
    assert any("does not resolve" in e for e in _tamper(lambda r, t: r["content_keys"].append({"path": "/nope/0", "key": "disclaimer", "params": {}})))
    def leak(r, t):
        r["uncertainty"]["statements"].append({"en": KB.management[t["withheld_management"][0]["management_id"]]["summary"]})
    assert any("withheld management" in e for e in _tamper(leak))
