"""User-provided plant-name workflow (SC-M4-03): name resolution, correction, routing, disabled automatic identification,
report provenance and API/schema validation.

A FAKE disease-model adapter returning synthetic assessments (report_fixtures) is used; no model is loaded and no image
is opened (image metadata is synthetic).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stage10.knowledge_base.kb import KnowledgeBase, load_kb
from stage10.report_engine.inputs import ImageInfo, ReportInputError
from stage10.schemas.validate import structural_errors, validator
from stage10.tests import report_fixtures as rf
from stage10.workflow import assessment as AS
from stage10.workflow import cli
from stage10.workflow import plant_name as PN
from stage10.workflow import voice as V

KB = load_kb()
ROOT = Path(__file__).resolve().parents[2]
SIX = {"PLANT-TOMATO", "PLANT-APPLE", "PLANT-GRAPE", "PLANT-MAIZE", "PLANT-PEPPER-BELL", "PLANT-POTATO"}
BAD_IMG = ImageInfo(sha256="cd" * 32, width=120, height=90, format="JPEG", quality_issues=("shorter side 90 px is below 224 px",))


class FakeOSR:
    def __init__(self, cls="Tomato___Early_blight", known=True):
        self.cls, self.known, self.calls = cls, known, 0

    def assess(self, image):
        self.calls += 1
        return rf.synthetic_assessment(self.cls, self.known)


def confirmed(text=None, *, select=None, unlisted=False, dont_know=False):
    s = PN.PlantNameSession(KB)
    if dont_know:
        s.dont_know()
    elif select:
        s.select(select)
    else:
        r = s.enter(text)
        if unlisted:
            assert r["status"] in ("unknown", "needs_clarification")
            s.keep_unlisted_name()
    return s.confirm()


def run(choice, osr=None, image=rf.IMG, **kw):
    return AS.assess_plant_health(choice, None, KB, osr_adapter=osr or FakeOSR(), image_info=image, generated_at=rf.TS, **kw)


def resolution_ok(r):
    assert structural_errors(r, "plant_name_resolution") == [], r


# ------------------------------------------------------------------------------------------- names and aliases
@pytest.mark.parametrize("text,pid,supported", [
    ("Tomato", "PLANT-TOMATO", True), ("Apple", "PLANT-APPLE", True), ("Grape", "PLANT-GRAPE", True),
    ("Maize", "PLANT-MAIZE", True), ("Bell pepper", "PLANT-PEPPER-BELL", True), ("Potato", "PLANT-POTATO", True),
    ("Mango", "PLANT-MANGO", False), ("Rice", "PLANT-RICE", False)])
def test_valid_supported_plant_names(text, pid, supported):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "resolved" and r["plant_id"] == pid and r["plant"]["disease_model_supported"] is supported
    assert ({p["plant_id"] for p in PN.supported_plant_options(KB) if p["disease_model_supported"]}) == SIX


@pytest.mark.parametrize("text,pid,kind", [
    ("  CORN  ", "PLANT-MAIZE", "common_name"), ("corn (maize)", "PLANT-MAIZE", "synonym"), ("Paddy", "PLANT-RICE", "common_name"),
    ("grapevine", "PLANT-GRAPE", "common_name"), ("Sweet   Pepper", "PLANT-PEPPER-BELL", "synonym"),
    ("capsicum", "PLANT-PEPPER-BELL", "common_name"), ("Lycopersicon esculentum", "PLANT-TOMATO", "synonym"),
    ("solanum TUBEROSUM", "PLANT-POTATO", "synonym"), ("Mangifera indica L.", "PLANT-MANGO", "scientific_name")])
def test_verified_aliases_and_harmless_normalisation(text, pid, kind):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "resolved" and r["plant_id"] == pid and r["alias_kind"] == kind


def test_every_alias_comes_from_the_knowledge_base_only():
    idx = PN.alias_index(KB)
    for alias, pids in idx.items():
        for pid in pids:
            p = KB.plants[pid]
            names = {n["name"] for n in p["common_names"]} | set(p["synonyms"]) | {p["scientific_name"], p["taxonomy"]["species"], pid}
            assert alias in {PN.normalise(n) for n in names}


@pytest.mark.parametrize("text", [None, "", "   ", "\t\n"])
def test_empty_plant_name(text):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "empty" and r["plant_id"] is None and r["requires_user_choice"]
    s = PN.PlantNameSession(KB)
    s.enter(text)
    with pytest.raises(PN.PlantNameError):
        s.confirm()


@pytest.mark.parametrize("text", ["tomato1", "<script>", "x" * 61, "tomato; drop", 42])
def test_invalid_plant_name(text):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "invalid" and r["plant_id"] is None


@pytest.mark.parametrize("text,expect", [("pepper", {"PLANT-PEPPER-BELL"}), ("tomatoe", {"PLANT-TOMATO"}), ("potatos", {"PLANT-POTATO"})])
def test_partial_or_misspelt_names_are_only_suggested_never_mapped(text, expect):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "needs_clarification" and r["plant_id"] is None
    assert expect <= {p["plant_id"] for p in r["suggestions"]}
    s = PN.PlantNameSession(KB)
    s.enter(text)
    with pytest.raises(PN.PlantNameError):
        s.confirm()                                   # a suggestion is never applied without the user's choice


def test_ambiguous_name_requires_a_choice():
    raw = copy.deepcopy(KB.raw)
    for p in raw["plants"]:
        if p["plant_id"] in ("PLANT-MANGO", "PLANT-POTATO"):
            p["synonyms"].append("aam")               # synthetic: one local name shared by two records
    kb2 = KnowledgeBase(raw)
    r = PN.resolve_plant_name("Aam", kb2)
    resolution_ok(r)
    assert r["status"] == "ambiguous" and r["plant_id"] is None
    assert {p["plant_id"] for p in r["suggestions"]} == {"PLANT-MANGO", "PLANT-POTATO"}
    s = PN.PlantNameSession(kb2)
    s.enter("aam")
    with pytest.raises(PN.PlantNameError):
        s.confirm()
    with pytest.raises(PN.PlantNameError):
        s.keep_unlisted_name()                         # an ambiguous verified name cannot be kept as "unlisted"
    s.select("PLANT-MANGO")
    assert s.confirm()["plant_id"] == "PLANT-MANGO"


@pytest.mark.parametrize("text", ["Coconut", "banana", "Neem tree"])
def test_unknown_names_get_clarification_not_an_identity(text):
    r = PN.resolve_plant_name(text, KB)
    resolution_ok(r)
    assert r["status"] == "unknown" and r["plant"] is None and r["suggestions"] == []
    c = confirmed(text, unlisted=True)
    assert c["outcome"] == "unlisted_name" and c["plant_id"] is None and c["plant_name"] == r["entered_text"]
    assert not c["disease_model_supported"] and not c["independently_verified"]


# ------------------------------------------------------------------------------------------- correction
def test_user_can_correct_the_selection_before_proceeding():
    s = PN.PlantNameSession(KB)
    s.enter("tomato")
    s.select("PLANT-POTATO")                           # change of mind before confirming
    assert s.confirm()["plant_id"] == "PLANT-POTATO"
    with pytest.raises(PN.PlantNameError):
        s.enter("mango")                               # confirmed: must explicitly correct
    s.correct()
    s.enter("mango")
    c = s.confirm()
    assert c["plant_id"] == "PLANT-MANGO" and c["corrections"] == 1
    assert [h["action"] for h in s.history] == ["enter", "select", "confirm", "correct", "enter", "confirm"]
    out = run(c)
    assert out["report"]["plant_identity"]["crop_confirmed_by_user"] == "Mango"


def test_dont_know_is_a_valid_choice():
    c = confirmed(dont_know=True)
    assert c["outcome"] == "not_known" and c["plant_name"] is None


# ------------------------------------------------------------------------------------------- routing
@pytest.mark.parametrize("pid", sorted(SIX))
def test_supported_crop_routes_to_the_existing_disease_model(pid):
    osr = FakeOSR()
    out = run(confirmed(select=pid), osr)
    assert osr.calls == 1 and out["workflow"]["routing"]["invoke_disease_model"] and out["report"]["model_prediction"]["ran"]
    assert out["report"]["plant_identity"]["in_model_scope"] is True


def test_supported_crop_with_known_matching_class_gives_a_model_class_candidate():
    out = run(confirmed("tomato"), reported_features=dict(rf.EB_FEATURES))
    rep = out["report"]
    assert rep["report_type"] == "known_disease"
    assert rep["candidate_diagnoses"][0]["kb_disease_id"] == "COND-TOMATO-EARLY-BLIGHT"
    assert rep["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] is True
    assert out["workflow"]["plant_selection"]["independently_verified"] is False      # agreement does not verify the name


def test_model_prediction_never_confirms_or_overrides_the_user_crop():
    out = run(confirmed("potato"), FakeOSR("Tomato___Early_blight", True))
    rep = out["report"]
    assert rep["plant_identity"]["crop_confirmed_by_user"] == "Potato"
    assert rep["plant_identity"]["model_supporting_evidence"]["agrees_with_user"] is False
    assert all(c["kb_disease_id"] != "COND-TOMATO-EARLY-BLIGHT" or c["ranking_method"] != "model_class_evidence"
               for c in rep["candidate_diagnoses"])
    assert any("not applicable" in s["en"] for s in rep["uncertainty"]["statements"])


@pytest.mark.parametrize("choice_kw,rtype", [({"text": "mango"}, "out_of_model_scope"), ({"text": "Paddy"}, "out_of_model_scope"),
                                             ({"text": "Coconut", "unlisted": True}, "out_of_model_scope"),
                                             ({"dont_know": True}, "unknown_investigation")])
def test_unsupported_or_unknown_plants_route_to_knowledge_guidance_without_the_model(choice_kw, rtype):
    osr = FakeOSR()
    kw = dict(choice_kw)
    out = run(confirmed(kw.pop("text", None), **kw), osr, reported_features={"FEAT-FRUIT-LESIONS": True})
    rep = out["report"]
    assert osr.calls == 0 and rep["model_prediction"]["ran"] is False and rep["osr_status"]["decision"] == "not_run"
    assert rep["report_type"] == rtype and not out["workflow"]["routing"]["invoke_disease_model"]
    assert not any(c["ranking_method"] == "model_class_evidence" for c in rep["candidate_diagnoses"])
    keys = {k["key"] for k in rep["content_keys"]}
    assert "act_consult_expert" in keys and "act_no_spray_yet" in keys


def test_quality_gate_failure_does_not_run_the_model():
    osr = FakeOSR()
    out = run(confirmed("tomato"), osr, image=BAD_IMG)
    assert osr.calls == 0 and out["report"]["report_type"] == "insufficient_image" and "quality" in out["workflow"]["routing"]["reason"]


def test_forged_or_tampered_choices_are_refused_and_never_enable_the_model():
    with pytest.raises(AS.AssessmentError):
        run({"outcome": "known_plant", "plant_id": "PLANT-MANGO", "plant_name": "Mango"})       # not from a session
    c = confirmed("mango")
    c["disease_model_supported"] = True                                                       # tampered flag is ignored
    osr = FakeOSR("Tomato___Early_blight", True)
    out = run(c, osr)
    assert osr.calls == 0 and out["report"]["model_prediction"]["ran"] is False
    bad = confirmed("tomato")
    bad["plant_id"] = "PLANT-COCONUT"
    with pytest.raises(AS.AssessmentError):
        run(bad)


def test_routing_agrees_with_the_m4_routing_library():
    from stage10.plant_id.routing import route
    for pid in KB.plants:
        d = AS.routing_decision(confirmed(select=pid), KB, rf.IMG)
        r = route(KB.plants[pid]["common_names"][0]["name"], "confirmed", KB, None)
        assert d["invoke_disease_model"] == (r["frozen_disease_model"] == "run")


# ------------------------------------------------------------------------------------------- disabled identification
def test_disabled_bioclip_identification_is_never_invoked(monkeypatch):
    from stage10.plant_id import adapter as PA

    def boom(*a, **k):
        raise AssertionError("automatic plant identification was invoked")

    monkeypatch.setattr(PA.PlantIdAdapter, "identify", boom)
    monkeypatch.setattr(PA.BioCLIPBackend, "__init__", boom)
    for c in (confirmed("tomato"), confirmed("mango"), confirmed("Coconut", unlisted=True), confirmed(dont_know=True)):
        out = run(c)
        assert out["workflow"]["plant_selection"]["automatic_identification_used"] is False
        assert out["workflow"]["routing"]["automatic_plant_identification"].startswith("disabled")


def test_workflow_does_not_load_identification_code_or_model_libraries():
    code = "\n".join([
        "import sys",
        "from stage10.knowledge_base.kb import load_kb",
        "from stage10.workflow import plant_name as PN, assessment as AS, cli",
        "from stage10.tests import report_fixtures as rf",
        "kb = load_kb(); s = PN.PlantNameSession(kb); s.enter('tomato'); c = s.confirm()",
        "class F:",
        "    def assess(self, i): return rf.synthetic_assessment('Tomato___Early_blight', True)",
        "AS.assess_plant_health(c, None, kb, osr_adapter=F(), image_info=rf.IMG, generated_at=rf.TS)",
        "bad = [m for m in ('stage10.plant_id.adapter', 'stage10.plant_id.pilot.evaluate_pilot',",
        "                   'stage10.plant_id.protocol_v3.methods', 'open_clip') if m in sys.modules]",
        "print(bad); sys.exit(1 if bad else 0)"])
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    for f in (ROOT / "stage10" / "workflow").glob("*.py"):
        t = f.read_text(encoding="utf-8")
        for forbidden in ("plant_id.adapter", "plant_id.pilot", "plant_id.protocol_v3", "plant_id.d2", "open_clip"):
            assert forbidden not in t, (f.name, forbidden)


def test_identification_operating_points_remain_unresolved():
    for p in (ROOT / "stage10/plant_id/d2/protocol_v2_operating_point.json", ROOT / "stage10/plant_id/protocol_v3/protocol_v3_operating_point.json"):
        assert json.loads(p.read_text(encoding="utf-8"))["status"] == "unresolved"
    assert not (ROOT / "stage10/plant_id/d2/results/d2_sealed_results.json").exists()


# ------------------------------------------------------------------------------------------- report provenance
def test_report_provenance_and_uncertainty():
    out = run(confirmed("Tomato"), reported_features={"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": False},
              reported_channels={"FEAT-CONCENTRIC-RINGS": "typed_text_confirmed"})
    rep, wf = out["report"], out["workflow"]
    pi = rep["plant_identity"]
    assert pi["user_confirmation"] == "confirmed" and pi["crop_confirmed_by_user"] == "Tomato" and pi["species_level_claimed"] is False
    assert any("crop name you confirmed" in s["en"] for s in pi["uncertainty"]["statements"])
    assert wf["plant_selection"]["source"] == "user_supplied" and wf["plant_selection"]["entered_text"] == "Tomato"
    users = [s for s in rep["visible_symptoms"] if s["origin"] == "user_report"]
    assert users and all(s["evidence_status"] == "observed" for s in users)
    assert {s["input_channel"] for s in users} <= {"symptom_checklist", "typed_text_confirmed"}
    assert all(s["evidence_status"] != "observed" for s in rep["visible_symptoms"] if s["origin"] == "knowledge_base_typical")
    assert not [s for s in rep["visible_symptoms"] if s["origin"] == "image_analysis"]       # no image analyser exists
    assert rep["uncertainty"]["level"] in ("moderate", "low") and rep["uncertainty"]["statements"]
    assert rep["candidate_diagnoses"] and all(c["source_ids"] for c in rep["candidate_diagnoses"])
    assert out["evidence_trace"]["entries"] and all(e["evidence"] for e in out["evidence_trace"]["entries"])
    keys = {k["key"] for k in rep["content_keys"]}
    assert {"warn_no_doses", "warn_no_chemicals", "warn_not_confirmed", "disclaimer"} <= keys
    text = json.dumps(rep, ensure_ascii=False).lower()
    assert "new species" not in text and "newly discovered" not in text


def test_no_disease_is_claimed_from_a_plant_name_alone():
    out = run(confirmed("mango"))
    rep = out["report"]
    assert rep["model_prediction"]["ran"] is False and rep["uncertainty"]["level"] in ("not_assessable", "low")
    assert all(c["ranking_method"] == "kb_symptom_match" for c in rep["candidate_diagnoses"])


# ------------------------------------------------------------------------------------------- API / schema / errors
def test_schemas_are_valid_and_workflow_records_conform():
    for n in ("plant_name_resolution", "plant_name_assessment"):
        validator(n).check_schema(validator(n).schema)
    wf = run(confirmed("tomato"))["workflow"]
    assert structural_errors(wf, "plant_name_assessment") == []
    bad = copy.deepcopy(wf)
    bad["routing"].update(disease_model_supported=False, invoke_disease_model=True)
    assert structural_errors(bad, "plant_name_assessment")
    bad = copy.deepcopy(wf)
    bad["plant_selection"]["automatic_identification_used"] = True
    assert structural_errors(bad, "plant_name_assessment")


def test_engine_input_validation_still_applies():
    with pytest.raises(ReportInputError):
        run(confirmed("tomato"), reported_features={"FEAT-DOES-NOT-EXIST": True})
    with pytest.raises(ReportInputError):
        run(confirmed("tomato"), reported_features={"FEAT-CONCENTRIC-RINGS": True},
            reported_channels={"FEAT-CONCENTRIC-RINGS": "voice_transcript"})       # unconfirmed voice is never an observation


def test_session_channel_errors_and_voice_gating(monkeypatch):
    s = PN.PlantNameSession(KB)
    with pytest.raises(PN.PlantNameError):
        s.enter("tomato", channel="sms")
    with pytest.raises(PN.PlantNameError):
        s.select("PLANT-COCONUT")
    # without installed local engines the confirmed-voice channel is refused
    monkeypatch.setattr(V, "voice_input_available", lambda: False)
    with pytest.raises(PN.PlantNameError, match="voice input is not available"):
        PN.PlantNameSession(KB).enter("tomato", channel=PN.VOICE_CHANNEL)
    # with engines, a confirmed transcript is resolved by exactly the typed rules (no looser matching for speech)
    monkeypatch.setattr(V, "voice_input_available", lambda: True)
    for text in ("tomato", "tomatoe", "Coconut", ""):
        sv = PN.PlantNameSession(KB)
        assert sv.enter(text, channel=PN.VOICE_CHANNEL) == PN.resolve_plant_name(text, KB)
    doc = (ROOT / "stage10" / "workflow" / "VOICE_INPUT_CONTRACT.md").read_text(encoding="utf-8")
    assert "confirm" in doc.lower() and "same rules as typed" in doc.lower()


def test_cli_entry_point(capsys):
    assert cli.main(["--list-plants"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == len(KB.plants)
    assert cli.main(["--plant", "pepper"]) == 2
    assert cli.main(["--plant", "tomato"]) == 2                # not confirmed -> does not proceed
    assert cli.main(["--plant", "tomato", "--confirm"]) == 2   # no image -> no assessment
    assert cli.main(["--plant", "mango", "--confirm", "--image", "missing.jpg"]) == 2
    assert cli.main(["--plant", "Coconut", "--keep-unlisted"]) == 2   # kept, but still needs --confirm
