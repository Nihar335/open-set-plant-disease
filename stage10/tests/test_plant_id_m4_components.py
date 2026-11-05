"""Milestone 4 components: result contract (M4-11), adapter with abstention (M4-06), identification registry (M4-07) and
routing (M4-08). Synthetic images and a FAKE model backend only: no model is loaded, no external service is called, and
nothing is wired into a farmer-facing application.
"""
from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stage10.knowledge_base.kb import load_kb
from stage10.plant_id import adapter as A
from stage10.plant_id import identification_registry as R
from stage10.plant_id import routing as RT
from stage10.plant_id.contract import contract_errors
from stage10.report_engine.engine import build_report
from stage10.schemas.validate import validator
from stage10.tests import report_fixtures as rf

KB = load_kb()
ROOT = Path(__file__).resolve().parents[2]
VOCAB = [t["scientific_name"] for t in json.loads(A.VOCAB.read_text(encoding="utf-8"))["taxa"]]
OP = {"status": "resolved", "threshold": 0.85, "threshold_source": A.THRESHOLD_SOURCE, "operating_point_sha256": "a" * 64}


class FakeBackend:
    name, version, weights_sha256, licence = "fake", "0", "0" * 64, "test-only"

    def __init__(self, top: str, top_score: float):
        rest = (1 - top_score) / (len(VOCAB) - 1)
        self.s = [(n, top_score if n == top else rest) for n in VOCAB]

    def scores(self, image_bytes):
        return sorted(self.s, key=lambda x: -x[1])


def img(size=(320, 240)) -> bytes:
    rng = np.random.default_rng(0)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(b, format="JPEG")
    return b.getvalue()


def result(top="Mangifera indica", score=0.93, op=OP, image=None):
    return A.PlantIdAdapter(FakeBackend(top, score), operating_point=dict(op), kb=KB).identify(image or img())


# ------------------------------------------------------------------------------------------- M4-11 contract
def test_schema_is_valid_2020_12_and_every_adapter_outcome_conforms():
    validator("plant_identification").check_schema(validator("plant_identification").schema)
    for r in (result(), result(score=0.40), result(image=img((100, 80))), result(op={**OP, "status": "absent", "threshold": None,
                                                                                         "operating_point_sha256": None})):
        assert contract_errors(r) == [], r["status"]


VIOLATIONS = {
    "more than 5 candidates": lambda d: d["candidates"].extend(copy.deepcopy(d["candidates"][:1]) for _ in range(1)),
    "user confirmation not required": lambda d: d.update(requires_user_confirmation=False),
    "sets the confirmed crop": lambda d: d.update(sets_confirmed_crop=True),
    "claims a disease assessment": lambda d: d.update(disease_assessment_performed=True),
    "score claimed calibrated": lambda d: d["candidates"][0].update(score_is_calibrated_probability=True),
    "candidates while abstaining": lambda d: d["abstention"].update(abstained=True, reason="below_threshold"),
    "unidentified with candidates": lambda d: d.update(status="unidentified", message_key="plant_id.unidentified"),
    "candidates without resolved operating point": lambda d: d["operating_point"].update(status="unresolved"),
    "validated without evaluation id": lambda d: d["evaluated_scope"].update(validated=True),
    "image left device without consent": lambda d: d["privacy"].update(image_left_device=True),
    "external service without consent": lambda d: d["model"].update(external_service="pl@ntnet"),
    "extra disease field": lambda d: d["candidates"][0].update(disease="Anthracnose"),
    "threshold source not development": lambda d: d["operating_point"].update(threshold_source="test set"),
    "ranks out of order": lambda d: d["candidates"].reverse(),
    "duplicate taxa": lambda d: d["candidates"][1].update(scientific_name=d["candidates"][0]["scientific_name"]),
    "rank-1 below threshold": lambda d: d["candidates"][0].update(above_threshold=False),
    "validated without registry entry": lambda d: d["evaluated_scope"].update(validated=True, evaluation_id="EVAL-X"),
    "percentage shown": lambda d: d["message"].update(en="Mango (93% confidence)"),
    "new species wording": lambda d: d["message"].update(en="This may be a new species."),
    "image leaves device even with consent": lambda d: d["privacy"].update(image_left_device=True, consent_reference="C-1"),
}


@pytest.mark.parametrize("name", list(VIOLATIONS))
def test_each_contract_rule_rejects_a_violating_fixture(name):
    d = result()
    assert d["status"] == "candidates" and len(d["candidates"]) == 5 and contract_errors(d) == []
    if name == "more than 5 candidates":
        d["candidates"].append({**copy.deepcopy(d["candidates"][-1]), "rank": 6, "scientific_name": "Extra taxon"})
    else:
        VIOLATIONS[name](d)
    assert contract_errors(d), name


# ------------------------------------------------------------------------------------------- M4-06 adapter
def test_adapter_abstains_below_the_development_threshold():
    above, below = result(score=0.93), result(score=0.84)
    assert above["status"] == "candidates" and above["candidates"][0]["scientific_name"] == "Mangifera indica"
    assert below["status"] == "unidentified" and below["candidates"] == [] and below["abstention"]["reason"] == "below_threshold"
    assert below["message"]["en"].startswith("Potentially unfamiliar to this model; further investigation is recommended.")
    assert "KVK" in below["message"]["en"] or "plant name yourself" in below["message"]["en"]


def test_adapter_refuses_without_a_resolved_development_operating_point():
    for st in ("absent", "unresolved"):
        r = result(op={**OP, "status": st, "threshold": None, "operating_point_sha256": None})
        assert r["status"] == "not_run" and r["candidates"] == []
    assert A.load_operating_point(Path("does-not-exist.json"))["status"] == "absent"


def test_adapter_reports_insufficient_images_instead_of_guessing():
    r = result(image=img((150, 150)))
    assert r["status"] == "insufficient_image" and r["candidates"] == [] and "224" in r["message"]["en"]


def test_adapter_output_is_a_suggestion_only():
    r = result()
    assert r["requires_user_confirmation"] and not r["sets_confirmed_crop"] and not r["disease_assessment_performed"]
    assert r["candidates"][0]["kb_plant_id"] == "PLANT-MANGO"
    assert all(not c["score_is_calibrated_probability"] for c in r["candidates"])
    assert r["model"]["external_service"] is None and r["privacy"]["image_left_device"] is False
    assert r["message_key"] in A.MESSAGES and set(r["message"]) == {"en"}          # no unreviewed translation is emitted


def test_operating_point_file_if_present_comes_from_development_data_only():
    op = A.OPERATING_POINT
    if not op.exists():
        pytest.skip("protocol v2 development phase not run")
    d = json.loads(op.read_text(encoding="utf-8"))
    assert d["data"] == "development only" and d["sealed_test_read"] is False


# ------------------------------------------------------------------------------------------- M4-07 registry
def test_registry_separates_identifiable_from_diagnosable_and_is_current():
    reg = R.build()
    assert json.loads(R.OUT.read_text(encoding="utf-8")) == reg
    for name, t in reg["taxa"].items():
        assert set(t) == {"common_name_en", "identifiable", "diagnosable"}
        assert t["identifiable"]["status"] in R.STATUSES
        assert t["identifiable"]["status"] == "not_established" and t["identifiable"]["evaluation_id"] is None
    frozen = {n for n, t in reg["taxa"].items() if t["diagnosable"]["frozen_disease_model_scope"]}
    assert frozen == {"Solanum lycopersicum", "Zea mays", "Capsicum annuum", "Solanum tuberosum"}   # Apple and grape are not in vocabulary v1
    assert reg["outside_vocabulary"]["status"] == "not_supported"


CLAIM = re.compile(r"(identif\w*|recogni[sz]\w*) (any|every|all) plants?|universal (plant )?(recognition|identification)", re.I)
NEGATION = re.compile(r"\b(not|never|no|cannot|without|nor)\b", re.I)


def test_no_universal_recognition_claim_anywhere():
    files = [p for p in (ROOT / "stage10").rglob("*") if p.suffix in (".md", ".yaml", ".json", ".py") and p.is_file()
             and "__pycache__" not in p.parts and p.name != Path(__file__).name]
    bad = []
    for p in files:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", p.read_text(encoding="utf-8", errors="ignore")):
            if CLAIM.search(sentence) and not NEGATION.search(sentence):
                bad.append(f"{p.relative_to(ROOT)}: {sentence.strip()[:120]}")
    assert bad == []


# ------------------------------------------------------------------------------------------- M4-08 routing
def test_routing_in_scope_kb_only_and_unknown_match_the_report_engine():
    ident_tomato = result(top="Solanum lycopersicum")
    # in scope: confirmed tomato -> frozen model applicable
    r = RT.route("Tomato", "confirmed", KB, ident_tomato)
    assert r["frozen_disease_model"] == "run" and r["identity_source"] == "user_confirmed"
    rep = build_report(rf.known_eb(), KB).report
    assert rep["report_type"] == "known_disease"
    # KB-only: confirmed mango -> not applicable, even when identification (wrongly) suggests tomato
    r = RT.route("Mango", "confirmed", KB, ident_tomato)
    assert r["frozen_disease_model"] == "not_applicable" and r["kb_plant_id"] == "PLANT-MANGO"
    assert r["knowledge_base"] == "plant_specific" and r["expected_report_type"] == "out_of_model_scope"
    rep = build_report(rf.request(plant_name="Mango", plant_confirmation="confirmed",
                                  reported_features={"FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True, "FEAT-FRUIT-LESIONS": True},
                                  model_assessment=rf.synthetic_assessment("Tomato___Early_blight", True)), KB).report
    assert rep["report_type"] == r["expected_report_type"] and rep["candidate_diagnoses"][0]["kb_disease_id"].startswith("COND-MANGO")
    # unknown plant: not confirmed, identification refuses -> unconfirmed scope, investigation, user asked to confirm
    r = RT.route(None, "not_provided", KB, result(score=0.3))
    assert r["plant_identification"] == "refused:unidentified" and r["frozen_disease_model"] == "unconfirmed_scope"
    assert r["user_prompt"] and r["confirmed_plant"] is None
    assert build_report(rf.unknown_tomato(plant_confirmation="not_provided", plant_name=None), KB).report["report_type"] == "unknown_investigation"


def test_frozen_model_is_never_applied_as_a_diagnosis_outside_its_crops():
    for plant in ("Mango", "paddy", "Banana", "Coconut"):
        for ident in (None, result(top="Solanum lycopersicum", score=0.99)):
            r = RT.route(plant, "confirmed", KB, ident)
            assert r["frozen_disease_model"] == "not_applicable" and not r["disease_assessment_from_identification"]


def test_identification_never_confirms_a_plant_and_contract_violations_are_refused():
    r = RT.route(None, "unsure", KB, result(top="Mangifera indica", score=0.99))
    assert r["identity_source"] == "unconfirmed" and r["plant_identification"] == "suggestion_only" and r["confirmed_plant"] is None
    bad = result()
    bad["sets_confirmed_crop"] = True
    with pytest.raises(ValueError):
        RT.route(None, "not_provided", KB, bad)


# ------------------------------------------------------------------------------------------- threshold method (synthetic)
def _synthetic(seed=1, n_unsup=300, n_sup=150, sep=0.35):
    rng = np.random.default_rng(seed)
    from stage10.plant_id.d2 import thresholds as T
    uns = [{"score": float(np.clip(rng.beta(2, 5), 0, 1)), "photographer": f"u{i % 150}", "group": T.GROUPS[i % 5],
            "taxon": f"t{i % 30}"} for i in range(n_unsup)]
    sup = [{"score": float(np.clip(rng.beta(2, 5) + sep, 0, 1)), "photographer": f"s{i % 120}", "taxon": f"k{i % 10}",
            "correct": bool(rng.random() < 0.95)} for i in range(n_sup)]
    return uns, sup


def test_threshold_bound_is_conservative_and_monotone():
    from stage10.plant_id.d2 import thresholds as T
    uns, _ = _synthetic()
    bd = T.Bound(uns, b=2000)
    us = [bd.components(t)["U"] for t in np.linspace(0, 1, 21)]
    assert all(a >= b - 1e-12 for a, b in zip(us, us[1:]))                 # non-increasing in tau
    c = bd.components(0.4)
    assert c["U"] >= max(c["cp_upper"], c["fa"]) and c["U"] == max(c["cp_upper"], c["boot_micro_upper"], c["boot_macro_upper"])
    sel = T.select(uns, b=2000)
    assert sel["U"] <= T.TARGET and sel["fa"] < T.TARGET                       # aims below 10%, not at it
    cand = T.candidates(uns)
    prev = cand[cand.index(sel["tau"]) - 1]
    assert T.Bound(uns, b=2000).components(prev)["U"] > T.TARGET               # smallest feasible


def test_threshold_unresolved_when_minimums_fail_and_never_relaxed():
    from stage10.plant_id.d2 import thresholds as T
    uns, sup = _synthetic(n_unsup=120)
    r = T.run(uns, sup, b=500)
    assert r["status"] == "unresolved" and "unsupported_images" in r["reason"] and "tau" not in r
    assert T.TARGET == 0.10 and T.MIN["unsupported_images"] == 200 and T.MIN["unsupported_photographers"] == 100


def test_threshold_feasibility_failure_reports_unresolved():
    from stage10.plant_id.d2 import thresholds as T
    uns, sup = _synthetic(sep=-0.2)             # supported plants score like unsupported ones -> coverage collapses
    r = T.run(uns, sup, b=500)
    assert r["status"] == "unresolved" and "coverage" in r["reason"]


# ------------------------------------------------------------------------------------------- sealed analysis + guards
def test_sealed_analysis_verdicts_on_synthetic_data():
    from stage10.plant_id.d2 import analysis as AN
    rng = np.random.default_rng(3)
    uns = [{"score": float(rng.beta(2, 8)), "photographer": f"p{i % 300}", "top1": "Mangifera indica"} for i in range(600)]
    o = AN.unsupported_outcome(uns, 0.6)
    assert o["accepted"] == sum(u["score"] >= 0.6 for u in uns) and o["one_sided_95_upper"] >= o["false_acceptance"]
    assert o["verdict"] in ("met in this study", "not met", "not established")
    bad = AN.unsupported_outcome([{**u, "score": 0.9} for u in uns[:50]] + uns[50:100], 0.6)
    assert bad["verdict"] == "not met"
    sup = [{"score": 0.95, "photographer": f"s{i}", "taxon": "Carica papaya", "top1": "Carica papaya", "top5": ["Carica papaya"]}
           for i in range(120)]
    assert AN.supported_outcome(sup, 0.6)["verdict"] == "met in this study"
    c1 = ([{"score": 0.9, "taxon": "Cocos nucifera", "top1": "Areca catechu"}] * 2 +
          [{"score": 0.9, "taxon": "Cocos nucifera", "top1": "Cocos nucifera"}] * 58 + [{"score": 0.9, "taxon": "Areca catechu", "top1": "Areca catechu"}] * 16)
    out = AN.c1_outcome(c1, 0.6)
    assert out["Cocos nucifera -> Areca catechu"]["verdict"] == "not met"
    assert out["Areca catechu -> Cocos nucifera"]["verdict"].startswith("descriptive only")


def test_development_phase_reads_only_the_development_manifest_and_sealed_phase_is_gated(tmp_path, monkeypatch):
    import inspect
    from stage10.plant_id.d2 import evaluate_d2 as EV
    src = inspect.getsource(EV.phase_dev)
    assert 'FINAL["dev"]' in src and 'FINAL["test"]' not in src and "verify_seal()" in src
    monkeypatch.setattr(EV, "APPROVAL", tmp_path / "none.json")
    with pytest.raises(SystemExit, match="not authorised"):
        EV.phase_sealed("cpu")
    assert not (EV.RESULTS / "d2_sealed_results.json").exists()            # the fresh test set has not been scored


# ------------------------------------------------------------------------------------------- report communication (M4 brief)
REPORT_CASES = {
    "known_disease": lambda: rf.known_eb(),
    "unknown_investigation": lambda: rf.unknown_tomato(),
    "out_of_model_scope": lambda: rf.request(plant_name="Mango", plant_confirmation="confirmed",
                                             reported_features={"FEAT-SUNKEN-BLACK-SPOTS-RIPENING-FRUIT": True, "FEAT-FRUIT-LESIONS": True}),
    "no_plant_given": lambda: rf.request(reported_features={"FEAT-YELLOW-HALO": True}),
}


@pytest.mark.parametrize("case", list(REPORT_CASES))
def test_reports_communicate_assessment_evidence_uncertainty_and_next_steps(case):
    from stage10.schemas.validate import _walk_localized
    r = build_report(REPORT_CASES[case](), KB).report
    # whether a disease assessment was actually performed is explicit
    assert r["osr_status"]["decision"] in ("known", "unknown", "not_run") and isinstance(r["model_prediction"]["ran"], bool)
    assert (r["osr_status"]["decision"] == "not_run") == (not r["model_prediction"]["ran"])
    if not r["model_prediction"]["ran"]:
        assert "not run" in r["osr_status"]["message"]["en"]
    # observed symptoms are distinguished from model or knowledge-base suggestions
    for s in r["visible_symptoms"]:
        assert s["evidence_status"] in ("observed", "inferred", "unverified") and s["origin"] and s["input_channel"]
    # guidance cites sources; uncertainty and next steps are always present
    if r["candidate_diagnoses"]:
        assert r["evidence_sources"]
    assert r["uncertainty"]["level"] and r["uncertainty"]["statements"] and r["follow_up_plan"]
    if r["report_type"] != "known_disease":
        assert {x["subcategory"] for x in r["immediate_actions"]} & {"consult_expert", "photography"}
    # scores never shown as percentages; nothing described as newly discovered
    for _, t in _walk_localized(r):
        assert not re.search(r"\d\s*%", t["en"])
        for sentence in re.split(r"(?<=[.!?])\s+", t["en"]):           # a discovery claim is allowed only when negated
            if re.search(r"new(ly)? (species|disease)", sentence, re.I) and re.search(r"discover", sentence, re.I):
                assert NEGATION.search(sentence), sentence
    if r["osr_status"]["decision"] == "unknown":
        assert r["osr_status"]["message"]["en"].startswith("Potentially unfamiliar to this model; further investigation is recommended.")
