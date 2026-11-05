"""Coverage registry: up to date, complete, and never equating KB coverage with image-diagnosis capability."""
from __future__ import annotations

import json

from stage10.knowledge_base.coverage import OUT, build
from stage10.knowledge_base.kb import ROOT, load_kb

KB = load_kb()
REG = build(KB)


def test_committed_registry_is_up_to_date():
    assert json.loads(OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(REG, ensure_ascii=False))


def test_every_plant_and_condition_is_registered():
    assert set(REG["plants"]) == set(KB.plants) and set(REG["conditions"]) == set(KB.conditions)


def test_image_diagnosis_never_claims_validation():
    for sec in ("plants", "conditions"):
        for k, v in REG[sec].items():
            st = v["image_diagnosis"]["status"]
            assert st in {"frozen_model_crop", "frozen_model_known_class", "held_out_unknown_in_research", "not_covered_by_frozen_model"}
            assert "NOT validated" in v["image_diagnosis"]["note"] or st in {"held_out_unknown_in_research", "not_covered_by_frozen_model"}
            assert v["expert_reviewed"] is False


def test_out_of_model_plants_have_kb_but_no_image_capability():
    for pid in ("PLANT-MANGO", "PLANT-RICE"):
        p = REG["plants"][pid]
        assert p["frozen_model_scope"] is False and p["image_diagnosis"]["status"] == "not_covered_by_frozen_model"
        assert p["conditions_sourced"]


def test_held_out_diseases_reference_frozen_stage8_numbers():
    nums = {r["unknown_class"]: r for r in json.loads((ROOT / "results/final/final_numbers.json").read_text())["per_disease"]}
    held = {c: v for c, v in REG["conditions"].items() if v["image_diagnosis"]["status"] == "held_out_unknown_in_research"}
    assert len(held) == 5
    for v in held.values():
        ref = v["image_diagnosis"]["frozen_stage8_osr_reference"]
        assert ref["detection_at_frozen_threshold"] == round(nums[v["image_diagnosis"]["class_name"]]["Contrastive Energy*::detection"], 4)


def test_stubs_are_visible_as_gaps():
    stubs = [c for c, v in REG["conditions"].items() if v["knowledge_base_information"] == "stub_unsourced"]
    assert all(REG["conditions"][c]["kb_text_retrieval"].startswith("excluded") for c in stubs)
    assert REG["summary"]["conditions_stub"] == len(stubs)
    # the 8 frozen-model classes that were stubs until 2026-10-15 are now sourced (KB curation for SC-M5-02)
    for c in ("COND-TOMATO-BACTERIAL-SPOT", "COND-TOMATO-LEAF-MOLD", "COND-TOMATO-YELLOW-LEAF-CURL-VIRUS", "COND-POTATO-EARLY-BLIGHT",
              "COND-POTATO-LATE-BLIGHT", "COND-PEPPER-BACTERIAL-SPOT", "COND-APPLE-CEDAR-APPLE-RUST", "COND-GRAPE-LEAF-BLIGHT-ISARIOPSIS"):
        assert REG["conditions"][c]["knowledge_base_information"] != "stub_unsourced", c


def test_capabilities_state_unvalidated_and_missing_parts():
    cap = REG["capabilities"]
    assert "UNVALIDATED" in cap["kb_text_retrieval"] and cap["expert_review"] == "none performed"
    assert cap["reference_image_retrieval"].startswith("not available")
