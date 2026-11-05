"""KB retrieval behaviour (text/feature-based, UNVALIDATED). These are behaviour and consistency tests on the
knowledge base itself, NOT an accuracy evaluation: no independent labelled data exist yet."""
from __future__ import annotations

import random

import pytest

from stage10.knowledge_base.kb import load_kb
from stage10.retrieval.kb_retrieval import INSUFFICIENT, Query, retrieve

KB = load_kb()


def ids(r):
    return [c["condition_id"] for c in r["candidates"]]


def test_tomato_ring_spots_return_both_documented_lookalikes():
    r = retrieve(Query("tomato", {"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}), KB)
    assert set(ids(r)) == {"COND-TOMATO-TARGET-SPOT", "COND-TOMATO-EARLY-BLIGHT"}   # overlapping symptoms -> not forced to one


def test_corn_vein_limited_lesions_rank_gray_leaf_spot_without_padding():
    r = retrieve(Query("corn", {"FEAT-VEIN-LIMITED-RECTANGULAR": True, "FEAT-YELLOW-HALO": True}), KB)
    assert ids(r) == ["COND-MAIZE-GRAY-LEAF-SPOT"]                                   # only one supported -> one returned


def test_questionnaire_answers_drive_retrieval_and_not_sure_sets_nothing():
    r = retrieve(Query("Zea mays", {"FEAT-OLDER-LOWER-LEAVES-FIRST": True}, answers={"Q-LESION-SHAPE": "cigar", "Q-WEATHER": "cool_wet"}), KB)
    assert ids(r)[0] == "COND-MAIZE-NORTHERN-LEAF-BLIGHT"
    base = retrieve(Query("tomato", {"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True}), KB)
    unsure = retrieve(Query("tomato", {"FEAT-CONCENTRIC-RINGS": True, "FEAT-YELLOW-HALO": True},
                            answers={q: "not_sure" for q in ("Q-LEAF-AGE", "Q-WEATHER", "Q-BLACK-SPECKS")}), KB)
    assert [(c["condition_id"], c["ranking_score"]) for c in base["candidates"]] == \
           [(c["condition_id"], c["ranking_score"]) for c in unsure["candidates"]]


def test_esca_text_retrieval():
    r = retrieve(Query("grapevine", {"FEAT-INTERVEINAL-STRIPING": True, "FEAT-DARK-SPOTS-WOOD-SECTION": True}), KB)
    assert ids(r) == ["COND-GRAPE-ESCA"]
    c = r["candidates"][0]
    assert any(d["diagnostic_id"] == "DIAG-WOOD-CROSS-SECTION" for d in c["suggested_diagnostics"])
    assert c["region_note"] is None and "international_not_verified_for_India" in c["geographic_applicability"]


def test_crop_mismatch_gives_no_candidate():
    r = retrieve(Query("tomato", {"FEAT-INTERVEINAL-STRIPING": True, "FEAT-DARK-SPOTS-WOOD-SECTION": True}), KB)
    assert r["status"] == "insufficient_evidence" and r["candidates"] == [] and r["insufficient_evidence_statement"] == INSUFFICIENT


@pytest.mark.parametrize("obs", [{}, {"FEAT-YELLOW-HALO": True}, {"FEAT-ENV-COOL-WET": True, "FEAT-ENV-HIGH-HUMIDITY-WET": True}])
def test_abstains_when_evidence_is_inadequate(obs):
    r = retrieve(Query("tomato", obs), KB)
    assert r["status"] == "insufficient_evidence" and not r["candidates"]


def test_plant_outside_kb_is_reported_not_guessed():
    r = retrieve(Query("banana", {"FEAT-YELLOW-HALO": True, "FEAT-CONCENTRIC-RINGS": True}), KB)
    assert r["status"] == "plant_not_in_knowledge_base" and r["candidates"] == []


def test_unconfirmed_plant_is_flagged_and_candidates_marked():
    r = retrieve(Query(None, {"FEAT-BLACK-SPECKS-IN-LESION": True, "FEAT-FRUIT-MUMMY": True}), KB)
    assert r["notes"] and all(c["crop_compatible"] is None for c in r["candidates"])
    assert len({tuple(c["affected_plants"]) for c in r["candidates"]}) > 1        # spans crops -> user must confirm


def test_reported_absence_counts_as_conflicting_evidence():
    r = retrieve(Query("tomato", {"FEAT-CONCENTRIC-RINGS": False, "FEAT-YELLOW-HALO": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True,
                                  "FEAT-RAPID-BLIGHT-DEFOLIATION": True}), KB)
    assert "COND-TOMATO-EARLY-BLIGHT" not in ids(r)
    ts = next(c for c in r["candidates"] if c["condition_id"] == "COND-TOMATO-TARGET-SPOT")
    assert [x["feature_id"] for x in ts["conflicting_evidence"]] == ["FEAT-CONCENTRIC-RINGS"]


def test_non_infectious_causes_are_retrievable():
    r = retrieve(Query("tomato", {"FEAT-INTERVEINAL-CHLOROSIS-GREEN-MARGINS": True, "FEAT-OLDER-LOWER-LEAVES-FIRST": True}), KB)
    assert ids(r)[0] == "COND-TOMATO-MAGNESIUM-DEFICIENCY"
    assert r["candidates"][0]["cause_category"] == "nutrient_deficiency"


def test_region_note_flags_non_indian_evidence():
    r = retrieve(Query("corn", {"FEAT-VEIN-LIMITED-RECTANGULAR": True, "FEAT-YELLOW-HALO": True}, region="Karnataka"), KB)
    assert "not verified" in r["candidates"][0]["region_note"]


def test_extensible_beyond_model_crops():
    r = retrieve(Query("paddy", {"FEAT-SPINDLE-GREY-CENTRE-BROWN-MARGIN": True, "FEAT-NODE-OR-NECK-BLACK-LESIONS": True}), KB)
    assert ids(r) == ["COND-RICE-BLAST"]


def _check_invariants(r, plant_id):
    assert len(r["candidates"]) <= 3 and r["validation_status"] == "unvalidated"
    scores = [c["ranking_score"] for c in r["candidates"]]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    for c in r["candidates"]:
        rec = KB.conditions[c["condition_id"]]
        assert rec["curation_status"] == "sourced"                                      # stubs never returned
        assert c["ranking_score_is_probability"] is False and c["validation_status"] == "unvalidated"
        assert c["source_ids"] and all(s in KB.sources for s in c["source_ids"])
        if plant_id:
            assert c["crop_compatible"] is True and plant_id in c["affected_plants"]
        assert sum(1 for s in c["supporting_evidence"] if KB.features[s["feature_id"]]["category"] != "environment") >= 2
    if not r["candidates"]:
        assert r["insufficient_evidence_statement"] == INSUFFICIENT


def test_random_queries_respect_invariants():
    rng = random.Random(42)
    feats = list(KB.features)
    plants = [None, "tomato", "corn", "grape", "apple", "rice", "mango", "potato"]
    for _ in range(300):
        obs = {f: rng.random() < 0.8 for f in rng.sample(feats, rng.randint(0, 7))}
        p = rng.choice(plants)
        r = retrieve(Query(p, obs), KB)
        _check_invariants(r, KB.resolve_plant(p))


def test_kb_self_consistency_each_sourced_condition_retrievable_from_its_own_features():
    """Consistency check of the KB (NOT accuracy): a sourced condition queried with its own documented symptom features and
    host plant must appear among the candidates, whenever it has >= 2 symptom features."""
    for c in KB.sourced_conditions():
        feats = [f["feature_id"] for f in c["features"]]
        if len(feats) < 2:
            continue
        for a in c["affected_plants"]:
            r = retrieve(Query(a["plant_id"], {f: True for f in feats}), KB)
            assert c["condition_id"] in ids(r), (c["condition_id"], ids(r))


def test_retrieval_query_has_no_model_input():
    assert set(Query.__dataclass_fields__) == {"plant", "observed", "answers", "region"}
