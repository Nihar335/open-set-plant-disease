"""Protocol v3 (palm-aware refusal; PROTOCOL_V3_PROPOSAL.md): targeted tests for palm ambiguity, unsupported plants, refusal
behaviour, missing inputs, routing, development-only data access and immutability of protocol v2.

Synthetic probability vectors and a FAKE backend only; no model is loaded and no sealed test image is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stage10.knowledge_base.kb import load_kb
from stage10.plant_id import adapter as A
from stage10.plant_id import routing as RT
from stage10.plant_id.d2 import thresholds as T
from stage10.plant_id.protocol_v3 import evaluate_v3 as EV
from stage10.plant_id.protocol_v3 import methods as M

KB = load_kb()
ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "stage10" / "plant_id" / "protocol_v3"
D2 = ROOT / "stage10" / "plant_id" / "d2"
NAMES = EV.vocab_names()
OP_B = {"status": "resolved", "adopted": "B", "tau_g": 0.968, "tau_c": 0.61}
OP_A = {"status": "resolved", "adopted": "A", "tau": 0.94}
OP_C = {"status": "resolved", "adopted": "C", "tau": 0.98}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probs(top: str, s: float, second: str | None = None, s2: float = 0.0) -> list[float]:
    rest = (1 - s - s2) / (len(NAMES) - (2 if second else 1))
    p = [rest] * len(NAMES)
    p[NAMES.index(top)] = s
    if second:
        p[NAMES.index(second)] = s2
    return p


# ------------------------------------------------------------------------------------------------ palm ambiguity
def test_coconut_split_with_areca_is_answered_only_by_the_coconut_stratum():
    p = probs(M.COCONUT, 0.62, M.ARECA, 0.37)                     # typical development coconut: mass split with areca
    assert M.decide(p, NAMES, OP_B) == {"outcome": "suggestion", "taxon": M.COCONUT}
    assert M.decide(p, NAMES, OP_A)["outcome"] == "unidentified"   # palm confirmation step never answers a palm
    assert M.decide(p, NAMES, {"status": "resolved", "adopted": "C", "tau": 0.99})["taxon"] is None


def test_areca_look_alike_palm_is_refused_by_palm_confirmation_but_not_by_the_coconut_stratum():
    p = probs(M.ARECA, 0.99, M.COCONUT, 0.005)                   # e.g. Dypsis lutescens scored as areca
    assert M.decide(p, NAMES, OP_A)["outcome"] == "unidentified"
    # V3-B deliberately keeps areca at the general threshold: it does NOT resolve areca look-alikes (residual risk)
    assert M.decide(p, NAMES, OP_B) == {"outcome": "suggestion", "taxon": M.ARECA}


def test_sibling_renormalised_score():
    p = probs(M.COCONUT, 0.6, M.ARECA, 0.38)
    assert M.score_c({"probs": p}, NAMES) == pytest.approx(0.6 / 0.62)
    q = probs("Mangifera indica", 0.9)
    assert M.score_c({"probs": q}, NAMES) == pytest.approx(0.9)
    assert M.score_a({"probs": probs(M.ARECA, 0.999)}, NAMES) == M.REFUSED


# ---------------------------------------------------------------------------------------------- unsupported plants
@pytest.mark.parametrize("op", [OP_A, OP_B, OP_C])
def test_low_scoring_unsupported_plant_is_refused_by_every_candidate(op):
    for top in ("Mangifera indica", M.ARECA, M.COCONUT):
        assert M.decide(probs(top, 0.2), NAMES, op) == {"outcome": "unidentified", "taxon": None}


def test_coconut_stratum_threshold_never_exceeds_the_bound_and_defaults_to_the_general_threshold():
    rng = np.random.default_rng(3)
    uns = [{"score": float(rng.uniform(0.1, 0.99)), "photographer": f"u{i % 120}", "group": T.GROUPS[i % 5], "taxon": f"t{i % 30}",
            "top1": "Mangifera indica"} for i in range(300)]
    sel = M.select_b(uns, b=400)
    assert sel["tau_c"] == sel["tau_g"] and sel["U"] <= T.TARGET          # no unsupported coconut evidence -> no lower bar
    for i in range(0, 30, 3):
        uns[i]["top1"] = M.COCONUT
    sel = M.select_b(uns, b=400)
    assert sel["U"] <= T.TARGET and sel["tau_g"] is not None and sel["tau_c"] is not None


def test_palm_confirmation_caps_supported_coverage_by_construction():
    recs = [{"probs": probs(M.COCONUT, 0.99), "taxon": M.COCONUT, "role": "supported_control", "group": "",
             "photographer": "p1"}]
    uns, sup = M._records(recs, NAMES, M.score_a)
    assert sup[0]["score"] < 0 and not any(r["score"] >= 0 for r in sup)


# ---------------------------------------------------------------------------------------- refusal and missing inputs
@pytest.mark.parametrize("op", [None, {}, {"status": "unresolved", "adopted": None}, {"status": "resolved", "adopted": "Z"}])
def test_no_resolved_operating_point_means_not_run(op):
    assert M.decide(probs("Mangifera indica", 0.999), NAMES, op) == {"outcome": "not_run", "taxon": None}


@pytest.mark.parametrize("p", [None, [0.5, 0.5], []])
def test_missing_or_malformed_scores_mean_not_run(p):
    assert M.decide(p, NAMES, OP_B)["outcome"] == "not_run"


def test_lock_detects_any_change_and_refuses_overwrites(tmp_path, monkeypatch):
    lk = json.loads(EV.LOCK.read_text(encoding="utf-8"))
    EV.verify_lock()                                              # the real lock verifies
    f = tmp_path / "x.md"
    f.write_text("a", encoding="utf-8")
    fake = tmp_path / "lock.json"
    fake.write_text(json.dumps({"locked_utc": "t", "files": {str(f): sha(f)}}), encoding="utf-8")
    monkeypatch.setattr(EV, "LOCK", fake)
    monkeypatch.setattr(EV, "ROOT", Path("/"))
    EV.verify_lock()
    f.write_text("b", encoding="utf-8")
    with pytest.raises(SystemExit):
        EV.verify_lock()
    monkeypatch.setattr(EV, "LOCK", tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        EV.verify_lock()
    assert lk["files"]
    monkeypatch.setattr(EV, "ROOT", ROOT)
    monkeypatch.setattr(EV, "LOCK", V3 / "protocol_v3_prereg_lock.json")
    with pytest.raises(SystemExit):                                # real lock exists: no re-lock
        EV.write_lock()
    with pytest.raises(SystemExit):                                # real operating point exists: no re-selection
        EV.select()


# -------------------------------------------------------------------------------------------- development-only access
def test_development_reader_refuses_any_test_row(monkeypatch, tmp_path):
    rows = EV.dev_rows()
    assert rows and all(r["split"] == "dev" for r in rows)
    man = tmp_path / "manifests"
    man.mkdir()
    bad = dict(rows[0], split="test", stored_path=EV.TEST_DIR + "/x/1.jpg")
    with open(man / "d2_dev_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bad))
        w.writeheader()
        w.writerow(bad)
    monkeypatch.setattr(EV, "D2", tmp_path)
    with pytest.raises(SystemExit):
        EV.dev_rows()


def test_v3_scores_cover_exactly_the_development_images():
    with open(EV.FULL, newline="", encoding="utf-8") as f:
        ids = [r["plan_id"] for r in csv.DictReader(f)]
    assert len(ids) == 535 and len(set(ids)) == 535
    assert all(i.startswith(("D2-dev-", "C1-dev-")) for i in ids)
    with open(D2 / "results" / "d2_dev_scores.csv", newline="", encoding="utf-8") as f:
        v2 = {r["plan_id"]: float(r["score"]) for r in csv.DictReader(f)}
    full = EV.load_full()
    assert set(v2) == {r["plan_id"] for r in full}
    assert max(abs(max(r["probs"]) - v2[r["plan_id"]]) for r in full) < 1e-5     # same model, same images as v2


# -------------------------------------------------------------------------------------------- recorded v3 outcome
def test_recorded_v3_outcome_is_unresolved_and_nothing_was_relaxed():
    op = json.loads(EV.OP.read_text(encoding="utf-8"))
    assert op["protocol"] == "v3" and op["data"] == "development only" and op["sealed_test_read"] is False
    assert op["prereg_lock_sha256"] == sha(EV.LOCK) and op["dev_full_softmax_sha256"] == sha(EV.FULL)
    assert op["status"] == "unresolved" and op["adopted"] is None and "tau" not in op
    c = op["candidates"]
    assert "coverage" in c["A"]["reason"] and "tau_c range" in c["B"]["reason"] and "correct-with-refusals" in c["C"]["reason"]
    assert T.TARGET == 0.10 and T.MIN["supported_images"] == 100
    for k in "ABC":
        sel = c[k]["selection"]
        assert sel["U"] <= 0.10                                   # the 10% bound held for every selected rule
    assert not (V3 / "SEALED_SCORING_APPROVAL_D3.json").exists()


def test_v3_unresolved_keeps_identification_disabled():
    op = json.loads(EV.OP.read_text(encoding="utf-8"))
    assert M.decide(probs(M.COCONUT, 0.99), NAMES, op)["outcome"] == "not_run"
    assert A.load_operating_point()["status"] == "unresolved"      # adapter still bound to v2, which is unresolved


# ----------------------------------------------------------------------------------------------------- routing
def test_palm_refusal_routes_to_user_confirmation_and_never_to_a_diagnosis():
    class Fake:
        name, version, weights_sha256, licence = "fake", "0", "0" * 64, "test-only"

        def scores(self, b):
            return sorted(zip(NAMES, probs(M.COCONUT, 0.62, M.ARECA, 0.37)), key=lambda x: -x[1])

    res = A.PlantIdAdapter(Fake(), kb=KB).identify(b"")            # real (unresolved) operating point -> not_run
    assert res["status"] == "not_run"
    r = RT.route(None, "not_provided", KB, res)
    assert r["plant_identification"] == "refused:not_run" and r["frozen_disease_model"] == "unconfirmed_scope"
    assert r["confirmed_plant"] is None and r["user_prompt"]
    r = RT.route("Coconut", "confirmed", KB, res)
    assert r["frozen_disease_model"] == "not_applicable" and not r["disease_assessment_from_identification"]


# ------------------------------------------------------------------------------------------- v2 left unchanged
V2_HASHES = {
    "protocol_v2_operating_point.json": "b65f4ab37b5c24f9cde27c4a91f206a03e1bc2ed601de76713fa7ef43c23a015",
    "results/d2_dev_scores.csv": "77bfc95619b5d6f6dfa92e3aecf9dcb942f3c14affea1f92507f56460a9cd819",
    "PROTOCOL_V2.md": "d80747c41e0e0fcaa8036a42399661606c1462f8644278dce3f153b570632081",
    "SEAL_D2.json": "8a2d094533f9a83d110d8f1803da8e93414018f1a33b367df325def2188f6185",
}


def test_protocol_v2_artifacts_are_unchanged_and_v2_lock_verifies():
    for rel, h in V2_HASHES.items():
        assert sha(D2 / rel) == h, rel
    lk = json.loads((D2 / "protocol_v2_lock.json").read_text(encoding="utf-8"))
    for rel, h in lk["files"].items():
        assert sha(ROOT / rel) == h, rel
    assert not (D2 / "results" / "d2_sealed_results.json").exists()
    assert not (D2 / "SEALED_SCORING_APPROVAL_D2.json").exists()
