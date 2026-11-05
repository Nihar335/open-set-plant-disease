"""Real pilot manifests: run the independent validator, and check that the protocol was frozen and nothing was scored."""
from __future__ import annotations

import json

import pytest

from stage10.plant_id.pilot import common as C
from stage10.plant_id.pilot import validate_pilot as V

HAVE = (C.HERE / "manifests" / "p1p2_dev_manifest.csv").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="pilot not acquired")
LOCAL = (C.ROOT / C.config()["storage_root"] / "images").exists()


def test_independent_validation_passes():
    err, summary = V.validate(check_files=LOCAL)
    assert err == [], err[:10]
    assert summary["rows"] > 0


def test_p3_is_sealed_only_and_p2_not_in_vocabulary():
    rs = V.rows()
    assert {r["split"] for r in rs if r["block"] == "P3"} <= {"sealed"}      # P3 may be absent (blocked), but never dev
    vocab = {t["scientific_name"] for t in json.loads((C.HERE / "vocabulary_v1.json").read_text(encoding="utf-8"))["taxa"]}
    assert not {r["taxon"] for r in rs if r["block"] == "P2"} & vocab


def test_protocol_frozen_with_unchanged_provisional_gates():
    p = json.loads((C.HERE / "protocol_v1.json").read_text(encoding="utf-8"))
    g = p["provisional_gates_unchanged"]
    assert (g["top1_accuracy_min"], g["at_coverage_min"], g["unsupported_false_acceptance_max"], g["p95_latency_ms_max"]) == (0.80, 0.70, 0.10, 2000)
    assert p["status"].startswith("frozen_v1") and "EXPLORATORY" in p["nature"]
    assert "P3" in p["data"] and "never used for any model choice" in p["data"]["P3"]


def test_sealed_scoring_followed_the_approved_order():
    """Owner 2026-10-15: score only after resealing passed; operating point frozen on dev BEFORE sealed scoring; scored once."""
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    appr = json.loads(C.APPROVAL.read_text(encoding="utf-8"))
    lock = json.loads((C.HERE / "protocol_lock.json").read_text(encoding="utf-8"))
    assert appr["approved_by_owner"] is True and appr["seal_sha256"] == sha(C.HERE / "SEAL.json")
    assert appr["protocol_sha256"] == lock["files"]["protocol_v1.json"] == sha(C.HERE / "protocol_v1.json")
    op = C.HERE / "protocol_v1_operating_point.json"
    assert lock["files"]["protocol_v1_operating_point.json"] == sha(op)
    res = json.loads((C.HERE / "results" / "pilot_sealed_results.json").read_text(encoding="utf-8"))
    opj = json.loads(op.read_text(encoding="utf-8"))
    assert opj["created_utc"] < res["created_utc"] and res["protocol"]["operating_point_sha256"] == sha(op)
    assert opj["template"] == "T1" and opj["dev"]["p2_images"] == 14


def test_results_keep_p1_p2_separate_and_respect_count_only_rule():
    res = json.loads((C.HERE / "results" / "pilot_sealed_results.json").read_text(encoding="utf-8"))
    assert "P1_known_vocabulary_identification" in res and "P2_rejection_of_unsupported_plants" in res
    assert not any(k.lower().startswith("pooled") or k.lower().startswith("overall") for k in res)
    assert res["P1_known_vocabulary_identification"]["n_images"] == 70 and res["P2_rejection_of_unsupported_plants"]["n_images"] == 45
    for t, d in res["per_taxon"].items():
        assert d["percentages_shown"] == (d["n"] >= 10)
        assert d["percentages_shown"] or not any(k.endswith("_pct") for k in d), t
    assert "EXPLORATORY" in res["status"] and res["not_scored"]["quarantined_pending_expert"] == 2
    assert all(v is None for k, v in res["provisional_gates_reference_only"].items() if k != "note")   # no gate verdict recorded


def test_acquisition_log_declares_no_scoring():
    log = json.loads((C.HERE / "manifests" / "acquisition_log.json").read_text(encoding="utf-8"))
    assert "No image was scored" in log["note"] and "No identification API" in log["note"]


def _csv(name):
    import csv
    with open(C.HERE / "manifests" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_owner_review_decisions_applied():
    """Owner 2026-10-15: 3 excluded, 2 quarantined (not scored), 14 kept with caveats; nothing deleted."""
    flags = _csv("visual_review.csv")
    dec = [f["owner_decision"].split(" ")[0] for f in flags]
    assert dec.count("exclude") == 3 and dec.count("quarantine_pending_expert") == 2 and dec.count("keep_with_caveat") == 14
    excluded, quarantined = _csv("p1p2_review_excluded_manifest.csv"), _csv("p1p2_quarantined_manifest.csv")
    assert len(excluded) == 3 and len(quarantined) == 2
    scored = {r["stored_sha256"] for r in V.rows()}
    assert not ({r["stored_sha256"] for r in excluded + quarantined} & scored)
    assert {f["stored_sha256"] for f in flags if f["owner_decision"].startswith("keep")} <= scored
    assert (C.HERE / "manifests" / "previous_seals" / "seal_v1" / "SEAL.json").exists()      # v1 preserved


def test_seal_v2_and_p3_dropped():
    seal = json.loads((C.HERE / "SEAL.json").read_text(encoding="utf-8"))
    assert seal["version"] == 2 and seal["files"]["p3_sealed_manifest.csv"]["rows"] == 0 and "dropped" in seal["p3_status"]
    assert seal["files"]["p1p2_sealed_manifest.csv"]["rows"] == 115


def test_amendment_preserves_limitations_and_unchanged_settings():
    a = json.loads((C.HERE / "protocol_v1_amendment_1.json").read_text(encoding="utf-8"))
    text = " ".join(a["limitations_to_preserve"]).lower()
    for k in ("tomato and brinjal", "development images", "fewer than ten", "photographer identity", "re-encoded",
              "whole plants, flowers and fruits", "farmer-facing"):
        assert k in text, k
    assert "EXPLORATORY" in a["approval_scope"] and "vocabulary_v1.json (40 taxa)" in a["unchanged"]


def test_sealed_pilot_retired_and_preserved():
    """Owner 2026-10-15: preserve sealed predictions, manifests, protocol hashes and audit trail; never re-score or tune."""
    import hashlib
    rec = json.loads((C.HERE / "SEALED_PILOT_RETIRED.json").read_text(encoding="utf-8"))
    assert rec["status"].startswith("RETIRED") and rec["recorded_outcome"]["gate_verdict"] is None
    assert "6/45" in rec["recorded_outcome"]["P2_rejection_of_unsupported_plants"]
    for f, h in rec["preserved_files_sha256"].items():
        assert hashlib.sha256((C.HERE / f).read_bytes()).hexdigest() == h, f
    assert {"results/pilot_sealed_predictions.csv", "results/pilot_sealed_results.json", "SEALED_SCORING_APPROVAL.json",
            "protocol_v1_operating_point.json", "manifests/p1p2_sealed_manifest.csv"} <= set(rec["preserved_files_sha256"])
    lock = json.loads((C.HERE / "protocol_lock.json").read_text(encoding="utf-8"))
    for f, h in lock["files"].items():
        assert rec["preserved_files_sha256"][f] == h, f


def test_every_evaluation_result_reports_overlap_risk_per_source():
    """M4-19 criterion 1: every plant-identification evaluation result reports training-data overlap risk per source,
    and never claims independence from BioCLIP's training data."""
    reports = sorted(C.HERE.parent.rglob("*EVALUATION_RESULTS*.md"))
    assert reports, "no evaluation result reports found"
    for rep in reports:
        text = rep.read_text(encoding="utf-8")
        assert "Training-data overlap risk by source" in text, rep.name
        sec = text.split("Training-data overlap risk by source", 1)[1].split("\n## ", 1)[0]
        assert "Reduced overlap risk" in sec and "not** independent" in sec, rep.name
        res = json.loads((C.HERE / "results" / "pilot_sealed_results.json").read_text(encoding="utf-8"))
        assert "GBIF" in sec and "iNaturalist" in sec and "P3" in sec, rep.name     # every source of the scored blocks
        assert res["not_scored"]                                                   # P3 / excluded recorded as not scored
