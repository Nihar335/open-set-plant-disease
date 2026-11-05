"""D2/C1 dataset approval summary (M4-29) and protected external gates (M4-19, M4-29, sealed scoring).

These tests guard against fabricated approvals: a task that needs an owner or legal decision may not be 'verified' unless
the owner-written / reviewer-written file exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from stage10.plant_id.dataset_approval import build_summary as BS
from stage10.tools import project_tracker as PT

PID = Path(__file__).resolve().parents[1] / "plant_id"


def test_dataset_summary_verifies_and_is_current():
    s, err = BS.build()
    assert err == [] and s["verification"] == "PASS"
    assert BS.OUT.read_text(encoding="utf-8") == json.dumps(s, indent=1, ensure_ascii=False) + "\n"
    c = s["counts"]
    assert c["development"]["images"] == 535 and c["test"]["images"] == 778
    assert c["test"]["pilot_photographer_images"] == 0
    assert set(c["licences_total"]) <= BS.LICENCES
    assert s["sealed"] == {"sealed_utc": s["sealed"]["sealed_utc"], "test_scored": False, "sealed_scoring_approval_present": False}
    assert s["owner_approval"].startswith("NOT RECORDED")


def test_summary_document_quotes_the_exact_hashes():
    s, _ = BS.build()
    md = (PID / "dataset_approval" / "DATASET_APPROVAL_SUMMARY.md").read_text(encoding="utf-8")
    for k in ("seal_sha256", "d2_dev_manifest.csv_sha256", "d2_test_manifest.csv_sha256", "dataset_manifest_pair_sha256",
              "test_stored_files_aggregate_sha256"):
        assert s["hashes"][k] in md, k


def test_no_gate_is_verified_without_its_owner_or_legal_record():
    tasks = {x["task_id"]: x for x in PT.load()["tasks"]}
    if not (PID / "dataset_approval" / "DATASET_APPROVAL_D2.json").exists():
        assert tasks["M4-29"]["status"] != "verified"
    if not (PID / "LEGAL_REVIEW_RESULT.md").exists():
        assert tasks["M4-19"]["status"] == "blocked" and tasks["M4-05"]["status"] != "verified"
    if not (PID / "d2" / "results" / "d2_sealed_results.json").exists():
        assert tasks["M4-04"]["status"] != "verified"
    # the identification adapter is never verified without the model approval gate
    assert tasks["M4-06"]["status"] != "verified" or tasks["M4-05"]["status"] == "verified"
    # routing (revised by SC-M4-03 to user-provided plant names): library tests alone never verify it; an end-to-end run
    # through the real frozen adapter must be on record
    m8 = tasks["M4-08"]
    if m8["status"] == "verified":
        assert any(e["type"] == "command" and "FrozenOSRAdapter" in e["result"] for e in m8["verification_evidence"])
        assert any(e["type"] == "test" for e in m8["verification_evidence"])
    assert tasks["M4-31"]["status"] != "verified" or any(m["milestone_id"] == "M4" and m["approved_by_owner"]
                                                         for m in PT.load()["milestones"])
    legal = (PID / "LEGAL_REVIEW_REQUIRED.md").read_text(encoding="utf-8")
    assert "No legal clearance exists or is claimed" in legal
