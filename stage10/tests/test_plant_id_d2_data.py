"""Acquired D2/C1 data under frozen protocol v2: lock, licence evidence, independent validation, seal, development-only
operating point, and the internal non-commercial restrictions. Skips cleanly where a stage has not happened yet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stage10.plant_id.d2 import acquire_d2 as Q
from stage10.plant_id.d2 import evaluate_d2 as EV

ROOT = Path(__file__).resolve().parents[2]
LICENCE_DIR = ROOT / "stage10_local" / "d2_licence_sources"
HAVE_ACQ = (Q.MAN / "d2_acquired.csv").exists()


def test_protocol_v2_is_frozen_and_lock_verifies():
    lock = Q.verify_lock()
    assert "stage10/plant_id/d2/PROTOCOL_V2.md" in lock["files"] and "stage10/plant_id/d2/THRESHOLD_SELECTION_METHOD.md" in lock["files"]
    proto = json.loads((Q.HERE / "protocol_v2.json").read_text(encoding="utf-8"))
    assert proto["status"] == "FROZEN" and proto["intended_use"] == "internal, non-commercial research evaluation only"
    assert proto["threshold_method"]["target_false_acceptance"] == 0.10 and "NOT authorized" in proto["sealed_scoring"]


def test_licence_and_access_terms_evidence_is_cached_and_hash_recorded():
    if not LICENCE_DIR.exists():
        pytest.skip("licence sources not cached on this machine")
    sums = dict(reversed(l.split("  ")) for l in (LICENCE_DIR / "SHA256SUMS.txt").read_text().splitlines() if l)
    for name in ("cc_by_nc_4_0_legalcode.html", "cc_by_4_0_legalcode.html", "cc0_1_0_legalcode.html", "inat_open_data_README.md",
                 "gbif_dataset_inat_rg.json", "gbif_docs_publisher_visibility.html"):
        assert hashlib.sha256((LICENCE_DIR / name).read_bytes()).hexdigest() == sums[name], name
    review = (Q.HERE / "LICENCE_AND_INTENDED_USE_REVIEW.md").read_text(encoding="utf-8")
    assert "internal, non-commercial research" in review and "No legal clearance for any other use is claimed" in review
    assert "Residual dependency" in review                      # GBIF DUA full text not read (bot challenge, not bypassed)


@pytest.mark.skipif(not HAVE_ACQ, reason="nothing acquired")
def test_acquired_data_passes_independent_validation_once_sealed():
    if not Q.SEAL.exists():
        pytest.skip("not sealed yet")
    from stage10.plant_id.d2 import validate_d2
    err, summary = validate_d2.validate()
    assert err == [], err[:10]
    assert summary["test"] > 0 and summary["dev"] > 0


@pytest.mark.skipif(not HAVE_ACQ, reason="nothing acquired")
def test_internal_nc_store_is_separate_and_never_referenced_by_app_or_training_code():
    assert (Q.STORE / "NOTICE_INTERNAL_NC_EVALUATION_ONLY.txt").exists()
    # evaluation-only readers; evaluate_v3.py is the protocol-v3 development scorer (hash-locked by its pre-registration; it
    # refuses any test path, see test_plant_id_protocol_v3.py)
    allowed = {"acquire_d2.py", "seal_d2.py", "validate_d2.py", "evaluate_d2.py", "evaluate_v3.py", "test_plant_id_d2_data.py"}
    offenders = []
    for d in (ROOT / "stage10", ROOT / "demo", ROOT / "scripts", ROOT / "src"):
        for p in d.rglob("*.py"):
            if p.name not in allowed and "d2_internal_nc_eval" in p.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(p.relative_to(ROOT)))
    assert offenders == []
    asset = json.loads((ROOT / "stage10" / "ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    assert not any("d2_internal_nc_eval" in k for k in asset["stage10_files"])       # never in the distributable manifest


def test_operating_point_if_present_is_development_only_and_after_the_seal():
    if not EV.OP.exists():
        pytest.skip("development phase not run")
    op = json.loads(EV.OP.read_text(encoding="utf-8"))
    seal = json.loads(Q.SEAL.read_text(encoding="utf-8"))
    assert op["data"] == "development only" and op["sealed_test_read"] is False
    assert op["seal_sha256"] == hashlib.sha256(Q.SEAL.read_bytes()).hexdigest() and seal["created_utc"] <= op["created_utc"]
    assert op["status"] in ("resolved", "unresolved") and (op["status"] == "resolved") == (op["tau"] is not None)
    if op["status"] == "unresolved":
        assert op["reason"]                                       # reported, never silently relaxed


def test_fresh_sealed_test_set_has_not_been_scored_without_authorisation():
    if not EV.APPROVAL.exists():
        assert not (EV.RESULTS / "d2_sealed_results.json").exists()
