"""Manual-results reconciler: synthetic result files only (written to tmp_path, never to the real results file)."""
from __future__ import annotations

import copy
import json

import pytest

from stage10.app import manual_results as MR

TPL = json.loads(MR.TEMPLATE.read_text(encoding="utf-8"))
META = {"tester_role": "project owner", "date": "2026-10-20", "device": "laptop", "os": "Windows 11", "browser": "Edge 140",
        "screen_reader": "Narrator (Windows 11)"}


def filled(pass_sections=(), fail=None, meta=True):
    r = copy.deepcopy(TPL)
    if meta:
        r.update(META)
    for k in r["results"]:
        if k.split("-")[0] in pass_sections:
            r["results"][k]["result"] = "pass"
    for k, note in (fail or {}).items():
        r["results"][k] = {"result": "fail", "notes": note}
    return r


def test_template_alone_changes_nothing():
    r = copy.deepcopy(TPL)
    assert MR.validate(r) == [] and MR.plan(r) == []


def test_validation_rejects_incomplete_or_unsafe_results():
    r = filled(("M1",), meta=False)
    assert any("tester_role" in e for e in MR.validate(r))
    r = filled(fail={"M1-3": ""})
    assert any("needs a note" in e for e in MR.validate(r))
    r = filled(("M4",))
    r["screen_reader"] = ""
    assert any("screen_reader" in e for e in MR.validate(r))
    r = filled(("M1",))
    r["results"]["M1-1"]["notes"] = "see photo.wav"
    assert any("recordings" in e for e in MR.validate(r))
    r = filled(("M1",))
    r["results"]["M1-1"]["result"] = "ok"
    assert MR.validate(r)


def test_partial_section_never_verifies():
    r = filled(("M1",))
    r["results"]["M1-6"]["result"] = "not_run"
    assert MR.section(r, "M1") == "incomplete" and not [c for c in MR.plan(r) if c[1] == "verified"]


def test_passes_map_to_the_right_tasks_and_never_to_the_acceptance_gate():
    r = filled(("M1", "M2", "M3", "M4", "M5"))
    plan = MR.plan(r)
    verified = {t for t, k, _ in plan if k == "verified"}
    assert verified == {"M5-A1", "M5-A2", "M5-A3", "M5-A4", "M5-A7"}
    assert ("M5-D5", "evidence") in {(t, k) for t, k, _ in plan}                 # D5 still needs M8-08 low-literacy testing
    assert not [c for c in plan if c[0] == "M5-R1" and c[1] == "verified"]
    t = MR.apply(r, plan)
    by = {x["task_id"]: x for x in t["tasks"]}
    assert by["M5-A1"]["status"] == by["M5-A7"]["status"] == "verified" and by["M5-A4"]["status"] == "verified"
    assert by["M5-D5"]["status"] != "verified" and by["M5-R1"]["status"] != "verified"


def test_a_failure_blocks_instead_of_verifying():
    r = filled(("M5",), fail={"M1-3": "preview stayed black"})
    plan = MR.plan(r)
    assert ("M5-A1", "failed_check") in {(t, k) for t, k, _ in plan}
    by = {x["task_id"]: x for x in MR.apply(r, plan)["tasks"]}
    assert by["M5-A1"]["status"] != "verified" and any("preview stayed black" in b for b in by["M5-A1"]["blockers"])
    assert by["M5-A7"]["status"] == "verified"


def test_real_results_file_if_present_is_valid():
    if not MR.RESULTS.exists():
        pytest.skip("owner has not recorded manual results yet")
    assert MR.validate(json.loads(MR.RESULTS.read_text(encoding="utf-8"))) == []
