"""Project tracker discipline: statuses backed by evidence, roadmap in sync, no premature 'verified' or 'supported' claims."""
from __future__ import annotations

import copy

import pytest

from stage10.tools.project_tracker import load, progress, render, roadmap_block_current, validate

T = load()


def test_tracker_is_valid_and_roadmap_in_sync():
    assert validate(T) == []
    assert roadmap_block_current(T)


def test_current_status_reconciled():
    p = progress(T)["milestones"]
    assert p["M1"]["computed_status"] == p["M2"]["computed_status"] == "complete"
    assert p["M3"]["computed_status"] == "complete"                      # gate closed 2026-10-14 (stage10/report_engine/M3_REVIEW.md)
    assert p["M4"]["computed_status"] == "complete"                      # owner accepted under SC-M4-03, 2026-10-15
    m4 = next(m for m in T["milestones"] if m["milestone_id"] == "M4")
    assert m4["approved_by_owner"] is True and any("does NOT solve or approve automatic plant identification" in k
                                                   for k in m4["known_limitations"])
    m4_verified = {x["task_id"] for x in T["tasks"] if x["milestone_id"] == "M4" and x["status"] == "verified"}
    assert m4_verified == {"M4-01", "M4-02", "M4-03", "M4-08", "M4-09", "M4-10", "M4-11", "M4-12", "M4-14", "M4-15", "M4-16", "M4-17",
                           "M4-21", "M4-22", "M4-23", "M4-24", "M4-25", "M4-26", "M4-28", "M4-30", "M4-31"}
    assert set(p["M4"]["pending_review"]) == set()              # M4-31 verified by the owner's recorded acceptance
    assert p["M4"]["blocked"] == []
    # SC-M4-03: automatic identification is no longer a required capability, but its protected gates keep their real status
    tasks = {x["task_id"]: x for x in T["tasks"]}
    assert {k: tasks[k]["status"] for k in ("M4-04", "M4-05", "M4-19")} == dict.fromkeys(("M4-04", "M4-05", "M4-19"), "blocked")
    assert {k: tasks[k]["status"] for k in ("M4-06", "M4-07", "M4-29")} == dict.fromkeys(("M4-06", "M4-07", "M4-29"), "implemented")
    assert all(tasks[k]["release_scope"] == "future_enhancement" and "SC-M4-03" in tasks[k]["next_action"]
               for k in ("M4-04", "M4-05", "M4-06", "M4-07", "M4-19", "M4-29"))
    moved = {x["task_id"]: x for x in T["tasks"] if x["task_id"] in ("M4-18", "M4-20", "M4-27")}
    assert all(x["status"] == "deferred" and x["release_scope"] == "future_enhancement" for x in moved.values())   # SC-M4-01/02
    carried = {x["task_id"]: x for x in T["tasks"] if x["task_id"] in ("M8-21", "M8-22")}
    assert set(carried) == {"M8-21", "M8-22"} and all(x["release_scope"] == "required" for x in carried.values())  # requirements preserved
    assert "M4-29" in next(x for x in T["tasks"] if x["task_id"] == "M4-04")["dependencies"]
    assert all(p[m]["verified"] == 0 for m in ("M6", "M7", "M8"))
    # M5 in progress (2026-10-15): only tasks with complete evidence are verified; privacy-gated and capability-gated
    # tasks are blocked, never verified
    m5 = {x["task_id"]: x["status"] for x in T["tasks"] if x["milestone_id"] == "M5"}
    assert p["M5"]["computed_status"] == "in_progress"
    assert {k for k, v in m5.items() if v == "verified"} == {"M5-D1", "M5-D2", "M5-D3", "M5-D4", "M5-C1", "M5-C3", "M5-B1", "M5-B2",
                                                            "M5-B3", "M5-B4", "M5-B5", "M5-E1", "M5-E2", "M5-E3", "M5-E4"}
    # hardware- or person-dependent tasks stay unverified until MANUAL_VERIFICATION results exist
    assert {k for k, v in m5.items() if v == "implemented"} == {"M5-A1", "M5-A2", "M5-A3", "M5-A4", "M5-A7", "M5-D5", "M5-C2"}
    assert {k for k, v in m5.items() if v == "blocked"} == {"M5-A5", "M5-A6", "M5-R1"}
    tasks0 = {x["task_id"]: x for x in T["tasks"]}
    assert tasks0["M5-R1"]["release_scope"] == "required" and p["M5"]["computed_status"] != "complete"
    # voice stays required (never downgraded); farmer marking is only proposed until the owner accepts SC-M5-01
    tasks = {x["task_id"]: x for x in T["tasks"]}
    assert all(tasks[k]["release_scope"] == "required" for k in ("M5-B2", "M5-B3", "M5-E1", "M5-E2"))
    # SC-M5-01 (owner-accepted 2026-10-15): farmer marking required; automatic localisation a future enhancement whose
    # status and criteria are unchanged; marking not verified until the manual touch check is done
    assert tasks["M5-A7"]["release_scope"] == "required" and m5["M5-A7"] == "implemented"
    for k in ("M5-A5", "M5-A6"):
        assert tasks[k]["release_scope"] == "future_enhancement" and m5[k] == "blocked" and "SC-M5-01" in tasks[k]["next_action"]
    assert "Localisation tested on annotated images" in tasks["M5-A6"]["acceptance_criteria"]
    assert m5["M5-A1"] != "verified"                      # camera capture not tested on a device


def test_capabilities_require_the_app_and_none_is_verified():
    caps = {c["capability_id"]: c["requires_tasks"] for c in T["capabilities"]}
    assert all("M5-D1" in caps[c] for c in ("CAP-01", "CAP-02", "CAP-03"))   # backend engine alone is not a user-facing feature


def test_no_language_claimed_supported_and_no_capability_verified():
    p = progress(T)
    assert all(lang[-1] == "NOT supported" for lang in p["languages"])
    # only the two capabilities whose every required task is verified (app shell + frozen-model / KB reports)
    assert {c[0] for c in p["capabilities"] if c[2] == "verified"} == {"CAP-02", "CAP-03"}
    caps = {c[0]: c[2] for c in p["capabilities"]}
    assert caps["CAP-04"] == "not available" and caps["CAP-12"] == "not available"
    # voice works locally, but the capability is not verified until people complete the microphone/speaker acceptance (M5-R1)
    assert caps["CAP-06"] != "verified" and caps["CAP-07"] != "verified"
    reqs = {c["capability_id"]: c["requires_tasks"] for c in T["capabilities"]}
    assert all("M5-R1" in reqs[c] for c in ("CAP-01", "CAP-06", "CAP-07"))
    assert "not application functionality" in render(T)


def _bad(mutate):
    t = copy.deepcopy(T)
    mutate(t)
    return validate(t)


def _task(t, tid):
    return next(x for x in t["tasks"] if x["task_id"] == tid)


@pytest.mark.parametrize("mutate,expect", [
    (lambda t: _task(t, "M4-05").update(status="verified"), "verified without evidence"),
    (lambda t: _task(t, "M4-05").update(status="verified", completion_date="2026-10-14"), "verified while dependency"),
    (lambda t: _task(t, "M4-05").update(completion_date="2026-10-14"), "completion_date set"),
    (lambda t: _task(t, "M4-05").update(status="implemented"), "implemented without any evidence"),
    (lambda t: _task(t, "M4-01").update(status="blocked"), "blocked without a stated blocker"),
    (lambda t: _task(t, "M4-01").update(dependencies=["M4-31"]), "dependency cycle"),
    (lambda t: _task(t, "M4-01").update(dependencies=["B-01"]), "backlog"),
    (lambda t: _task(t, "M4-01").update(acceptance_criteria=[]), "no acceptance criteria"),
    (lambda t: _task(t, "M1-01")["verification_evidence"].append({"type": "test", "ref": "stage10/tests/nope.py", "result": "x", "date": "2026-10-14"}), "missing"),
    (lambda t: t["milestones"][4].update(declared_status="complete"), "declared_status"),
    (lambda t: t["languages"][2]["features"]["speech_recognition"].update(status="verified"), "verified without evidence"),
    (lambda t: t["tasks"].append(copy.deepcopy(_task(t, "M1-01"))), "duplicate"),
])
def test_validator_rejects_bad_tracker_states(mutate, expect):
    assert any(expect in e for e in _bad(mutate)), _bad(mutate)


def test_deferred_required_tasks_are_reported_and_block_completion():
    """A deferred REQUIRED task is not 'remaining work' but is still unverified: it must be listed and block completion."""
    prog = progress(T)["milestones"]
    assert "M8-10" in prog["M8"]["deferred_required"] and "M8-10" in prog["M8"]["unmet_criteria"]
    assert prog["M8"]["computed_status"] != "complete"
    for mid, p in prog.items():
        unverified = [x["task_id"] for x in T["tasks"] if x["milestone_id"] == mid and x["release_scope"] == "required"
                      and x["status"] != "verified"]
        assert sorted(unverified) == sorted(p["remaining"] + p["deferred_required"]), mid
