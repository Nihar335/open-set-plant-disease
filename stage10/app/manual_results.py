"""Validate the owner's manual check results and reconcile the M5 tracker from them.

  python -m stage10.app.manual_results            # validate MANUAL_VERIFICATION_RESULTS.json and show what it would change
  python -m stage10.app.manual_results --apply    # write the resulting task updates to stage10/project_tracker.json

Rules (no task is ever verified without a recorded pass):
  * M1 all pass -> M5-A1 verified, then M5-A2/A3/A4 (their criteria already pass; they only waited on A1);
  * M5 all pass -> M5-A7 verified;
  * M4 all pass -> evidence for M5-D5, which STAYS implemented: its criteria include low-literacy testing (M8-08);
  * any failed step -> the task stays implemented and the failure is recorded as a blocker;
  * steps left "not_run" -> no change;
  * M5-R1 (acceptance gate) is never set by this tool: it needs the owner's explicit acceptance.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "MANUAL_VERIFICATION_RESULTS.json"
TEMPLATE = HERE / "MANUAL_VERIFICATION_RESULTS.template.json"
TRACKER = HERE.parent / "project_tracker.json"
CHECKS = {"M1": 6, "M2": 11, "M3": 6, "M4": 8, "M5": 7}
VALID = {"pass", "fail", "not_run"}
REQUIRED_META = ("tester_role", "date", "device", "os", "browser")


def validate(res: dict) -> list[str]:
    err = []
    tpl = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if set(res.get("results", {})) != set(tpl["results"]):
        err.append("results must contain exactly the template's check ids (M1-1 ... M5-7)")
    for k, v in res.get("results", {}).items():
        if not isinstance(v, dict) or v.get("result") not in VALID:
            err.append(f"{k}: result must be pass, fail or not_run")
        elif v["result"] == "fail" and not str(v.get("notes", "")).strip():
            err.append(f"{k}: a failed step needs a note saying what happened")
    ran = any(v.get("result") != "not_run" for v in res.get("results", {}).values() if isinstance(v, dict))
    if ran:
        for m in REQUIRED_META:
            if not str(res.get(m, "")).strip():
                err.append(f"{m} is required when any check was run")
        try:
            date.fromisoformat(str(res.get("date")))
        except ValueError:
            err.append("date must be YYYY-MM-DD")
    if ran and any(res["results"].get(f"M4-{i}", {}).get("result") != "not_run" for i in range(1, 9)) and not str(res.get("screen_reader", "")).strip():
        err.append("screen_reader (name and version) is required for M4")
    text = json.dumps(res, ensure_ascii=False).lower()
    if any(w in text for w in ("data:audio", "data:image", ".wav", ".webm", "base64,")):
        err.append("results must not contain recordings or images")
    return err


def section(res: dict, m: str) -> str:
    vals = [res["results"][f"{m}-{i}"]["result"] for i in range(1, CHECKS[m] + 1)]
    if all(v == "pass" for v in vals):
        return "pass"
    if any(v == "fail" for v in vals):
        return "fail"
    return "incomplete"


def plan(res: dict) -> list[tuple[str, str, str]]:
    """(task_id, new_status_or_'evidence', reason) for each change."""
    out = []
    s = {m: section(res, m) for m in CHECKS}
    who = f"{res.get('tester_role')} on {res.get('device')} / {res.get('browser')}, {res.get('date')}"
    if s["M1"] == "pass":
        out += [("M5-A1", "verified", f"MANUAL M1 camera capture passed ({who})")]
        out += [(t, "verified", "dependency M5-A1 verified; own criteria already passed") for t in ("M5-A2", "M5-A3", "M5-A4")]
    elif s["M1"] == "fail":
        out.append(("M5-A1", "failed_check", f"MANUAL M1 failed ({who})"))
    if s["M5"] == "pass":
        out.append(("M5-A7", "verified", f"MANUAL M5 touchscreen marking passed ({who})"))
    elif s["M5"] == "fail":
        out.append(("M5-A7", "failed_check", f"MANUAL M5 failed ({who})"))
    if s["M4"] == "pass":
        out.append(("M5-D5", "evidence", f"MANUAL M4 screen-reader check passed with {res.get('screen_reader')} ({who}); "
                                         "still needs low-literacy testing (M8-08)"))
    elif s["M4"] == "fail":
        out.append(("M5-D5", "failed_check", f"MANUAL M4 failed ({who})"))
    for m, name in (("M2", "microphone speech"), ("M3", "speaker output")):
        if s[m] in ("pass", "fail"):
            out.append(("M5-R1", "evidence", f"MANUAL {m} {name}: {s[m]} ({who})"))
    return out


def apply(res: dict, changes) -> dict:
    t = json.loads(TRACKER.read_text(encoding="utf-8"))
    by = {x["task_id"]: x for x in t["tasks"]}
    today = str(res["date"])
    fails = {k: v["notes"] for k, v in res["results"].items() if v["result"] == "fail"}
    for tid, kind, why in changes:
        x = by[tid]
        x["verification_evidence"].append({"type": "review", "ref": "stage10/app/MANUAL_VERIFICATION_RESULTS.json", "result": why, "date": today})
        if kind == "verified":
            deps_ok = all(by[d]["status"] == "verified" for d in x["dependencies"])
            if not deps_ok:
                continue                                          # tracker rule: dependencies first
            x.update(status="verified", completion_date=today, blockers=[], decisions_required=[], next_action="Verified. " + why)
        elif kind == "failed_check":
            prefix = tid.replace("M5-", "")
            x["blockers"] = sorted(set(x["blockers"]) | {f"Manual check failed: {k}: {v}" for k, v in fails.items()
                                                         if k.startswith({"A1": "M1", "A7": "M5", "D5": "M4"}.get(prefix, "?"))})
            x["next_action"] = "Fix the failed manual steps, then re-run the check."
    t["updated"] = today
    return t


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not RESULTS.exists():
        print(f"No results yet: copy {TEMPLATE.name} to {RESULTS.name} and fill it in (MANUAL_VERIFICATION.md).")
        sys.exit(1)
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    err = validate(res)
    for e in err:
        print("ERROR", e)
    if err:
        sys.exit(1)
    for m in CHECKS:
        print(f"{m}: {section(res, m)}")
    changes = plan(res)
    for c in changes:
        print("  ->", *c)
    if a.apply:
        TRACKER.write_text(json.dumps(apply(res, changes), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("tracker updated; run: python -m stage10.tools.project_tracker --write-roadmap")
    print("M5-R1 (acceptance) is not changed by this tool: it needs the owner's explicit acceptance.")


if __name__ == "__main__":
    main()
