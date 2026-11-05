"""Developer command-line entry point for the user-provided plant-name workflow (not the farmer-facing app: that is M5-D1).

  python -m stage10.workflow.cli --list-plants
  python -m stage10.workflow.cli --plant "tomato"                              # resolve only; shows what to confirm
  python -m stage10.workflow.cli --plant "tomato" --confirm --image photo.jpg  # confirmed name -> routed report (JSON)
  python -m stage10.workflow.cli --select PLANT-MANGO --confirm --image photo.jpg
  python -m stage10.workflow.cli --plant "Coconut" --keep-unlisted --confirm --image photo.jpg
  python -m stage10.workflow.cli --dont-know --confirm --image photo.jpg
      [--symptom FEAT-ID ...] [--absent FEAT-ID ...] [--state Kerala]

Nothing proceeds without --confirm (the user's explicit confirmation). Exit code 2 = clarification needed.
"""
from __future__ import annotations

import argparse
import json
import sys

from stage10.knowledge_base.kb import load_kb
from stage10.report_engine.inputs import Region, ReportInputError
from stage10.workflow.assessment import AssessmentError, assess_plant_health
from stage10.workflow.plant_name import PlantNameError, PlantNameSession, supported_plant_options


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-plants", action="store_true")
    g.add_argument("--plant")
    g.add_argument("--select")
    g.add_argument("--dont-know", action="store_true")
    ap.add_argument("--keep-unlisted", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--symptom", action="append", default=[])
    ap.add_argument("--absent", action="append", default=[])
    ap.add_argument("--state")
    a = ap.parse_args(argv)
    kb = load_kb()
    if a.list_plants:
        print(json.dumps(supported_plant_options(kb), indent=1, ensure_ascii=False))
        return 0
    s = PlantNameSession(kb)
    try:
        if a.plant is not None:
            res = s.enter(a.plant)
            print(json.dumps(res, indent=1, ensure_ascii=False))
            if res["status"] != "resolved":
                if not a.keep_unlisted:
                    return 2
                s.keep_unlisted_name()
        elif a.select:
            print(json.dumps(s.select(a.select), indent=1, ensure_ascii=False))
        else:
            s.dont_know()
        if not a.confirm:
            print("Not confirmed: re-run with --confirm to proceed with this plant name.")
            return 2
        choice = s.confirm()
    except PlantNameError as e:
        print(f"ERROR: {e}")
        return 2
    if not a.image:
        print("ERROR: --image is required for an assessment")
        return 2
    feats = {f: True for f in a.symptom} | {f: False for f in a.absent}
    try:
        out = assess_plant_health(choice, a.image, kb, reported_features=feats, region=Region(state=a.state))
    except (OSError, ReportInputError, AssessmentError) as e:
        print(f"ERROR: {e}")
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
