"""Apply the owner's visual-review decisions (2026-10-15) and reseal. Run once:

  python -m stage10.plant_id.pilot.apply_review

* exclude (3) and quarantine-pending-expert (2) rows are MOVED out of the scored manifests into their own manifests,
  with all provenance kept; image files are kept on disk (nothing deleted);
* keep-with-caveat (14) rows stay in the scored manifests; their caveats live in visual_review.csv and are joined into
  every evaluation record;
* P3 is recorded as dropped from this pilot (deferred), its manifest stays empty;
* the v1 seal and v1 sealed manifest are archived under manifests/previous_seals/seal_v1/, and a v2 seal is written.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone

from stage10.plant_id.pilot import common as C
from stage10.plant_id.pilot.acquire import FIELDS

M = C.HERE / "manifests"
DECISION = {"recommend_exclude": "exclude", "expert_check": "quarantine_pending_expert", "keep_with_caveat": "keep_with_caveat"}
EXPECTED = {"exclude": 3, "quarantine_pending_expert": 2, "keep_with_caveat": 14}
OWNER = "owner 2026-10-15"
SCORED = ["p1p2_dev_manifest.csv", "p1p2_sealed_manifest.csv", "p3_sealed_manifest.csv"]
OUT_OF_SCORING = {"exclude": "p1p2_review_excluded_manifest.csv", "quarantine_pending_expert": "p1p2_quarantined_manifest.csv"}


def read(name):
    with open(M / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, rows, fields):
    with open(M / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    if (M / "previous_seals" / "seal_v1").exists():
        raise SystemExit("review already applied (seal v1 archived); refusing to apply twice")
    review = read("visual_review.csv")
    for r in review:
        r["owner_decision"] = f"{DECISION[r['recommendation']]} ({OWNER})"
    counts = {k: sum(r["owner_decision"].startswith(k) for r in review) for k in EXPECTED}
    if counts != EXPECTED:
        raise SystemExit(f"decision counts {counts} differ from the owner's instruction {EXPECTED}")
    move = {r["stored_sha256"]: DECISION[r["recommendation"]] for r in review if DECISION[r["recommendation"]] != "keep_with_caveat"}

    arch = M / "previous_seals" / "seal_v1"
    arch.mkdir(parents=True)
    shutil.copy2(C.HERE / "SEAL.json", arch / "SEAL.json")
    for n in SCORED:
        shutil.copy2(M / n, arch / n)

    moved = {k: [] for k in OUT_OF_SCORING}
    for n in SCORED:
        rows = read(n)
        keep = []
        for r in rows:
            d = move.get(r["stored_sha256"])
            if d:
                moved[d].append(dict(r, review_decision=f"{d} ({OWNER})"))
            else:
                keep.append(r)
        write(n, keep, FIELDS)
    if sum(len(v) for v in moved.values()) != len(move):
        raise SystemExit("not every excluded/quarantined image was found in the manifests")
    for d, n in OUT_OF_SCORING.items():
        write(n, sorted(moved[d], key=lambda r: (r["taxon"], r["record_id"])), FIELDS + ["review_decision"])
    write("visual_review.csv", review, list(review[0].keys()))

    v1 = json.loads((arch / "SEAL.json").read_text(encoding="utf-8"))
    seal = {"version": 2, "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "supersedes": {"version": 1, "sha256": C.sha256((arch / "SEAL.json").read_bytes()), "archived_at": "manifests/previous_seals/seal_v1/"},
            "rule": v1["rule"], "p3_status": "dropped from this pilot by owner decision 2026-10-15 (deferred; not completed)",
            "review": "owner decisions 2026-10-15: 3 excluded, 2 quarantined pending qualified expert, 14 kept with caveats (visual_review.csv)",
            "files": {}}
    for n in ["p1p2_sealed_manifest.csv", "p3_sealed_manifest.csv", *OUT_OF_SCORING.values(), "visual_review.csv"]:
        seal["files"][n] = {"rows": len(read(n)), "sha256": C.sha256((M / n).read_bytes())}
    (C.HERE / "SEAL.json").write_text(json.dumps(seal, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in seal.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
