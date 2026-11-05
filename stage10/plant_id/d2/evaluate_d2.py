"""Protocol v2 evaluation. Run ONLY with the isolated environment:

  C:\\Users\\nihar\\venvs\\agrovaidya-plantid\\Scripts\\python.exe -m stage10.plant_id.d2.evaluate_d2 --phase dev
  C:\\Users\\nihar\\venvs\\agrovaidya-plantid\\Scripts\\python.exe -m stage10.plant_id.d2.evaluate_d2 --phase sealed

--phase dev    requires a verified SEAL_D2.json; reads ONLY d2_dev_manifest.csv; selects tau with thresholds.run() exactly
               as THRESHOLD_SELECTION_METHOD.md; writes protocol_v2_operating_point.json (resolved or unresolved; never
               overwritten). No test image is read.
--phase sealed requires SEALED_SCORING_APPROVAL_D2.json (owner, hash-bound to lock, seal and operating point) and a resolved
               operating point; scores every test image ONCE; refuses if results exist. NOT authorised as of 2026-10-15.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAN = HERE / "manifests"
OP = HERE / "protocol_v2_operating_point.json"
APPROVAL = HERE / "SEALED_SCORING_APPROVAL_D2.json"
RESULTS = HERE / "results"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name: str) -> list[dict]:
    with open(MAN / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_rows(rows: list[dict], device: str) -> list[dict]:
    from stage10.plant_id.pilot import evaluate_pilot as E
    sc = E.Scorer("T1", device)
    out = []
    for r in rows:
        top5, ms = sc.score(ROOT / r["stored_path"])
        out.append({"plan_id": r["plan_id"], "taxon": r["taxon"], "group": r["group"], "role": r["role"], "study": r["study"],
                    "photographer": r["photographer_key"], "top1": top5[0][0], "score": top5[0][1],
                    "top5": [t for t, _ in top5], "ms": ms, "caveat": r.get("screen_caveat", "")})
    return out


def phase_dev(device: str):
    from stage10.plant_id.d2 import seal_d2, thresholds as T
    seal = seal_d2.verify_seal()
    if OP.exists():
        raise SystemExit("operating point already written; refusing to overwrite")
    rows = read(seal_d2.FINAL["dev"])                      # development manifest ONLY
    assert all(r["split"] == "dev" for r in rows)
    scored = score_rows(rows, device)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "d2_dev_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["plan_id", "taxon", "group", "role", "study", "photographer", "top1", "score", "top5", "ms", "caveat"])
        w.writeheader()
        w.writerows({**s, "top5": "|".join(s["top5"])} for s in scored)
    uns = [{"score": s["score"], "photographer": s["photographer"], "group": s["group"], "taxon": s["taxon"]}
           for s in scored if s["role"] == "unsupported"]
    sup = [{"score": s["score"], "photographer": s["photographer"], "taxon": s["taxon"], "correct": s["top1"] == s["taxon"]}
           for s in scored if s["role"] != "unsupported"]
    res = T.run(uns, sup)
    ms = sorted(s["ms"] for s in scored)
    op = {"protocol": "v2", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": "development only",
          "sealed_test_read": False, "seal_sha256": sha(seal_d2.Q.SEAL), "dev_manifest_sha256": sha(MAN / seal_d2.FINAL["dev"]),
          "dev_scores_sha256": sha(RESULTS / "d2_dev_scores.csv"), "template": "T1", "device": device,
          "n_unsupported": len(uns), "n_supported": len(sup),
          "latency_ms": {"p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))]},
          **{k: v for k, v in res.items()}}
    op["tau"] = res.get("tau") if res["status"] == "resolved" else None
    OP.write_text(json.dumps(op, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"operating point: {op['status']}; tau={op['tau']}; reason={op.get('reason')}")


def phase_sealed(device: str):
    from stage10.plant_id.d2 import analysis as AN, seal_d2
    if not APPROVAL.exists():
        raise SystemExit("SEALED_SCORING_APPROVAL_D2.json missing: sealed scoring is not authorised")
    ap = json.loads(APPROVAL.read_text(encoding="utf-8"))
    need = {"approved_by_owner": True, "protocol_v2_lock_sha256": sha(HERE / "protocol_v2_lock.json"),
            "seal_sha256": sha(seal_d2.Q.SEAL), "operating_point_sha256": sha(OP) if OP.exists() else None}
    for k, v in need.items():
        if ap.get(k) != v:
            raise SystemExit(f"approval does not match current {k}: refusing to score")
    if not ap.get("date"):
        raise SystemExit("approval has no date")
    seal_d2.verify_seal()
    op = json.loads(OP.read_text(encoding="utf-8"))
    if op["status"] != "resolved":
        raise SystemExit("operating point unresolved: nothing to score")
    out = RESULTS / "d2_sealed_results.json"
    if out.exists():
        raise SystemExit("sealed results exist: one pass only")
    tau = op["tau"]
    scored = score_rows(read(seal_d2.FINAL["test"]), device)
    uns = [s for s in scored if s["role"] == "unsupported"]
    sup = [s for s in scored if s["role"] == "supported_control"]
    c1 = [s for s in scored if s["study"] == "C1"]
    ms = sorted(s["ms"] for s in scored)
    res = {"status": "SEALED PROTOCOL-V2 RESULTS", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tau": tau, "approval_sha256": sha(APPROVAL),
           "D2_unsupported_rejection": AN.unsupported_outcome(uns, tau), "D2_supported_identification": AN.supported_outcome(sup, tau),
           "C1": AN.c1_outcome(c1, tau), "per_group": AN.by_key(uns, "group", tau), "per_taxon_unsupported": AN.by_key(uns, "taxon", tau),
           "latency_ms": {"p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))]},
           "overlap_statement": "Reduced overlap risk by date and source controls only; not independent of BioCLIP training data."}
    out.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("D2_unsupported_rejection", "D2_supported_identification")}, indent=1, default=str)[:2000])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "sealed"], required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    phase_dev(a.device) if a.phase == "dev" else phase_sealed(a.device)


if __name__ == "__main__":
    main()
