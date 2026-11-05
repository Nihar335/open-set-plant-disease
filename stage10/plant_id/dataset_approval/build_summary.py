"""Dataset approval summary for the acquired D2/C1 dataset (M4-29 criterion 4). Read-only over the sealed dataset.

  python -m stage10.plant_id.dataset_approval.build_summary          # verify + write dataset_approval_summary.json
  python -m stage10.plant_id.dataset_approval.build_summary --check  # verify only; fails if the written summary is stale

Independent of validate_d2.py: re-derives counts, licence categories, attribution completeness, photographer separation,
pilot-photographer placement, screening exclusions and the seal's stored-file aggregate from the manifests and the stored
bytes. Test images are hashed for integrity only; they are never decoded, viewed or scored. No model is loaded.
This file does NOT record owner approval: approval is a separate owner-written file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from stage10.plant_id.d2 import acquire_d2 as Q
from stage10.plant_id.d2 import seal_d2
from stage10.plant_id.pilot import common as PC
from stage10.plant_id.d2 import survey_availability as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "dataset_approval_summary.json"
STORE = Q.STORE.relative_to(ROOT).as_posix() + "/"
LICENCES = {"CC0 1.0", "CC BY 4.0", "CC BY-NC 4.0"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> tuple[dict, list[str]]:
    err = []
    seal = seal_d2.verify_seal()
    man = {k: Q.read_csv(Q.MAN / fn) for k, fn in seal_d2.FINAL.items()}
    excl = Q.read_csv(Q.MAN / "d2_screen_excluded_manifest.csv")
    quar = Q.read_csv(Q.MAN / "d2_quarantined_manifest.csv")
    dec = {r["plan_id"]: r for r in Q.read_csv(Q.MAN / "screen_decisions.csv")}
    rej = Q.read_csv(Q.MAN / "d2_rejections.csv")
    _, pilot_phot = S.pilot_exclusions()

    for split, rows in man.items():
        for r in rows:
            pid = r["plan_id"]
            if r["split"] != split:
                err.append(f"{pid}: split column {r['split']} in {split} manifest")
            if r["media_licence"] not in LICENCES:
                err.append(f"{pid}: licence {r['media_licence']}")
            if (r["nc_restricted"] == "True") != (r["media_licence"] == "CC BY-NC 4.0"):
                err.append(f"{pid}: NC flag disagrees with licence")
            a = r["attribution"]
            if not (r["photographer_display"] and r["observation_url"] in a and r["media_licence"] in a and r["gbif_id"] in a):
                err.append(f"{pid}: incomplete attribution")
            if not r["licence_verified_utc"]:
                err.append(f"{pid}: licence not verified")
            if not r["stored_path"].startswith(STORE + f"images/{split}/"):
                err.append(f"{pid}: stored outside {STORE}images/{split}/")
            f = ROOT / r["stored_path"]
            if not f.exists() or sha(f) != r["stored_sha256"]:
                err.append(f"{pid}: stored bytes differ from manifest hash")
            if dec.get(pid, {}).get("decision") not in ("keep", "keep_with_caveat"):
                err.append(f"{pid}: not a kept screen decision")
            if r["media_created"] < "2025-01-01" or r["event_date"] < "2025-01-01":
                err.append(f"{pid}: dated before 2025-01-01")
    ids = {k: {r["plan_id"] for r in v} for k, v in man.items()}
    gb = {k: {r["gbif_id"] for r in v} for k, v in man.items()}
    if ids["dev"] & ids["test"] or gb["dev"] & gb["test"]:
        err.append("an image is in both splits")
    for key in ("photographer_key", "inat_login"):
        both = {r[key] for r in man["dev"]} & {r[key] for r in man["test"]}
        if both:
            err.append(f"{key} in both splits: {sorted(both)[:3]}")
    test_pilot = [r["plan_id"] for r in man["test"] if PC.normalise_photographer(r["photographer_display"]) in pilot_phot]
    if test_pilot:
        err.append(f"pilot photographer in test: {test_pilot[:3]}")
    for r in excl + quar:
        if r["plan_id"] in ids["dev"] | ids["test"]:
            err.append(f"{r['plan_id']}: excluded/quarantined image in a final manifest")
    agg = hashlib.sha256("".join(sorted(sha(ROOT / r["stored_path"]) for r in man["test"])).encode()).hexdigest()
    if agg != seal["test_stored_files_sha256"]:
        err.append("sealed test stored-file aggregate does not match the bytes on disk")
    if len(man["test"]) != seal["test_rows"] or len(man["dev"]) != seal["dev_rows"]:
        err.append("row counts differ from the seal")

    def counts(rows):
        return {"images": len(rows), "photographer_clusters": len({r["photographer_key"] for r in rows}),
                "by_study_role": dict(sorted(Counter(f"{r['study']}/{r['role']}" for r in rows).items())),
                "licences": dict(sorted(Counter(r["media_licence"] for r in rows).items())),
                "kept_with_caveat": sum(dec[r["plan_id"]]["decision"] == "keep_with_caveat" for r in rows),
                "pilot_photographer_images": sum(PC.normalise_photographer(r["photographer_display"]) in pilot_phot for r in rows)}

    allrows = man["dev"] + man["test"]
    summary = {
        "dataset": "AgroVaidya M4 D2 look-alike + C1 coconut/areca internal non-commercial evaluation set",
        "verification": "PASS" if not err else "FAIL",
        "errors": err,
        "hashes": {"seal_sha256": sha(Q.SEAL), "protocol_v2_lock_sha256": seal["protocol_v2_lock_sha256"],
                   **{Path(k).name + "_sha256": v for k, v in seal["files"].items()},
                   "test_stored_files_aggregate_sha256": agg,
                   "dataset_manifest_pair_sha256": hashlib.sha256((seal["files"]["stage10/plant_id/d2/manifests/d2_dev_manifest.csv"] +
                                                                    seal["files"]["stage10/plant_id/d2/manifests/d2_test_manifest.csv"]).encode()).hexdigest()},
        "sealed": {"sealed_utc": seal["created_utc"], "test_scored": (Q.MAN.parent / "results" / "d2_sealed_results.json").exists(),
                   "sealed_scoring_approval_present": (Q.MAN.parent / "SEALED_SCORING_APPROVAL_D2.json").exists()},
        "counts": {"development": counts(man["dev"]), "test": counts(man["test"]),
                   "total_final_images": len(allrows),
                   "licences_total": dict(sorted(Counter(r["media_licence"] for r in allrows).items())),
                   "nc_share": round(sum(r["media_licence"] == "CC BY-NC 4.0" for r in allrows) / len(allrows), 4)},
        "exclusions": {"acquisition_gate_rejections": len(rej),
                       "gate_rejection_reasons": dict(sorted(Counter(r.get("reason", "")[:40] for r in rej).items())),
                       "screen_excluded": len(excl),
                       "screen_excluded_flags": dict(sorted(Counter(dec[r["plan_id"]]["flag"] for r in excl).items())),
                       "quarantined_unscored": len(quar),
                       "quarantine_flags": dict(sorted(Counter(dec[r["plan_id"]]["flag"] for r in quar).items())),
                       "pilot_images_used": 0,
                       "pilot_photographers": "development only; none in test (checked)"},
        "intended_use": "internal, non-commercial research evaluation only (owner, 2026-10-15)",
        "owner_approval": "NOT RECORDED: requires an owner-written approval; this summary is not approval",
    }
    return summary, err


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    s, err = build()
    text = json.dumps(s, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            err.append("dataset_approval_summary.json is missing or stale")
    else:
        OUT.write_text(text, encoding="utf-8")
    for e in err[:20]:
        print("ERROR", e)
    print(f"DATASET APPROVAL SUMMARY CHECK: {'PASS' if not err else 'FAIL'} ({len(err)} errors)")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
