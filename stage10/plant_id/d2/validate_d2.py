"""Independent validator for ACQUIRED D2/C1 data (protocol v2). Re-derives every rule from the files themselves.

  python -m stage10.plant_id.d2.validate_d2            # -> manifests/d2_validation_report.json; exit 1 on any failure

Checks: protocol lock; every acquired row is a frozen-manifest row with unchanged metadata and passed live re-verification;
licence allow-list and NC flag; no pilot image and no pilot photographer in test; login and display-name separation;
both caps on the final manifests; stored files exist and match SHA-256; no exact Stage 1-9 MD5; perceptual near-duplicate
re-check of every stored copy against the read-only Stage 1-9 index; dHash duplicates within D2 and against the pilot;
every acquired image screened; excluded/quarantined images absent from the final manifests; seal (if present) verifies;
storage outside the frozen research tree with the internal-use notice present.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter

import numpy as np

from stage10.plant_id import neardup as N
from stage10.plant_id.d2 import acquire_d2 as Q
from stage10.plant_id.d2 import overlap_gate as G
from stage10.plant_id.d2 import plan_manifest as P
from stage10.plant_id.d2 import survey_availability as S
from stage10.plant_id.pilot import common as PC

SAME = ["study", "group", "role", "taxon", "gbif_taxon_key", "split", "status", "rank", "gbif_id", "observation_url", "photographer_key",
        "inat_login", "photographer_display", "media_licence", "media_licence_uri", "record_licence_uri", "media_identifier"]


def validate() -> tuple[list[str], dict]:
    err = []
    try:
        Q.verify_lock()
    except Q.AcquisitionStop as e:
        err.append(f"lock: {e}")
    cfg = P.config()
    plan = {r["plan_id"]: r for r in Q.frozen_rows()}
    acq = Q.read_csv(Q.MAN / "d2_acquired.csv")
    rv = {r["plan_id"]: r["status"] for r in Q.read_csv(Q.MAN / "reverification.csv")}
    dec = {r["plan_id"]: r["decision"] for r in Q.read_csv(Q.MAN / "screen_decisions.csv")}
    ids, phot = S.pilot_exclusions()
    allowed = {u.rstrip("/") + "/" for u in cfg["licences"]["allowed_media_licence_uris"]}
    md5s = PC.prohibited_md5()
    for a in acq:
        pid = a["plan_id"]
        p = plan.get(pid)
        if p is None or any(a[k] != p[k] for k in SAME):
            err.append(f"{pid}: not an unchanged frozen-manifest row")
        if rv.get(pid) != "ok":
            err.append(f"{pid}: live re-verification {rv.get(pid)}")
        if a["media_licence_uri"].rstrip("/") + "/" not in allowed or (a["nc_restricted"] == "True") != (a["media_licence"] == "CC BY-NC 4.0"):
            err.append(f"{pid}: licence")
        if a["gbif_id"] in ids:
            err.append(f"{pid}: pilot image")
        if a["split"] == "test" and PC.normalise_photographer(a["photographer_display"]) in phot:
            err.append(f"{pid}: pilot photographer in test")
        if a["original_md5"].lower() in md5s:
            err.append(f"{pid}: Stage 1-9 MD5")
        f = Q.ROOT / a["stored_path"]
        if not f.exists() or PC.sha256(f.read_bytes()) != a["stored_sha256"]:
            err.append(f"{pid}: stored file missing or changed")
        if not a["stored_path"].startswith("stage10_local/d2_internal_nc_eval/"):
            err.append(f"{pid}: stored outside the internal NC store")
        if pid not in dec:
            err.append(f"{pid}: not screened")
    # separation by login and display name across splits (all acquired rows)
    for key in ("inat_login", "photographer_display"):
        sp = {}
        for a in acq:
            k = PC.normalise_photographer(a[key])
            if sp.setdefault(k, a["split"]) != a["split"]:
                err.append(f"{key} {k!r} in both splits")
    # caps on acquired rows (superset of the final manifests)
    for k, v in Counter((a["split"], a["taxon"], a["photographer_key"]) for a in acq).items():
        if v > cfg["caps"]["per_photographer_per_taxon"]:
            err.append(f"cap per photographer per taxon exceeded: {k}")
    for k, v in Counter((a["split"], a["photographer_key"]) for a in acq).items():
        if v > cfg["caps"]["per_photographer_per_split"]:
            err.append(f"cap per photographer per split exceeded: {k}")
    # duplicates within D2 and against the pilot (stored copies, 8 orientations)
    from PIL import Image
    seen, pilot = [], G.pilot_dhashes()
    paths, D, Pp, td, tp = G.stage1_9_index()
    flagged = []
    for a in acq:
        im = Image.open(Q.ROOT / a["stored_path"]).convert("RGB")
        hs = PC.dihedral_hashes(im)
        d = PC.near_duplicate(hs, seen, G.WITHIN_MAX_BITS)
        if d:
            err.append(f"{a['plan_id']}: near-duplicate of {d} within D2")
        if PC.near_duplicate(hs, pilot, G.WITHIN_MAX_BITS):
            err.append(f"{a['plan_id']}: near-duplicate of a pilot image")
        seen.append((hs[0], a["plan_id"]))
        qd, qp = N.query_hashes(im)
        hit = np.where((N.min_distances(qd, D) <= td) | (N.min_distances(qp, Pp) <= tp))[0]
        if hit.size:
            flagged.append({"plan_id": a["plan_id"], "stage1_9": [paths[i] for i in hit[:3]]})
    err += [f"{x['plan_id']}: perceptual match to Stage 1-9 {x['stage1_9']}" for x in flagged]
    # final manifests and seal
    summary = {"acquired": len(acq), "screen": dict(Counter(dec.values()))}
    if Q.SEAL.exists():
        from stage10.plant_id.d2 import seal_d2
        try:
            seal_d2.verify_seal()
        except Q.AcquisitionStop as e:
            err.append(f"seal: {e}")
        for split, fn in seal_d2.FINAL.items():
            rows = Q.read_csv(Q.MAN / fn)
            bad = [r["plan_id"] for r in rows if dec.get(r["plan_id"]) not in ("keep", "keep_with_caveat")]
            if bad:
                err.append(f"{fn}: excluded/quarantined images present {bad[:3]}")
            summary[split] = len(rows)
    if not (Q.STORE / "NOTICE_INTERNAL_NC_EVALUATION_ONLY.txt").exists():
        err.append("internal-use notice missing")
    summary["errors"] = len(err)
    return err, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    err, summary = validate()
    (Q.MAN / "d2_validation_report.json").write_text(json.dumps({"errors": err, "summary": summary}, indent=1) + "\n", encoding="utf-8")
    for e in err[:30]:
        print("  FAIL", e)
    print(json.dumps(summary))
    print("D2 DATA VALIDATION:", "PASS" if not err else "FAIL")
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()
