"""Independent post-acquisition validation of the pilot (read-only). Exit code 1 on any failure.

  python -P -m stage10.plant_id.pilot.validate_pilot          (or without -P if the root is not importable)

Checks: licence/attribution per row; block/taxon/vocabulary consistency; P3 sealed-only; photographer-disjoint P1/P2 splits;
per-photographer caps; observation date cut-off; no exact MD5 match with ANY Stage 1-9 image (original and stored copy);
no duplicate or near-duplicate (8 rotations/reflections) across the whole pilot; stored files exist and match their hashes;
SEAL.json matches the sealed manifests; protocol/vocabulary lock matches. Writes validation_report.json with counts.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict

from stage10.plant_id.pilot import common as C

M = C.HERE / "manifests"
NAMES = {"p1p2_dev_manifest.csv": "dev", "p1p2_sealed_manifest.csv": "sealed", "p3_sealed_manifest.csv": "sealed"}
LOCK = C.HERE / "protocol_lock.json"
REVIEW_MANIFESTS = ["p1p2_review_excluded_manifest.csv", "p1p2_quarantined_manifest.csv"]


def rows():
    out = []
    for name, split in NAMES.items():
        with open(M / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_file"] = name
                out.append(r)
    return out


def validate(check_files: bool = True) -> tuple[list[str], dict]:
    cfg = C.config()
    err: list[str] = []
    rs = rows()
    vocab = {t["scientific_name"] for t in json.loads((C.HERE / "vocabulary_v1.json").read_text(encoding="utf-8"))["taxa"]}
    bad = C.prohibited_md5()
    allowed_uri = cfg["gbif"]["media_licence_uris"] + ["http://creativecommons.org/licenses/by/4.0/"]
    for r in rs:
        rid = f"{r['block']}/{r['taxon']}/{r['record_id']}"
        if r["licence"] not in ("CC0 1.0", "CC BY 4.0") or not C.licence_ok(r["licence_uri"], allowed_uri):
            err.append(f"{rid}: licence not allowed ({r['licence']}, {r['licence_uri']})")
        if len(r["attribution"]) < 30 or "Changes:" not in r["attribution"]:
            err.append(f"{rid}: attribution incomplete")
        expected_split = NAMES[r["_file"]]
        if r["split"] != expected_split:
            err.append(f"{rid}: split column {r['split']} != manifest {r['_file']}")
        if r["block"] == "P3" and (r["split"] != "sealed" or r["_file"] != "p3_sealed_manifest.csv"):
            err.append(f"{rid}: P3 must be sealed/test-only")
        if r["block"] in ("P1", "P3") and r["taxon"] not in vocab:
            err.append(f"{rid}: in-vocabulary block taxon not in vocabulary")
        if r["block"] == "P2" and r["taxon"] in vocab:
            err.append(f"{rid}: unsupported taxon is in the vocabulary")
        if r["taxon"] not in {**cfg["blocks"]["P1"]["taxa"], **cfg["blocks"]["P2"]["taxa"], **cfg["blocks"]["P3"]["datasets"]}:
            err.append(f"{rid}: taxon not approved")
        if r["block"] in ("P1", "P2") and not (r["observed_on"] and r["observed_on"] >= "2025-01-01"):
            err.append(f"{rid}: observation date before cut-off or missing")
        if r["original_md5"] in bad:
            err.append(f"{rid}: original matches a Stage 1-9 image (exact MD5)")
        if check_files:
            p = C.ROOT / r["stored_path"]
            if not p.exists():
                err.append(f"{rid}: stored file missing")
                continue
            b = p.read_bytes()
            if C.sha256(b) != r["stored_sha256"]:
                err.append(f"{rid}: stored file hash mismatch")
            if C.md5(b) in bad:
                err.append(f"{rid}: stored copy matches a Stage 1-9 image (exact MD5)")
    # photographer disjointness and caps (P1/P2)
    split_of = defaultdict(set)
    for r in rs:
        if r["block"] in ("P1", "P2"):
            split_of[r["photographer_key"]].add(r["split"])
            if not r["photographer_key"]:
                err.append(f"{r['record_id']}: missing photographer")
    for k, s in split_of.items():
        if len(s) > 1:
            err.append(f"photographer {k!r} appears in both dev and sealed")
    cap = Counter((r["block"], r["taxon"], r["photographer_key"]) for r in rs if r["block"] in ("P1", "P2"))
    for k, n in cap.items():
        if n > cfg["gbif"]["max_images_per_photographer_per_taxon"]:
            err.append(f"photographer cap exceeded: {k} = {n}")
    # duplicates
    for field in ("original_sha256", "stored_sha256", "record_id"):
        c = Counter((r["block"] if field == "record_id" else "", r[field]) for r in rs)
        for k, n in c.items():
            if n > 1:
                err.append(f"duplicate {field}: {k[1]} x{n}")
    if check_files:
        from PIL import Image
        hashes = []
        for r in rs:
            p = C.ROOT / r["stored_path"]
            if p.exists():
                with Image.open(p) as im:
                    hashes.append((r, C.dihedral_hashes(im.convert("RGB"))))
        md = cfg["near_duplicate_dhash_max_distance"]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if min(C.hamming(hashes[i][1][0], h) for h in hashes[j][1]) <= md:
                    err.append(f"near-duplicate pair: {hashes[i][0]['record_id']} ~ {hashes[j][0]['record_id']}")
    # seal and protocol lock
    seal = json.loads((C.HERE / "SEAL.json").read_text(encoding="utf-8"))
    for name, rec in seal["files"].items():
        if C.sha256((M / name).read_bytes()) != rec["sha256"]:
            err.append(f"SEAL mismatch for {name}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for name, h in lock["files"].items():
        if C.sha256((C.HERE / name).read_bytes()) != h:
            err.append(f"protocol lock mismatch for {name}")
    # owner review decisions (2026-10-15): excluded/quarantined rows are out of scoring but preserved
    scored = {r["stored_sha256"] for r in rs}
    review_rows = []
    for name in REVIEW_MANIFESTS:
        if (M / name).exists():
            with open(M / name, newline="", encoding="utf-8") as f:
                review_rows += [dict(r, _file=name) for r in csv.DictReader(f)]
    for r in review_rows:
        if r["stored_sha256"] in scored:
            err.append(f"{r['record_id']}: {r['_file']} image is still in a scored manifest")
        if check_files:
            p = C.ROOT / r["stored_path"]
            if not p.exists() or C.sha256(p.read_bytes()) != r["stored_sha256"]:
                err.append(f"{r['record_id']}: preserved file for {r['_file']} missing or altered")
    vr = M / "visual_review.csv"
    if vr.exists():
        with open(vr, newline="", encoding="utf-8") as f:
            flags = list(csv.DictReader(f))
        out = {r["stored_sha256"] for r in review_rows}
        for fl in flags:
            d = fl["owner_decision"]
            if d == "pending":
                err.append(f"visual review flag {fl['record_id']} has no owner decision")
            elif d.startswith("keep_with_caveat") and fl["stored_sha256"] not in scored:
                err.append(f"keep-with-caveat image {fl['record_id']} missing from scored manifests")
            elif (d.startswith("exclude") or d.startswith("quarantine")) and fl["stored_sha256"] not in out:
                err.append(f"{d.split(' ')[0]} image {fl['record_id']} not in its review manifest")
    if C.APPROVAL.exists():
        a = json.loads(C.APPROVAL.read_text(encoding="utf-8"))
        if a.get("protocol_sha256") != lock["files"]["protocol_v1.json"] or a.get("amendment_sha256") != lock["files"].get("protocol_v1_amendment_1.json")                 or a.get("seal_sha256") != C.sha256((C.HERE / "SEAL.json").read_bytes()):
            err.append("SEALED_SCORING_APPROVAL.json does not reference the current protocol, amendment and seal")
    summary = {
        "rows": len(rs),
        "by_block_split": dict(Counter(f"{r['block']}/{r['split']}" for r in rs)),
        "by_taxon": {f"{r['block']}|{r['taxon']}": dict(Counter(x["split"] for x in rs if x["taxon"] == r["taxon"] and x["block"] == r["block"])) for r in rs},
        "photographers": {b: {s: len({r["photographer_key"] for r in rs if r["block"] == b and r["split"] == s}) for s in ("dev", "sealed")}
                          for b in ("P1", "P2")},
        "licences": dict(Counter(f"{r['block']}|{r['licence']}" for r in rs)),
        "p3_sources": dict(Counter(f"{r['taxon']}|{r['source']}|{r['observation_url']}" for r in rs if r["block"] == "P3")),
    }
    return err, summary


def main():
    err, summary = validate()
    (C.HERE / "manifests" / "validation_report.json").write_text(json.dumps({"errors": err, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for e in err:
        print("  ERROR", e)
    print("PILOT VALIDATION:", "PASS" if not err else "FAIL")
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()
