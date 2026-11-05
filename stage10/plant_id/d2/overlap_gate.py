"""Overlap gate for D2/C1 acquisition: decides, from downloaded bytes and BEFORE anything is stored, whether an image may enter.

Not used yet (no acquisition is approved). It exists so the plan's overlap rules are demonstrably enforceable, and is tested
against real Stage 1-9 and pilot files (opened read-only).

Rejection reasons (all recorded; nothing silently dropped):
  stage1_9_exact_md5          exact MD5 of the original bytes matches any of the 62,523 Stage 1-9 images
  stage1_9_near_duplicate     perceptual match to the read-only Stage 1-9 index (dHash <= td or pHash <= tp, 8 orientations)
  pilot_near_duplicate        dHash (<= 4 bits, 8 orientations) match to any of the 157 retired pilot images
  d2_near_duplicate           dHash match to an image already admitted to D2/C1
  unreadable / too_small      decode failure or short side < 224 px
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache

import numpy as np

from stage10.plant_id import neardup as N
from stage10.plant_id.pilot import common as PC

PILOT_MANIFESTS = ["p1p2_dev_manifest.csv", "p1p2_sealed_manifest.csv", "p1p2_review_excluded_manifest.csv",
                   "p1p2_quarantined_manifest.csv"]
WITHIN_MAX_BITS = 4          # same rule as the pilot (pilot_config.yaml near_duplicate_dhash_max_distance)
MIN_SIDE = 224


@lru_cache(maxsize=1)
def stage1_9_index():
    paths, D, P, meta = N.load_index()
    cal = json.loads((N.REPORTS / "calibration.json").read_text(encoding="utf-8"))
    if cal["index_sha256"] != meta["index_sha256"]:
        raise RuntimeError("calibration does not belong to this index")
    return paths, D, P, cal["thresholds"]["dhash_max_distance"], cal["thresholds"]["phash_max_distance"]


@lru_cache(maxsize=1)
def pilot_dhashes() -> list[tuple[int, str]]:
    out = []
    for name in PILOT_MANIFESTS:
        with open(PC.HERE / "manifests" / name, newline="", encoding="utf-8") as f:
            out += [(int(r["dhash"], 16), r["record_id"]) for r in csv.DictReader(f)]
    return out


def check(b: bytes, admitted: list[tuple[int, str]]) -> tuple[list[str], dict]:
    """Return (rejection reasons, evidence). Empty reasons = may be admitted. `admitted` = [(dHash, plan_id)] so far."""
    reasons, ev = [], {}
    if PC.md5(b).lower() in PC.prohibited_md5():
        reasons.append("stage1_9_exact_md5")
    im, size, why = PC.load_image(b, MIN_SIDE)
    if im is None:
        return reasons + [why.split(":")[0] if why else "unreadable"], ev
    paths, D, P, td, tp = stage1_9_index()
    qd, qp = N.query_hashes(im)
    dd, pp = N.min_distances(qd, D), N.min_distances(qp, P)
    hit = np.where((dd <= td) | (pp <= tp))[0]
    if hit.size:
        reasons.append("stage1_9_near_duplicate")
        ev["stage1_9_matches"] = [{"path": paths[i], "dhash": int(dd[i]), "phash": int(pp[i])} for i in hit[:5]]
    hs = PC.dihedral_hashes(im)
    p = PC.near_duplicate(hs, pilot_dhashes(), WITHIN_MAX_BITS)
    if p:
        reasons.append("pilot_near_duplicate")
        ev["pilot_record"] = p
    d = PC.near_duplicate(hs, admitted, WITHIN_MAX_BITS)
    if d:
        reasons.append("d2_near_duplicate")
        ev["d2_plan_id"] = d
    ev["dhash"] = hs[0]
    return reasons, ev
