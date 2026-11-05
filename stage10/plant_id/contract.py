"""Validator for the plant-identification result contract (M4-11).

Structural rules: stage10/schemas/plant_identification.schema.json (JSON Schema 2020-12).
Semantic rules that JSON Schema cannot express, checked here:
  S1 candidate ranks are 1..n in order and scores are non-increasing;
  S2 candidate scientific names are unique;
  S3 status 'candidates' => the rank-1 candidate is above the threshold, and every above_threshold flag agrees with the
     operating-point threshold;
  S4 'validated' is true only if the identification registry records the rank-1 taxon as evaluated and met (M4-07);
  S5 no message or limitation text shows a percentage, or describes an unfamiliar plant as newly discovered;
  S6 an external service is never used in this milestone (no external identification API; owner decision).
"""
from __future__ import annotations

import re

from stage10.schemas.validate import structural_errors

_FORBIDDEN = re.compile(r"(\d\s*%|percent|new species|new disease|newly discovered|discovered a)", re.I)


def _texts(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k != "message_key":
                yield v
            else:
                yield from _texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _texts(v)


def contract_errors(d: dict, registry: dict | None = None) -> list[str]:
    err = structural_errors(d, "plant_identification")
    if err:
        return err
    c = d["candidates"]
    if [x["rank"] for x in c] != list(range(1, len(c) + 1)):
        err.append("S1: candidate ranks must be 1..n in order")
    if any(c[i]["score"] < c[i + 1]["score"] for i in range(len(c) - 1)):
        err.append("S1: candidate scores must be non-increasing")
    names = [x["scientific_name"] for x in c]
    if len(names) != len(set(names)):
        err.append("S2: duplicate candidate taxon")
    tau = d["operating_point"]["threshold"]
    if d["status"] == "candidates":
        if not c[0]["above_threshold"]:
            err.append("S3: status 'candidates' but the rank-1 candidate is not above the threshold")
        if tau is not None and any(x["above_threshold"] != (x["score"] >= tau) for x in c):
            err.append("S3: above_threshold disagrees with the operating-point threshold")
    if d["evaluated_scope"]["validated"]:
        entry = (registry or {}).get(c[0]["scientific_name"] if c else "", {})
        if entry.get("identifiable_status") != "met" or entry.get("evaluation_id") != d["evaluated_scope"]["evaluation_id"]:
            err.append("S4: 'validated' claimed without a matching registry entry with status 'met'")
    for t in _texts({k: v for k, v in d.items() if k not in ("model", "vocabulary", "operating_point")}):
        if _FORBIDDEN.search(t):
            err.append(f"S5: forbidden wording (percentage or 'new species/disease'): {t[:60]!r}")
    if d["model"]["external_service"] is not None or d["privacy"]["image_left_device"]:
        err.append("S6: external identification services are not permitted (image must not leave the device)")
    return err
