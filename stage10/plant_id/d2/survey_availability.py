"""Source-availability survey for the D2 look-alike study and the C1 coconut-versus-areca study (owner approval 2026-10-15).

  python -m stage10.plant_id.d2.survey_availability --fetch     # GBIF metadata only -> stage10_local/d2_survey_cache/
  python -m stage10.plant_id.d2.survey_availability --analyse   # cache -> stage10/plant_id/d2/survey/availability_summary.json

METADATA ONLY. Requests go to the GBIF species-match and occurrence-search endpoints. No image file (media URL) is ever
requested, and no identification service is used. Nothing is scored.

The per-record eligibility rules are the pilot's (`pilot.sources.gbif_record_to_candidate`: taxon, country, still image,
media licence CC0 / CC BY 4.0, image host, photographer known), plus the record licence. The projected split is a
simulation on metadata: no image is selected, stored or opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from stage10.plant_id.pilot import common as PC
from stage10.plant_id.pilot import sources as PS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = HERE / "d2_survey_config.yaml"
CACHE = ROOT / "stage10_local" / "d2_survey_cache"
OUT = HERE / "survey"
PILOT_MANIFESTS = ["p1p2_dev_manifest.csv", "p1p2_sealed_manifest.csv", "p1p2_review_excluded_manifest.csv",
                   "p1p2_quarantined_manifest.csv"]


def config() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def taxa_plan(cfg: dict) -> list[tuple[str, str, str]]:
    """[(name, role, group)] with role in {unsupported, supported_control}."""
    out = [(n, "unsupported", g) for g, names in cfg["unsupported_groups"].items() for n in names]
    out += [(n, "supported_control", "controls") for n in cfg["supported_controls"]]
    return out


def vocabulary_keys() -> dict[str, int]:
    v = json.loads((PC.HERE / "vocabulary_v1.json").read_text(encoding="utf-8"))
    return {t["scientific_name"]: t["gbif_key"] for t in v["taxa"]}


def pilot_exclusions() -> tuple[set[str], set[str]]:
    """(GBIF record ids of all 157 pilot images incl. excluded/quarantined, normalised pilot photographer keys)."""
    ids, phot = set(), set()
    for name in PILOT_MANIFESTS:
        with open(PC.HERE / "manifests" / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ids.add(str(r["record_id"]))
                phot.add(r["photographer_key"])
    return ids, phot


def record_licence_ok(rec: dict, markers: list[str]) -> bool:
    lic = (rec.get("license") or "").lower()
    return any(m in lic for m in markers)


def _order(rid: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{rid}".encode()).hexdigest()


def capped(cands: list, per_photographer: int, salt: str, limit: int | None = None) -> list:
    """Deterministic selection: hash order of record id, at most `per_photographer` per photographer, then `limit`."""
    out, seen = [], Counter()
    for c in sorted(cands, key=lambda c: _order(c["record_id"], salt)):
        if seen[c["photographer_key"]] >= per_photographer:
            continue
        seen[c["photographer_key"]] += 1
        out.append(c)
    return out if limit is None else out[:limit]


def analyse_taxon(name: str, role: str, key: int, records: list[dict], total: int, cfg: dict,
                  pilot_ids: set[str], pilot_phot: set[str]) -> dict:
    g = cfg["gbif"]
    reasons = Counter()
    media_lic = Counter()
    meta = Counter()
    states = Counter()
    eligible = []
    for rec in records:
        media = [m for m in rec.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
        if media:
            media_lic[media[0].get("license") or "none"] += 1
        meta["creator_present"] += bool(media and (media[0].get("creator") or media[0].get("rightsHolder")))
        meta["recordedBy_present"] += bool(rec.get("recordedBy"))
        meta["identifiedBy_present"] += bool(rec.get("identifiedBy"))
        meta["eventDate_present"] += bool(rec.get("eventDate"))
        meta["stateProvince_present"] += bool(rec.get("stateProvince"))
        meta["coordinates_present"] += rec.get("decimalLatitude") is not None
        meta["observation_url_present"] += bool(rec.get("references") or rec.get("occurrenceID"))
        if not record_licence_ok(rec, g["record_licence_markers"]):
            reasons["record_licence_not_allowed"] += 1
            continue
        excl = []
        c = PS.gbif_record_to_candidate("D2", name, key, rec, excl)
        if c is None:
            reasons[excl[0]["reason"].split(":")[0]] += 1
            continue
        if c.record_id in pilot_ids:
            reasons["pilot_image"] += 1
            continue
        states[rec.get("stateProvince") or "unknown"] += 1
        eligible.append({"record_id": c.record_id, "photographer_key": c.photographer_key,
                         "pilot_photographer": c.photographer_key in pilot_phot})
    cap = cfg["max_images_per_photographer_per_taxon"]
    salt, frac = cfg["salt"], cfg["dev_fraction"]
    split = defaultdict(list)
    for c in eligible:
        s = "dev" if c["pilot_photographer"] else PC.split_for_photographer(c["photographer_key"], salt, frac)
        split["dev" if s == "dev" else "test"].append(c)
    tgt = cfg["targets"]["c1" if name in cfg["c1_species"] else role]
    proj = {}
    for s in ("dev", "test"):
        avail = capped(split[s], cap, salt)
        chosen = avail[:tgt[s]]
        proj[s] = {"available_after_photographer_cap": len(avail),
                   "photographers_available": len({c["photographer_key"] for c in avail}),
                   "projected_at_target": len(chosen), "target": tgt[s],
                   "projected_photographers": len({c["photographer_key"] for c in chosen})}
    allcap = capped(eligible, cap, salt)
    return {"taxon": name, "role": role, "gbif_key": key, "gbif_total_records": total, "records_scanned": len(records),
            "scan_complete": len(records) >= total,
            "eligible_images_one_per_observation": len(eligible),
            "eligible_photographers": len({c["photographer_key"] for c in eligible}),
            "eligible_after_photographer_cap_all_splits": len(allcap),
            "eligible_from_pilot_photographers_dev_only": sum(c["pilot_photographer"] for c in eligible),
            "excluded_by_reason": dict(reasons), "first_media_licence_counts": dict(media_lic),
            "metadata_fields_present": dict(meta), "top_states": states.most_common(5), "projected_split": proj}


# --------------------------------------------------------------------------------------------------------- fetch
def fetch():
    import requests
    cfg, g = config(), config()["gbif"]
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logrows = []

    def log(d):
        logrows.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **d})
    s = requests.Session()
    s.headers["User-Agent"] = "AgroVaidya-research-survey/0.1 (metadata only)"
    vk = vocabulary_keys()
    matches = {}
    for name, role, group in taxa_plan(cfg):
        if role == "supported_control":
            matches[name] = {"usageKey": vk[name], "source": "frozen vocabulary_v1.json"}
            continue
        r = PS.polite_get(s, f"{PS.GBIF_API}/species/match", {"name": name, "kingdom": "Plantae", "strict": "true"},
                          sleep=g["sleep_seconds"], log=log)
        log({"url": r.url, "attempt": 1, "outcome": f"http_{r.status_code}", "seconds": None})
        if r.status_code != 200:
            raise SystemExit(f"species/match failed for {name}: HTTP {r.status_code}; stopping")
        matches[name] = r.json()
    (CACHE / "species_match.json").write_text(json.dumps(matches, indent=1), encoding="utf-8")
    for name, role, group in taxa_plan(cfg):
        m = matches[name]
        key = m.get("acceptedUsageKey") or m.get("usageKey")
        if role == "unsupported" and (m.get("matchType") != "EXACT" or m.get("rank") != "SPECIES" or not key):
            log({"url": f"species/match {name}", "attempt": 1, "outcome": f"no_exact_species_match:{m.get('matchType')}", "seconds": None})
            continue
        pages, offset = [], 0
        while offset < g["max_records_scanned_per_taxon"]:
            params = [("taxonKey", key), ("country", g["country"]), ("mediaType", "StillImage"), ("datasetKey", g["dataset_key"]),
                      ("eventDate", g["event_date"]), ("limit", g["page_size"]), ("offset", offset)]
            r = PS.polite_get(s, f"{PS.GBIF_API}/occurrence/search", params, sleep=g["sleep_seconds"], log=log)
            log({"url": r.url, "attempt": 1, "outcome": f"http_{r.status_code}", "seconds": None})
            if r.status_code != 200:
                raise SystemExit(f"occurrence/search failed for {name}: HTTP {r.status_code}; stopping")
            d = r.json()
            pages.append(d)
            offset += g["page_size"]
            if d.get("endOfRecords", True):
                break
        fn = CACHE / f"occ_{name.replace(' ', '_')}.json"
        fn.write_text(json.dumps({"taxon": name, "key": key, "pages": pages}), encoding="utf-8")
        print(f"{name}: {pages[0].get('count')} records, {sum(len(p['results']) for p in pages)} scanned", flush=True)
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(CACHE.glob("*.json")))
    (CACHE / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    with open(OUT / "request_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utc", "url", "attempt", "outcome", "seconds"])
        w.writeheader()
        w.writerows(logrows)


# ------------------------------------------------------------------------------------------------------- analyse
def analyse():
    cfg = config()
    matches = json.loads((CACHE / "species_match.json").read_text(encoding="utf-8"))
    pilot_ids, pilot_phot = pilot_exclusions()
    vk = vocabulary_keys()
    taxa, unresolved = [], []
    for name, role, group in taxa_plan(cfg):
        fn = CACHE / f"occ_{name.replace(' ', '_')}.json"
        if not fn.exists():
            unresolved.append({"taxon": name, "match": {k: matches[name].get(k) for k in ("matchType", "rank", "status")}})
            continue
        d = json.loads(fn.read_text(encoding="utf-8"))
        if role == "unsupported" and d["key"] in vk.values():
            raise SystemExit(f"{name} resolves to a vocabulary key; unsupported taxa must be outside vocabulary v1")
        recs = [r for p in d["pages"] for r in p["results"]]
        t = analyse_taxon(name, role, d["key"], recs, d["pages"][0]["count"], cfg, pilot_ids, pilot_phot)
        m = matches[name]
        t.update(group=group, match={k: m.get(k) for k in ("matchType", "status", "scientificName", "usageKey",
                                                            "acceptedUsageKey", "source") if m.get(k) is not None})
        taxa.append(t)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "nature": "METADATA-ONLY source-availability survey; no image requested, selected, stored or scored",
               "config_sha256": hashlib.sha256(CFG.read_bytes()).hexdigest(),
               "cache_sha256sums": (CACHE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(),
               "pilot_exclusions": {"pilot_images": len(pilot_ids), "pilot_photographers": len(pilot_phot)},
               "taxa": taxa, "unresolved_names": unresolved}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "availability_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(taxa)} taxa analysed, {len(unresolved)} unresolved")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    fetch() if a.fetch else analyse()


if __name__ == "__main__":
    main()
