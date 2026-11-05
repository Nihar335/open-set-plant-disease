"""Proposed D2/C1 acquisition manifest, built from the cached GBIF METADATA only (no image is requested or opened).

  python -m stage10.plant_id.d2.plan_manifest      # -> stage10/plant_id/d2/proposed/{d2_proposed_manifest.csv, plan_summary.json}

Selection is deterministic and pre-registered:
  1. eligibility per record (taxon key, India, still image, approved host, known photographer, record + media licence in the
     draft allow-list, not a pilot image);
  2. photographer identity = iNaturalist login (GBIF field http://unknown.org/nick, present on every record), merged into one
     identity cluster with every other login that shares a normalised display name (conservative: name-alike accounts are
     never split apart). Split per cluster: clusters containing a pilot photographer (by pilot name or pilot record) ->
     development only; others by sha256(salt|cluster) < dev_fraction;
  3. per split, round-robin across taxa (config order), each taxon taking its next record in sha256(salt|record_id) order,
     subject to <= 2 per photographer per taxon and <= 10 per photographer per split; first `target` picks are PRIMARY,
     the next reserve_fraction x target are RESERVES (ranked), used only to replace primaries removed by image-level checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from stage10.plant_id.pilot import common as PC
from stage10.plant_id.d2 import survey_availability as S

HERE = Path(__file__).resolve().parent
CFG = HERE / "d2_config.yaml"
OUT = HERE / "proposed"
FIELDS = ["plan_id", "study", "group", "role", "taxon", "gbif_taxon_key", "split", "status", "rank", "gbif_id", "observation_url",
          "event_date", "media_created", "state_province", "photographer_key", "inat_login", "photographer_display",
          "media_licence", "media_licence_uri",
          "record_licence_uri", "nc_restricted", "media_identifier", "attribution"]


def config() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def licence_name(uri: str) -> str:
    u = (uri or "").lower()
    return "CC0 1.0" if "zero/1.0" in u else "CC BY-NC 4.0" if "by-nc/4.0" in u else "CC BY 4.0" if "/by/4.0" in u else "other"


def eligible(rec: dict, key: int, cfg: dict) -> tuple[dict | None, str | None]:
    """Mirror of the pilot rule (pilot.sources.gbif_record_to_candidate) with the draft licence allow-list."""
    src, lic = cfg["source"], cfg["licences"]
    if key not in {rec.get("speciesKey"), rec.get("acceptedTaxonKey"), rec.get("taxonKey")}:
        return None, "taxon_mismatch"
    if rec.get("countryCode") != src["country"]:
        return None, "not_in_country"
    if not any(m in (rec.get("license") or "").lower() for m in lic["allowed_record_licence_markers"]):
        return None, "record_licence_not_allowed"
    media = [m for m in rec.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
    if not media:
        return None, "no_still_image"
    m = media[0]
    if not PC.licence_ok(m.get("license"), lic["allowed_media_licence_uris"]):
        return None, "media_licence_not_allowed"
    url = m["identifier"]
    if (url.split("/")[2] if "://" in url else "") not in src["allowed_image_hosts"]:
        return None, "image_host_not_allowed"
    who = m.get("creator") or m.get("rightsHolder") or rec.get("recordedBy")
    nick = rec.get("http://unknown.org/nick")
    if not PC.normalise_photographer(who) or not nick:
        return None, "photographer_unknown"
    created = (m.get("created") or "")[:10]
    if not created or created < src["event_date"].split(",")[0]:
        return None, "photo_created_before_cutoff"          # photo older than the cut-off, whatever the observation date
    rid = str(rec.get("gbifID") or rec.get("key"))
    obs = rec.get("references") or rec.get("occurrenceID")
    ln = licence_name(m["license"])
    attribution = (f"{m.get('rightsHolder') or who}, iNaturalist observation {obs}, {ln} ({m['license']}); via GBIF occurrence {rid}. "
                   f"Changes: iNaturalist '{src['image_size_variant']}' derivative, downscaled and re-encoded.")
    return {"gbif_id": rid, "observation_url": obs, "event_date": (rec.get("eventDate") or "")[:10],
            "media_created": created, "state_province": rec.get("stateProvince") or "", "inat_login": nick,
            "photographer_key": None, "photographer_display": who,
            "media_licence": ln, "media_licence_uri": m["license"], "record_licence_uri": rec.get("license") or "",
            "nc_restricted": ln == "CC BY-NC 4.0",
            "media_identifier": url.replace("/original.", f"/{src['image_size_variant']}."), "attribution": attribution}, None


def identity_clusters(records: list[dict]) -> dict[str, str]:
    """login -> cluster id; logins sharing a normalised display name are merged (union-find)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_name = {}
    for rec in records:
        nick = rec.get("http://unknown.org/nick")
        m = (rec.get("media") or [{}])[0]
        name = PC.normalise_photographer(m.get("creator") or m.get("rightsHolder") or rec.get("recordedBy"))
        if not nick:
            continue
        find(nick)
        if name:
            if name in by_name:
                parent[find(nick)] = find(by_name[name])
            else:
                by_name[name] = nick
    return {n: "inat:" + min(k for k in parent if find(k) == find(n)) for n in parent}


def _tokens(name: str) -> frozenset:
    import re
    return frozenset(t for t in re.split(r"[^a-z]+", PC.normalise_photographer(name) or "") if len(t) >= 3)


def possible_same_person(pools: dict) -> set[str]:
    """Test clusters whose display-name tokens are a subset/superset of a development cluster's AND who share a state:
    possibly one person with two accounts. Conservatively held back from test (listed for manual review)."""
    info = {}
    for split in ("dev", "test"):
        for cands in pools[split].values():
            for c in cands:
                d = info.setdefault((split, c["photographer_key"]), {"tokens": _tokens(c["photographer_display"]), "states": set()})
                d["states"].add(c["state_province"])
    dev = [(k, v) for (sp, k), v in info.items() if sp == "dev" and v["tokens"]]
    out = set()
    for (sp, k), v in info.items():
        if sp != "test" or not v["tokens"]:
            continue
        for _, dv in dev:
            if (v["tokens"] <= dv["tokens"] or dv["tokens"] <= v["tokens"]) and v["states"] & dv["states"]:
                out.add(k)
                break
    return out


def _h(salt: str, s: str) -> str:
    return hashlib.sha256(f"{salt}|{s}".encode()).hexdigest()


def select(pools: dict, order: list, targets: dict, reserve_fraction: float, cap_taxon: int, cap_split: int, salt: str) -> dict:
    """pools[taxon] -> candidates (one split). Round-robin; returns taxon -> [(cand, status, rank)]."""
    queues = {t: sorted(pools.get(t, []), key=lambda c: _h(salt, c["gbif_id"])) for t in order}
    per_split, per_taxon = Counter(), defaultdict(Counter)
    out = {t: [] for t in order}
    limit = {t: targets[t] + int(round(targets[t] * reserve_fraction)) for t in order}
    progress = True
    while progress:
        progress = False
        for t in order:
            if len(out[t]) >= limit[t]:
                continue
            q = queues[t]
            while q:
                c = q.pop(0)
                pk = c["photographer_key"]
                if per_taxon[t][pk] >= cap_taxon or per_split[pk] >= cap_split:
                    continue
                per_taxon[t][pk] += 1
                per_split[pk] += 1
                n = len(out[t])
                out[t].append((c, "primary" if n < targets[t] else "reserve", n + 1))
                progress = True
                break
    return out


def build():
    cfg, scfg = config(), S.config()
    ids, phot = S.pilot_exclusions()
    plan = S.taxa_plan(scfg)
    c1 = set(scfg["c1_species"])
    pools = {"dev": defaultdict(list), "test": defaultdict(list)}
    meta, reasons = {}, defaultdict(Counter)
    cache = {name: json.loads((S.CACHE / f"occ_{name.replace(' ', '_')}.json").read_text(encoding="utf-8")) for name, _, _ in plan}
    allrecs = [r for d in cache.values() for p in d["pages"] for r in p["results"]]
    cluster = identity_clusters(allrecs)
    pilot_clusters = set()
    for rec in allrecs:
        m = (rec.get("media") or [{}])[0]
        name = PC.normalise_photographer(m.get("creator") or m.get("rightsHolder") or rec.get("recordedBy"))
        if rec.get("http://unknown.org/nick") and (name in phot or str(rec.get("gbifID")) in ids):
            pilot_clusters.add(cluster[rec["http://unknown.org/nick"]])
    for name, role, group in plan:
        d = cache[name]
        study = "C1" if name in c1 else "D2"
        g = "C1" if name in c1 else group
        meta[name] = {"role": role, "group": g, "study": study, "key": d["key"], "gbif_total": d["pages"][0]["count"]}
        for rec in (r for p in d["pages"] for r in p["results"]):
            c, why = eligible(rec, d["key"], cfg)
            if c is None:
                reasons[name][why] += 1
                continue
            if c["gbif_id"] in ids:
                reasons[name]["pilot_image"] += 1
                continue
            c["photographer_key"] = cluster[c["inat_login"]]
            pilot_ph = c["photographer_key"] in pilot_clusters or PC.normalise_photographer(c["photographer_display"]) in phot
            split = "dev" if pilot_ph else ("dev" if PC.split_for_photographer(c["photographer_key"], cfg["split"]["salt"],
                                                                                   cfg["split"]["dev_fraction"]) == "dev" else "test")
            c["pilot_photographer"] = pilot_ph
            pools[split][name].append(c)
    held_back = possible_same_person(pools)
    for n in list(pools["test"]):
        keep = [c for c in pools["test"][n] if c["photographer_key"] not in held_back]
        reasons[n]["possible_same_person_as_dev_photographer"] += len(pools["test"][n]) - len(keep)
        pools["test"][n] = keep
    order = [n for n, _, _ in plan]
    rows = []
    for split in ("dev", "test"):
        tg = {n: cfg["targets"]["c1" if n in c1 else meta[n]["role"]][split] for n in order}
        sel = select(pools[split], order, tg, cfg["reserve_fraction"], cfg["caps"]["per_photographer_per_taxon"],
                     cfg["caps"]["per_photographer_per_split"], cfg["split"]["salt"])
        for n in order:
            for c, status, rank in sel[n]:
                m = meta[n]
                rows.append({"plan_id": f"{m['study']}-{split}-{c['gbif_id']}", "study": m["study"], "group": m["group"],
                             "role": m["role"], "taxon": n, "gbif_taxon_key": m["key"], "split": split, "status": status,
                             "rank": rank, **{k: c[k] for k in FIELDS if k in c}})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "d2_proposed_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = {"nature": "PROPOSED acquisition manifest from cached GBIF metadata; no image requested, opened or stored",
               "config_sha256": hashlib.sha256(CFG.read_bytes()).hexdigest(),
               "survey_cache_sha256sums": (S.CACHE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(),
               "identity_clusters": {"logins": len(cluster), "clusters": len(set(cluster.values())),
                                     "pilot_clusters": len(pilot_clusters)},
               "held_back_from_test_possible_same_person": len(held_back),
               "eligible_pool": {n: {s: len(pools[s][n]) for s in ("dev", "test")} for n in order},
               "excluded_by_reason": {n: dict(r) for n, r in reasons.items()}, "taxa": meta}
    (OUT / "plan_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows ({sum(r['status'] == 'primary' for r in rows)} primary)")


if __name__ == "__main__":
    build()
