"""Source adapters for the pilot: GBIF (iNaturalist research grade, blocks P1/P2) and Mendeley Data (block P3).

Network access is injected (`http_get(url, params) -> requests.Response-like`) so the selection logic is testable offline.
Only metadata endpoints and the image files themselves are requested; no identification service is ever called.
"""
from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field

from stage10.plant_id.pilot import common as C

GBIF_API = "https://api.gbif.org/v1"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets"


@dataclass
class Candidate:
    block: str
    taxon: str
    source: str
    record_id: str
    image_url: str
    photographer_key: str | None
    photographer_display: str | None
    licence: str
    licence_uri: str | None
    attribution: str
    observation_url: str | None = None
    observed_on: str | None = None
    label_quality: str = ""
    india_evidence: str = ""
    expected_sha256: str | None = None          # Mendeley publishes per-file SHA-256
    folder: str | None = None
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- GBIF / iNaturalist
def gbif_candidates(block: str, taxon: str, key: int, http_get, exclusions: list) -> list[Candidate]:
    g = C.config()["gbif"]
    out, offset = [], 0
    while offset < g["max_records_scanned_per_taxon"]:
        params = [("taxonKey", key), ("country", g["country"]), ("mediaType", "StillImage"), ("datasetKey", g["dataset_key"]),
                  ("eventDate", g["event_date"]), ("limit", 100), ("offset", offset)]
        params += [("license", lic) for lic in g["occurrence_licences"]]
        data = http_get(f"{GBIF_API}/occurrence/search", params).json()
        for rec in data.get("results", []):
            c = gbif_record_to_candidate(block, taxon, key, rec, exclusions)
            if c:
                out.append(c)
        offset += 100
        if data.get("endOfRecords", True):
            break
    out.sort(key=lambda c: int(c.record_id))
    return out


def gbif_record_to_candidate(block, taxon, key, rec, exclusions) -> Candidate | None:
    g = C.config()["gbif"]
    rid = str(rec.get("gbifID") or rec.get("key"))

    def drop(reason):
        exclusions.append({"block": block, "taxon": taxon, "source": "gbif_inaturalist", "record_id": rid, "reason": reason})
        return None
    if key not in {rec.get("speciesKey"), rec.get("acceptedTaxonKey"), rec.get("taxonKey")}:
        return drop("taxon_mismatch")
    if rec.get("countryCode") != g["country"]:
        return drop("not_in_country")
    media = [m for m in rec.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
    if not media:
        return drop("no_still_image")
    m = media[0]                                                   # one image per observation
    if not C.licence_ok(m.get("license"), g["media_licence_uris"]):
        return drop(f"media_licence_not_allowed:{m.get('license')}")
    url = m["identifier"]
    host = url.split("/")[2] if "://" in url else ""
    if host not in g["allowed_image_hosts"]:
        return drop(f"image_host_not_allowed:{host}")
    url = url.replace("/original.", f"/{g['image_size_variant']}.")
    who = m.get("creator") or m.get("rightsHolder") or rec.get("recordedBy")
    pk = C.normalise_photographer(who)
    if not pk:
        return drop("photographer_unknown")
    lic = "CC0 1.0" if "zero" in m["license"] else "CC BY 4.0"
    obs = rec.get("references") or rec.get("occurrenceID")
    attribution = (f"{m.get('rightsHolder') or who}, iNaturalist observation {obs}, {lic}; via GBIF occurrence {rid}. "
                   f"Changes: iNaturalist '{g['image_size_variant']}' derivative, downscaled and re-encoded.")
    return Candidate(block, taxon, "gbif_inaturalist_research_grade", rid, url, pk, who, lic, m.get("license"), attribution,
                     observation_url=obs, observed_on=(rec.get("eventDate") or "")[:10],
                     label_quality="community consensus (iNaturalist research grade); may include AI-suggested identifications",
                     india_evidence=f"GBIF countryCode={rec.get('countryCode')}; stateProvince={rec.get('stateProvince')}",
                     extra={"gbif_issues": ",".join(rec.get("issues", [])), "identified_by": rec.get("identifiedBy")})


# ----------------------------------------------------------------------------------------------- Mendeley Data
def mendeley_attribution(rec: dict, item: str) -> str:
    authors = ", ".join(f"{c.get('last_name', '').strip()}, {c.get('first_name', '').strip()}" for c in rec.get("contributors", []))
    year = (rec.get("publish_date") or "")[:4]
    doi = (rec.get("doi") or {}).get("id")
    return (f"{authors} ({year}). {rec.get('name', '').strip()}. Mendeley Data, V{rec.get('version')}. doi:{doi}. "
            f"Licence CC BY 4.0. Item: {item}. Changes: downscaled and re-encoded.")


def mendeley_candidates(taxon: str, spec: dict, rec: dict, folders: dict, exclusions: list) -> list[Candidate]:
    """Individually listed files (wheat, maize, okra). folders: folder_id -> folder name."""
    cfg = C.config()
    lic = (rec.get("data_licence") or {}).get("short_name")
    out = []
    for f in rec.get("files", []):
        name, fid = f["filename"], f["id"]
        folder = folders.get(f.get("folder_id"), "")

        def drop(reason):
            exclusions.append({"block": "P3", "taxon": taxon, "source": f"mendeley:{spec['mendeley_id']}", "record_id": fid,
                               "reason": reason, "item": f"{folder}/{name}"})
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in spec.get("include_extensions", ["jpg", "jpeg", "png"]):
            drop(f"not_an_image_file:{ext}")
            continue
        if spec.get("include_folders") and folder not in spec["include_folders"]:
            drop(f"folder_not_included:{folder}")
            continue
        pat = C.excluded_by_name(f"{folder}/{name}", cfg["exclude_name_patterns"])
        if pat:
            drop(f"name_suggests_derived_copy:{pat}")
            continue
        cd = f.get("content_details") or {}
        out.append(Candidate("P3", taxon, "mendeley_data", fid, cd.get("download_url"), f"dataset:{spec['mendeley_id']}",
                             "; ".join(f"{c.get('first_name', '')} {c.get('last_name', '')}" for c in rec.get("contributors", [])),
                             lic, "http://creativecommons.org/licenses/by/4.0/", mendeley_attribution(rec, f"{folder}/{name}"),
                             observation_url=f"https://doi.org/{(rec.get('doi') or {}).get('id')}",
                             label_quality="author-assigned crop label (dataset collected from the named crop's fields)",
                             expected_sha256=cd.get("sha256_hash"), folder=folder))
    return stratified_order(out)


def archive_candidates(taxon: str, spec: dict, rec: dict, archive_bytes: bytes, exclusions: list) -> list[tuple[Candidate, bytes]]:
    cfg = C.config()
    out = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            folder = name.rsplit("/", 1)[0] if "/" in name else ""

            def drop(reason):
                exclusions.append({"block": "P3", "taxon": taxon, "source": f"mendeley:{spec['mendeley_id']}", "record_id": name,
                                   "reason": reason, "item": name})
            if name.rsplit(".", 1)[-1].lower() not in ("jpg", "jpeg", "png"):
                drop("not_an_image_file")
                continue
            pat = C.excluded_by_name(name, cfg["exclude_name_patterns"])
            if pat:
                drop(f"name_suggests_derived_copy:{pat}")
                continue
            c = Candidate("P3", taxon, "mendeley_data", f"{spec['archive']}::{name}", f"zip:{spec['archive']}::{name}",
                          f"dataset:{spec['mendeley_id']}",
                          "; ".join(f"{x.get('first_name', '')} {x.get('last_name', '')}" for x in rec.get("contributors", [])),
                          (rec.get("data_licence") or {}).get("short_name"), "http://creativecommons.org/licenses/by/4.0/",
                          mendeley_attribution(rec, name), observation_url=f"https://doi.org/{(rec.get('doi') or {}).get('id')}",
                          label_quality="author-assigned crop label (dataset collected from the named crop's fields)", folder=folder)
            out.append((c, z.read(info)))
    order = {id(c): i for i, c in enumerate(stratified_order([c for c, _ in out]))}
    return sorted(out, key=lambda cb: order[id(cb[0])])


def stratified_order(cands: list[Candidate]) -> list[Candidate]:
    """Deterministic round-robin across folders/classes; within a folder ordered by a hash of the record id (no cherry-picking)."""
    by = {}
    for c in cands:
        by.setdefault(c.folder or "", []).append(c)
    for k in by:
        by[k].sort(key=lambda c: C.sha256(f"agrovaidya-m4-pilot-v1|{c.record_id}".encode()))
    out, i = [], 0
    keys = sorted(by)
    while any(by[k] for k in keys):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
    return out


class DownloadFailed(RuntimeError):
    pass


def polite_get(session, url, params=None, sleep=1.0, timeout=(20, 90), log=None, max_attempts=4):
    """GET with back-off. Every failed attempt is logged; 429/503 honour Retry-After (never shortened).
    403/404/410 are returned at once (not retried). Raises DownloadFailed after max_attempts."""
    import time as _t
    for attempt in range(1, max_attempts + 1):
        t0 = _t.monotonic()
        try:
            r = session.get(url, params=params, timeout=timeout)
            el = round(_t.monotonic() - t0, 2)
            if r.status_code == 200:
                if attempt > 1 and log is not None:
                    log({"url": url, "attempt": attempt, "outcome": "ok_after_retry", "seconds": el})
                time.sleep(sleep)
                return r
            if log is not None:
                log({"url": url, "attempt": attempt, "outcome": f"http_{r.status_code}", "seconds": el})
            if r.status_code in (403, 404, 410):
                return r
            wait = 5 * attempt
            if r.status_code in (429, 503):
                try:
                    wait = max(wait, int(r.headers.get("Retry-After", "0")))
                except ValueError:
                    pass
        except Exception as e:  # timeouts, connection errors
            if log is not None:
                log({"url": url, "attempt": attempt, "outcome": f"error_{type(e).__name__}", "seconds": round(_t.monotonic() - t0, 2)})
            wait = 5 * attempt
        time.sleep(wait)
    raise DownloadFailed(f"GET failed after {max_attempts} attempts: {url}")
