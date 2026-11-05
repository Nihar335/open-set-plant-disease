"""D2/C1 acquisition under frozen protocol v2 (PROTOCOL_V2.md). Internal, non-commercial research evaluation only.

  python -m stage10.plant_id.d2.acquire_d2 --reverify     # live GBIF metadata re-verification of the frozen manifest
  python -m stage10.plant_id.d2.acquire_d2 --acquire      # download / gate / store, in rank order, until targets are met

Rules enforced here (each also checked independently by validate_d2.py):
  * refuses to run unless protocol_v2_lock.json verifies and the D2 test seal does NOT yet exist;
  * only rows of the frozen manifest are eligible, in rank order per (split, taxon); reserves only replace primaries that
    failed re-verification, download, the overlap gate or the visual screen (screen_decisions.csv);
  * live re-verification: record still present with identical taxon key, licences, photo and login, else dropped;
  * downloads only the iNaturalist Open Data 'medium' derivative from the allowed host; polite rate, logged retries;
    any HTTP 403/429 challenge response stops the run (never bypassed);
  * overlap gate on the downloaded bytes before anything is stored; every rejection logged with evidence;
  * stored copy <= 1024 px in stage10_local/d2_internal_nc_eval/ (outside the frozen tree and the Stage-10 manifest).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stage10.plant_id.d2 import overlap_gate as G
from stage10.plant_id.d2 import plan_manifest as P
from stage10.plant_id.d2 import survey_availability as S
from stage10.plant_id.pilot import common as PC
from stage10.plant_id.pilot import sources as PS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCK = HERE / "protocol_v2_lock.json"
SEAL = HERE / "SEAL_D2.json"
MAN = HERE / "manifests"
STORE = ROOT / "stage10_local" / "d2_internal_nc_eval"
REVERIFY = STORE / "reverify_cache"
ACQ_FIELDS = P.FIELDS + ["stored_path", "stored_sha256", "original_sha256", "original_md5", "width", "height", "dhash",
                         "licence_verified_utc", "screen_decision", "screen_caveat"]
NOTICE = """INTERNAL, NON-COMMERCIAL RESEARCH EVALUATION ONLY (AgroVaidya Stage 10, Milestone 4, protocol v2).
Most images here are licensed CC BY-NC 4.0 by their photographers (see manifests/d2_acquired.csv for the licence,
creator, observation URL and GBIF ID of every file). Do NOT redistribute, publish, include in any public dataset, use to
train or fine-tune any model, or use in any deployed or commercial product. Keep separate from any public or commercially
intended dataset. Source: iNaturalist Open Data (via GBIF dataset doi:10.15468/ab3s5x). Changes: downscaled, re-encoded.
"""


class AcquisitionStop(RuntimeError):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_lock() -> dict:
    if not LOCK.exists():
        raise AcquisitionStop("protocol_v2_lock.json missing: protocol v2 is not frozen")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for rel, h in lock["files"].items():
        if sha(ROOT / rel) != h:
            raise AcquisitionStop(f"{rel} changed since protocol v2 was frozen")
    return lock


def frozen_rows() -> list[dict]:
    with open(P.OUT / "d2_proposed_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict], fields: list[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class Log:
    def __init__(self, path: Path):
        self.path, self.rows = path, read_csv(path)

    def __call__(self, d: dict):
        self.rows.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: d.get(k) for k in ("url", "attempt", "outcome", "seconds")}})

    def save(self):
        write_csv(self.path, self.rows, ["utc", "url", "attempt", "outcome", "seconds"])


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "AgroVaidya-research-evaluation/0.2 (internal non-commercial; metadata + open-data images)"
    return s


# ------------------------------------------------------------------------------------------------ re-verification
def reverify():
    verify_lock()
    if SEAL.exists():
        raise AcquisitionStop("test set already sealed")
    cfg, scfg = P.config(), S.config()
    s, log = _session(), Log(MAN / "acquisition_request_log.csv")
    REVERIFY.mkdir(parents=True, exist_ok=True)
    rows = frozen_rows()
    keys = {r["taxon"]: int(r["gbif_taxon_key"]) for r in rows}
    g = scfg["gbif"]
    live, unparseable = {}, []
    try:
        for taxon, key in keys.items():
            pages = []

            def fetch(offset, limit):
                params = [("taxonKey", key), ("country", g["country"]), ("mediaType", "StillImage"), ("datasetKey", g["dataset_key"]),
                          ("eventDate", g["event_date"]), ("limit", limit), ("offset", offset)]
                r = PS.polite_get(s, f"{PS.GBIF_API}/occurrence/search", params, sleep=1.0, log=log)
                log({"url": r.url, "attempt": 1, "outcome": f"http_{r.status_code}"})
                if r.status_code != 200:
                    raise AcquisitionStop(f"GBIF returned HTTP {r.status_code} ({r.headers.get('Cf-Mitigated')}); stopping, not bypassing")
                try:
                    return r.json()
                except ValueError:                       # server-side serialisation error: split the range, never guess
                    log({"url": r.url, "attempt": 1, "outcome": "malformed_json_split"})
                    if limit == 1:
                        unparseable.append({"taxon": taxon, "offset": offset})
                        return {"results": [], "endOfRecords": False, "count": None}
                    half = max(1, limit // 2)
                    a = fetch(offset, half)
                    b = fetch(offset + half, limit - half)
                    return {"results": a["results"] + b["results"], "endOfRecords": b.get("endOfRecords", False),
                            "count": a.get("count") or b.get("count")}
            offset = 0
            while True:
                d = fetch(offset, g["page_size"])
                pages.append(d)
                offset += g["page_size"]
                if d.get("endOfRecords", True) or offset >= 10000 or not d["results"] and d.get("count") is not None:
                    break
            (REVERIFY / f"occ_{taxon.replace(' ', '_')}.json").write_text(json.dumps({"taxon": taxon, "key": key, "pages": pages}), encoding="utf-8")
            for p in pages:
                for rec in p["results"]:
                    live[str(rec.get("gbifID"))] = (key, rec)
            print(f"  re-verified {taxon}", flush=True)
    finally:
        log.save()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for r in rows:
        status = "ok"
        if r["gbif_id"] not in live:
            status = "record_absent_or_no_longer_matching_query"
        else:
            key, rec = live[r["gbif_id"]]
            c, why = P.eligible(rec, key, cfg)
            if c is None:
                status = f"now_ineligible:{why}"
            elif (c["media_licence_uri"], c["record_licence_uri"], c["media_identifier"], c["inat_login"]) != \
                    (r["media_licence_uri"], r["record_licence_uri"], r["media_identifier"], r["inat_login"]):
                status = "changed_since_survey"
        out.append({"plan_id": r["plan_id"], "gbif_id": r["gbif_id"], "status": status, "verified_utc": now})
    write_csv(MAN / "reverification.csv", out, ["plan_id", "gbif_id", "status", "verified_utc"])
    (MAN / "reverification_unparseable.json").write_text(json.dumps(unparseable, indent=1), encoding="utf-8")
    sums = "".join(f"{sha(p)}  {p.name}\n" for p in sorted(REVERIFY.glob("*.json")))
    (MAN / "reverify_cache_SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    print(f"re-verification: {sum(o['status'] == 'ok' for o in out)} ok of {len(out)}")


# ---------------------------------------------------------------------------------------------------- acquisition
def acquire(max_new: int | None = None):
    verify_lock()
    if SEAL.exists():
        raise AcquisitionStop("test set already sealed: no further acquisition")
    cfg = P.config()
    rv = {r["plan_id"]: r for r in read_csv(MAN / "reverification.csv")}
    if not rv:
        raise AcquisitionStop("run --reverify first")
    rows = frozen_rows()
    acquired = {r["plan_id"]: r for r in read_csv(MAN / "d2_acquired.csv")}
    rejected = {r["plan_id"]: r for r in read_csv(MAN / "d2_rejections.csv")}
    screen = {r["plan_id"]: r for r in read_csv(MAN / "screen_decisions.csv")}
    for pid, a in acquired.items():
        d = screen.get(pid)
        a["screen_decision"] = d["decision"] if d else ""
        a["screen_caveat"] = d.get("caveat", "") if d else ""
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "NOTICE_INTERNAL_NC_EVALUATION_ONLY.txt").write_text(NOTICE, encoding="utf-8")
    admitted_hashes = [(int(a["dhash"], 16), pid) for pid, a in acquired.items()]
    s, log = _session(), Log(MAN / "acquisition_request_log.csv")
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["taxon"]), []).append(r)
    targets = {}
    for (split, taxon), rs in groups.items():
        targets[(split, taxon)] = sum(1 for r in rs if r["status"] == "primary")
    new = 0
    try:
        for (split, taxon), rs in sorted(groups.items()):
            rs.sort(key=lambda r: int(r["rank"]))

            def good(pid):
                a = acquired.get(pid)
                return a is not None and a["screen_decision"] not in ("exclude", "quarantine")
            for r in rs:
                if sum(good(x["plan_id"]) for x in rs) >= targets[(split, taxon)]:
                    break
                pid = r["plan_id"]
                if pid in acquired or pid in rejected:
                    continue
                if max_new is not None and new >= max_new:
                    return
                if rv.get(pid, {}).get("status") != "ok":
                    rejected[pid] = {"plan_id": pid, "reason": f"reverification:{rv.get(pid, {}).get('status', 'missing')}", "evidence": ""}
                    continue
                url = r["media_identifier"]
                if not url.startswith("https://inaturalist-open-data.s3.amazonaws.com/photos/") or "/medium." not in url:
                    rejected[pid] = {"plan_id": pid, "reason": "url_not_allowed", "evidence": url}
                    continue
                try:
                    resp = PS.polite_get(s, url, sleep=0.5, log=log)
                except PS.DownloadFailed as e:
                    rejected[pid] = {"plan_id": pid, "reason": "download_failed", "evidence": str(e)}
                    continue
                if resp.status_code in (403, 429) or resp.headers.get("Cf-Mitigated"):
                    raise AcquisitionStop(f"HTTP {resp.status_code} {resp.headers.get('Cf-Mitigated')} from image host; stopping, not bypassing")
                if resp.status_code != 200:
                    rejected[pid] = {"plan_id": pid, "reason": f"http_{resp.status_code}", "evidence": url}
                    continue
                b = resp.content
                reasons, ev = G.check(b, admitted_hashes)
                if reasons:
                    rejected[pid] = {"plan_id": pid, "reason": ";".join(reasons), "evidence": json.dumps(ev)[:500]}
                    continue
                im, size, _ = PC.load_image(b, G.MIN_SIDE)
                stored = PC.stored_copy(im, cfg["storage"]["stored_max_side_px"])
                rel = Path("stage10_local/d2_internal_nc_eval/images") / split / taxon.lower().replace(" ", "_") / f"{r['gbif_id']}.jpg"
                (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / rel).write_bytes(stored)
                acquired[pid] = {**r, "stored_path": rel.as_posix(), "stored_sha256": PC.sha256(stored),
                                 "original_sha256": PC.sha256(b), "original_md5": PC.md5(b), "width": size[0], "height": size[1],
                                 "dhash": f"{ev['dhash']:016x}", "licence_verified_utc": rv[pid]["verified_utc"],
                                 "screen_decision": "", "screen_caveat": ""}
                admitted_hashes.append((ev["dhash"], pid))
                new += 1
                if new % 50 == 0:
                    print(f"  {new} new images", flush=True)
                    _save(acquired, rejected)
    finally:
        _save(acquired, rejected)
        log.save()
    print(f"acquired {len(acquired)} total ({new} new); rejected {len(rejected)}")


def _save(acquired, rejected):
    order = sorted(acquired.values(), key=lambda a: (a["split"], a["taxon"], int(a["rank"])))
    write_csv(MAN / "d2_acquired.csv", order, ACQ_FIELDS)
    write_csv(MAN / "d2_rejections.csv", sorted(rejected.values(), key=lambda r: r["plan_id"]), ["plan_id", "reason", "evidence"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reverify", action="store_true")
    g.add_argument("--acquire", action="store_true")
    ap.add_argument("--max-new", type=int)
    a = ap.parse_args()
    try:
        reverify() if a.reverify else acquire(a.max_new)
    except AcquisitionStop as e:
        print("STOPPED:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
