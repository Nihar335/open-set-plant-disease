"""Milestone 4 exploratory pilot: acquisition (owner approval 2026-10-14). Run from the repository root:

  python -P -m stage10.plant_id.pilot.acquire      (use `python -m` WITHOUT -P if the repository root is not importable)

Downloads ONLY the approved blocks (P1, P2 via GBIF/iNaturalist; P3 via Mendeley Data), validates every image, records full
provenance, and writes manifests. It never calls an identification API and never scores an image.
Stops (raises) on any publisher-checksum mismatch or licence anomaly.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from stage10.plant_id.pilot import common as C
from stage10.plant_id.pilot import sources as S

MANIFESTS = C.HERE / "manifests"
SOURCE_RECORDS = C.HERE / "source_records"
FIELDS = ["block", "split", "taxon", "source", "record_id", "observation_url", "observed_on", "photographer_key", "photographer_display",
          "licence", "licence_uri", "attribution", "label_quality", "india_evidence", "folder", "image_url", "original_bytes",
          "original_sha256", "original_md5", "publisher_sha256_verified", "width", "height", "stored_path", "stored_sha256", "dhash"]


class IntegrityStop(RuntimeError):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class Run:
    def __init__(self):
        self.cfg = C.config()
        self.store = C.ROOT / self.cfg["storage_root"]
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "AgroVaidya-research-pilot/0.1 (licence-filtered research pilot)"
        self.bad_md5 = C.prohibited_md5()
        self.accepted: list[dict] = []
        self.excluded: list[dict] = []
        self.seen: list[tuple[int, str]] = []
        self.request_log: list[dict] = []

    def get(self, url, params=None, sleep=1.0):
        return S.polite_get(self.http, url, params, sleep=sleep, log=self.request_log.append)

    def fetch_image(self, c: S.Candidate, sleep: float):
        """Image download; failures become recorded exclusions, never silent and never fatal."""
        try:
            r = self.get(c.image_url, sleep=sleep)
        except S.DownloadFailed:
            self.exclude(c, "download_failed_after_retries")
            return None
        if r.status_code != 200:
            self.exclude(c, f"download_http_{r.status_code}")
            return None
        return r.content

    def exclude(self, c: S.Candidate, reason: str, **kw):
        self.excluded.append({"block": c.block, "taxon": c.taxon, "source": c.source, "record_id": c.record_id, "reason": reason,
                              "item": c.folder or "", **kw})

    def admit(self, c: S.Candidate, data: bytes, split: str) -> bool:
        osha = C.sha256(data)
        if c.expected_sha256:
            if osha != c.expected_sha256:
                raise IntegrityStop(f"publisher checksum mismatch for {c.record_id}: {osha} != {c.expected_sha256}")
        if C.md5(data) in self.bad_md5:
            self.exclude(c, "exact_md5_match_with_stage1_9_data")
            return False
        im, size, why = C.load_image(data, self.cfg["min_side_px"])
        if why:
            self.exclude(c, why)
            return False
        hashes = C.dihedral_hashes(im)
        dup = C.near_duplicate(hashes, self.seen, self.cfg["near_duplicate_dhash_max_distance"])
        if dup:
            self.exclude(c, f"near_duplicate_of:{dup}")
            return False
        stored = C.stored_copy(im, self.cfg["stored_max_side_px"])
        if C.md5(stored) in self.bad_md5:
            self.exclude(c, "exact_md5_match_with_stage1_9_data(stored_copy)")
            return False
        ssha = C.sha256(stored)
        rel = Path(self.cfg["storage_root"]) / "images" / c.block / split / slug(c.taxon) / f"{ssha[:20]}.jpg"
        (C.ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (C.ROOT / rel).write_bytes(stored)
        self.seen.append((hashes[0], c.record_id))
        self.accepted.append({"block": c.block, "split": split, "taxon": c.taxon, "source": c.source, "record_id": c.record_id,
                              "observation_url": c.observation_url, "observed_on": c.observed_on, "photographer_key": c.photographer_key,
                              "photographer_display": c.photographer_display, "licence": c.licence, "licence_uri": c.licence_uri,
                              "attribution": c.attribution, "label_quality": c.label_quality, "india_evidence": c.india_evidence,
                              "folder": c.folder or "", "image_url": c.image_url, "original_bytes": len(data), "original_sha256": osha,
                              "original_md5": C.md5(data), "publisher_sha256_verified": bool(c.expected_sha256),
                              "width": size[0], "height": size[1], "stored_path": rel.as_posix(), "stored_sha256": ssha,
                              "dhash": f"{hashes[0]:016x}"})
        return True

    # ------------------------------------------------------------------ P1 / P2
    def gbif_block(self, block: str):
        g = self.cfg["gbif"]
        spec = self.cfg["blocks"][block]
        for taxon, key in spec["taxa"].items():
            cands = S.gbif_candidates(block, taxon, key, lambda u, p: self.get(u, p, sleep=0.5), self.excluded)
            per_photographer: dict[str, int] = {}
            n = 0
            for c in cands:
                if n >= spec["max_images_per_taxon"]:
                    break
                if per_photographer.get(c.photographer_key, 0) >= g["max_images_per_photographer_per_taxon"]:
                    self.exclude(c, "photographer_cap_reached")
                    continue
                data = self.fetch_image(c, sleep=0.3)
                if data is None:
                    continue
                split = C.split_for_photographer(c.photographer_key, g["split"]["salt"], g["split"]["dev_fraction"])
                if self.admit(c, data, split):
                    per_photographer[c.photographer_key] = per_photographer.get(c.photographer_key, 0) + 1
                    n += 1
            print(f"{block} {taxon}: {n} accepted from {len(cands)} licence-eligible records; "
                  f"failed request attempts so far {len(run_failures(self))}", flush=True)

    # ------------------------------------------------------------------ P3
    def mendeley_record(self, did: str) -> dict:
        r = self.get(f"{S.MENDELEY_API}/{did}", sleep=3.0)
        if r.status_code != 200:
            raise IntegrityStop(f"Mendeley record {did} unavailable: HTTP {r.status_code}")
        SOURCE_RECORDS.mkdir(parents=True, exist_ok=True)
        (SOURCE_RECORDS / f"mendeley_{did}.json").write_bytes(r.content)
        rec = r.json()
        lic = (rec.get("data_licence") or {})
        if lic.get("short_name") != "CC BY 4.0" or "creativecommons.org/licenses/by/4.0" not in (lic.get("url") or ""):
            raise IntegrityStop(f"Mendeley {did}: licence is not unambiguously CC BY 4.0 ({lic.get('short_name')}) - ask the owner")
        return rec

    def p3(self):
        spec_all = self.cfg["blocks"]["P3"]
        for taxon, spec in spec_all["datasets"].items():
            rec = self.mendeley_record(spec["mendeley_id"])
            n = 0
            if spec.get("archive"):
                f = next(x for x in rec["files"] if x["filename"] == spec["archive"])
                cd = f["content_details"]
                r = self.get(cd["download_url"], sleep=2.0)
                if r.status_code != 200 or C.sha256(r.content) != cd["sha256_hash"]:
                    raise IntegrityStop(f"archive {spec['archive']} download/checksum failed")
                pairs = S.archive_candidates(taxon, spec, rec, r.content, self.excluded)
                for c, data in pairs:
                    if n >= spec_all["max_images_per_taxon"]:
                        break
                    n += self.admit(c, data, "sealed")
                total = len(pairs)
            else:
                fr = self.get(f"{S.MENDELEY_API}/{spec['mendeley_id']}/folders/{rec['version']}", sleep=3.0)
                folders = {x["id"]: x["name"] for x in fr.json()} if fr.status_code == 200 else {}
                if spec.get("include_folders") and not folders:
                    raise IntegrityStop(f"cannot resolve folder names for {spec['mendeley_id']}")
                cands = S.mendeley_candidates(taxon, spec, rec, folders, self.excluded)
                for c in cands:
                    if n >= spec_all["max_images_per_taxon"]:
                        break
                    data = self.fetch_image(c, sleep=2.0)
                    if data is None:
                        continue
                    n += self.admit(c, data, "sealed")
                total = len(cands)
            print(f"P3 {taxon}: {n} accepted from {total} eligible items", flush=True)


def run_failures(run: Run) -> list:
    return [x for x in run.request_log if not x["outcome"].startswith("ok")]


def write_outputs(run: Run, started: str, blocks: list[str], status: str = "complete", stop_reason: str | None = None):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    groups = {"p1p2_dev_manifest.csv": lambda r: r["block"] in ("P1", "P2") and r["split"] == "dev",
              "p1p2_sealed_manifest.csv": lambda r: r["block"] in ("P1", "P2") and r["split"] == "sealed",
              "p3_sealed_manifest.csv": lambda r: r["block"] == "P3"}
    seal = {"created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "rule":
            "Sealed manifests may not be scored, viewed for tuning or used for any model choice until the owner's approval is recorded "
            "in SEALED_SCORING_APPROVAL.json; P3 is test-only.", "files": {}}
    for name, pred in groups.items():
        rows = sorted((r for r in run.accepted if pred(r)), key=lambda r: (r["block"], r["taxon"], r["record_id"]))
        with open(MANIFESTS / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        if "sealed" in name:
            seal["files"][name] = {"rows": len(rows), "sha256": C.sha256((MANIFESTS / name).read_bytes())}
    with open(MANIFESTS / "exclusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["block", "taxon", "source", "record_id", "reason", "item"])
        w.writeheader()
        w.writerows(sorted(run.excluded, key=lambda r: (r["block"], r["taxon"], str(r["record_id"]))))
    (C.HERE / "SEAL.json").write_text(json.dumps(seal, indent=1), encoding="utf-8")
    log = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
           "accepted": len(run.accepted), "excluded": len(run.excluded), "config_sha256": C.sha256(C.CONFIG.read_bytes()),
           "prohibited_md5_count": len(run.bad_md5), "failed_request_attempts": len(run_failures(run)),
           "requests_recovered_after_retry": sum(1 for x in run.request_log if x["outcome"] == "ok_after_retry"), "python": sys.version.split()[0], "requests": requests.__version__,
           "blocks_requested": blocks, "status": status, "stop_reason": stop_reason,
           "note": "Exploratory pilot acquisition. No image was scored. No identification API was called."}
    (MANIFESTS / "acquisition_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    write_request_log(run)


def write_request_log(run: Run):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS / "request_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "attempt", "outcome", "seconds"])
        w.writeheader()
        w.writerows(run.request_log)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="P1,P2,P3", help="comma-separated subset of the approved blocks")
    a = ap.parse_args(argv)
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    if not set(blocks) <= {"P1", "P2", "P3"}:
        raise SystemExit("only the approved blocks P1, P2, P3 may be acquired")
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run = Run()
    if (C.ROOT / run.cfg["storage_root"] / "images").exists():
        raise SystemExit("pilot images already exist; refusing to re-acquire over them")
    try:
        for b in ("P1", "P2"):
            if b in blocks:
                run.gbif_block(b)
        if "P3" in blocks:
            run.p3()
    except (IntegrityStop, S.DownloadFailed) as e:
        # safe stop: everything accepted so far is written WITH its provenance, and the stop is recorded for review
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        (MANIFESTS / "STOPPED.json").write_text(json.dumps({"stopped_utc": datetime.now(timezone.utc).isoformat(), "reason": str(e),
                                                            "accepted_before_stop": len(run.accepted),
                                                            "excluded_before_stop": len(run.excluded)}, indent=1), encoding="utf-8")
        write_outputs(run, started, blocks, status="stopped", stop_reason=str(e))
        print(f"STOPPED SAFELY: {e}", flush=True)
        raise SystemExit(2)
    write_outputs(run, started, blocks)
    print(f"accepted {len(run.accepted)}, excluded {len(run.excluded)}")


if __name__ == "__main__":
    main()
