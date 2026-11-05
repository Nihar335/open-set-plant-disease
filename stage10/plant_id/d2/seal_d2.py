"""Finalise screened D2/C1 manifests and seal the fresh test set (protocol v2 section 5).

  python -m stage10.plant_id.d2.seal_d2 --contact-sheets   # build screening sheets for unscreened images (internal only)
  python -m stage10.plant_id.d2.seal_d2 --seal             # write final manifests + SEAL_D2.json (refuses to overwrite)

Sealing refuses unless: protocol v2 lock verifies; every acquired image has a screen decision; acquisition has filled
every (split, taxon) up to its target or exhausted its reserves; all stored files match their hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stage10.plant_id.d2 import acquire_d2 as Q

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHEETS = Q.STORE / "contact_sheets"
DECISIONS = {"keep", "keep_with_caveat", "exclude", "quarantine"}
FINAL = {"dev": "d2_dev_manifest.csv", "test": "d2_test_manifest.csv"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def contact_sheets(per_sheet: int = 30, tile: int = 220):
    from PIL import Image, ImageDraw
    acq = Q.read_csv(Q.MAN / "d2_acquired.csv")
    done = {r["plan_id"] for r in Q.read_csv(Q.MAN / "screen_decisions.csv")}
    todo = [a for a in acq if a["plan_id"] not in done]
    SHEETS.mkdir(parents=True, exist_ok=True)
    start = len(list(SHEETS.glob("sheet_*.jpg")))
    index = []
    for s in range(0, len(todo), per_sheet):
        chunk = todo[s:s + per_sheet]
        cols = 6
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile, rows * (tile + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for i, a in enumerate(chunk):
            im = Image.open(ROOT / a["stored_path"]).convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (i % cols) * tile, (i // cols) * (tile + 18)
            sheet.paste(im, (x + (tile - im.width) // 2, y))
            label = f"{i:02d} {a['taxon'][:22]}"
            d.text((x + 3, y + tile + 2), label, fill="black")
            index.append({"sheet": f"sheet_{start + s // per_sheet:03d}.jpg", "cell": i, "plan_id": a["plan_id"], "taxon": a["taxon"],
                          "split": a["split"]})
        sheet.save(SHEETS / f"sheet_{start + s // per_sheet:03d}.jpg", quality=85)
    old = Q.read_csv(SHEETS / "index.csv")
    Q.write_csv(SHEETS / "index.csv", old + index, ["sheet", "cell", "plan_id", "taxon", "split"])
    print(f"{len(todo)} images on {(len(todo) + per_sheet - 1) // per_sheet} new sheets")


def final_rows():
    acq = Q.read_csv(Q.MAN / "d2_acquired.csv")
    dec = {r["plan_id"]: r for r in Q.read_csv(Q.MAN / "screen_decisions.csv")}
    missing = [a["plan_id"] for a in acq if a["plan_id"] not in dec]
    if missing:
        raise Q.AcquisitionStop(f"{len(missing)} acquired images have no screen decision")
    bad = [d for d in dec.values() if d["decision"] not in DECISIONS]
    if bad:
        raise Q.AcquisitionStop(f"invalid screen decisions: {bad[:3]}")
    for a in acq:
        a["screen_decision"] = dec[a["plan_id"]]["decision"]
        a["screen_caveat"] = dec[a["plan_id"]].get("caveat", "")
    return acq


def seal():
    Q.verify_lock()
    if Q.SEAL.exists():
        raise Q.AcquisitionStop("SEAL_D2.json exists; refusing to overwrite")
    acq = final_rows()
    plan = Q.frozen_rows()
    targets = Counter((r["split"], r["taxon"]) for r in plan if r["status"] == "primary")
    tried = {r["plan_id"] for r in Q.read_csv(Q.MAN / "d2_rejections.csv")} | {a["plan_id"] for a in acq}
    keep = [a for a in acq if a["screen_decision"] in ("keep", "keep_with_caveat")]
    have = Counter((a["split"], a["taxon"]) for a in keep)
    for key, tgt in targets.items():
        pending = [r for r in plan if (r["split"], r["taxon"]) == key and r["plan_id"] not in tried]
        if have[key] < tgt and pending:
            raise Q.AcquisitionStop(f"{key}: {have[key]}/{tgt} kept and reserves remain: run --acquire again before sealing")
    for a in keep:
        if sha(ROOT / a["stored_path"]) != a["stored_sha256"]:
            raise Q.AcquisitionStop(f"stored file changed: {a['stored_path']}")
    for split, fn in FINAL.items():
        Q.write_csv(Q.MAN / fn, [a for a in keep if a["split"] == split], Q.ACQ_FIELDS)
    Q.write_csv(Q.MAN / "d2_screen_excluded_manifest.csv", [a for a in acq if a["screen_decision"] == "exclude"], Q.ACQ_FIELDS)
    Q.write_csv(Q.MAN / "d2_quarantined_manifest.csv", [a for a in acq if a["screen_decision"] == "quarantine"], Q.ACQ_FIELDS)
    test = [a for a in keep if a["split"] == "test"]
    seal_doc = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_v2_lock_sha256": sha(Q.LOCK),
                "files": {f"stage10/plant_id/d2/manifests/{fn}": sha(Q.MAN / fn) for fn in
                          (*FINAL.values(), "d2_screen_excluded_manifest.csv", "d2_quarantined_manifest.csv", "screen_decisions.csv")},
                "test_rows": len(test), "dev_rows": len(keep) - len(test),
                "test_stored_files_sha256": hashlib.sha256("".join(sorted(a["stored_sha256"] for a in test)).encode()).hexdigest(),
                "note": "Sealed before any development scoring or threshold work. Scoring the test set requires "
                        "SEALED_SCORING_APPROVAL_D2.json from the owner (protocol v2 section 7)."}
    Q.SEAL.write_text(json.dumps(seal_doc, indent=1) + "\n", encoding="utf-8")
    print(f"sealed: {seal_doc['test_rows']} test, {seal_doc['dev_rows']} development")


def verify_seal() -> dict:
    if not Q.SEAL.exists():
        raise Q.AcquisitionStop("test set not sealed")
    s = json.loads(Q.SEAL.read_text(encoding="utf-8"))
    if s["protocol_v2_lock_sha256"] != sha(Q.LOCK):
        raise Q.AcquisitionStop("protocol lock changed since sealing")
    for rel, h in s["files"].items():
        if sha(ROOT / rel) != h:
            raise Q.AcquisitionStop(f"{rel} changed since sealing")
    return s


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--contact-sheets", action="store_true")
    g.add_argument("--seal", action="store_true")
    a = ap.parse_args()
    try:
        contact_sheets() if a.contact_sheets else seal()
    except Q.AcquisitionStop as e:
        print("STOPPED:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
