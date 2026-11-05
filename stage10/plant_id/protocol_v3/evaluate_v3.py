"""Protocol v3 (PROTOCOL_V3_PROPOSAL.md). Development data only.

  python -m stage10.plant_id.protocol_v3.evaluate_v3 --lock          # pre-registration lock (before any v3 scoring)
  <isolated venv python> -m stage10.plant_id.protocol_v3.evaluate_v3 --score-dev   # full softmax, development images only
  python -m stage10.plant_id.protocol_v3.evaluate_v3 --select         # candidates A/B/C + adoption rule -> operating point

Sealed D2-test / C1-test images are never read by this module. Sealed scoring under v3 is not implemented and not authorised.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D2 = HERE.parent / "d2"
RESULTS = HERE / "results"
LOCK = HERE / "protocol_v3_prereg_lock.json"
OP = HERE / "protocol_v3_operating_point.json"
FULL = RESULTS / "d2_dev_full_softmax.csv"
LOCKED = ["stage10/plant_id/protocol_v3/PROTOCOL_V3_PROPOSAL.md", "stage10/plant_id/protocol_v3/methods.py",
          "stage10/plant_id/protocol_v3/evaluate_v3.py", "stage10/plant_id/d2/thresholds.py",
          "stage10/plant_id/d2/protocol_v2_lock.json", "stage10/plant_id/d2/SEAL_D2.json",
          "stage10/plant_id/d2/manifests/d2_dev_manifest.csv", "stage10/plant_id/pilot/vocabulary_v1.json"]
TEST_DIR = "stage10_local/d2_internal_nc_eval/images/test"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_lock():
    if LOCK.exists():
        raise SystemExit("pre-registration lock exists; refusing to overwrite")
    if FULL.exists():
        raise SystemExit("v3 development scores already exist: the lock must precede scoring")
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "files": {f: sha(ROOT / f) for f in LOCKED}}, indent=1) + "\n", encoding="utf-8")
    print("protocol v3 pre-registration locked")


def verify_lock() -> dict:
    if not LOCK.exists():
        raise SystemExit("protocol v3 is not pre-registered (no lock)")
    lk = json.loads(LOCK.read_text(encoding="utf-8"))
    for f, h in lk["files"].items():
        if sha(ROOT / f) != h:
            raise SystemExit(f"pre-registered file changed: {f}")
    return lk


def dev_rows() -> list[dict]:
    """The development manifest only, after the D2 seal verifies. Any test path is refused."""
    from stage10.plant_id.d2 import seal_d2
    seal_d2.verify_seal()
    with open(D2 / "manifests" / seal_d2.FINAL["dev"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["split"] != "dev" or r["stored_path"].replace("\\", "/").startswith(TEST_DIR):
            raise SystemExit(f"non-development row refused: {r['plan_id']}")
    return rows


def vocab_names() -> list[str]:
    v = json.loads((HERE.parent / "pilot" / "vocabulary_v1.json").read_text(encoding="utf-8"))
    return [t["scientific_name"] for t in v["taxa"]]


def score_dev(device: str):
    verify_lock()
    if FULL.exists():
        raise SystemExit("v3 development scores exist; refusing to overwrite")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from PIL import Image
    from stage10.plant_id.pilot import evaluate_pilot as E
    sc = E.Scorer("T1", device)
    torch = sc.torch
    names = vocab_names()
    RESULTS.mkdir(exist_ok=True)
    out = []
    for r in dev_rows():
        im = Image.open(ROOT / r["stored_path"]).convert("RGB")
        x = sc.pre(im).unsqueeze(0).to(device)
        with torch.no_grad():
            f = sc.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (sc.scale * f @ sc.tf.T).softmax(dim=-1)[0].float().cpu().tolist()
        out.append({"plan_id": r["plan_id"], "taxon": r["taxon"], "group": r["group"], "role": r["role"], "study": r["study"],
                    "photographer": r["photographer_key"], **{f"p{i:02d}": f"{v:.8g}" for i, v in enumerate(p)}})
    with open(FULL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"scored {len(out)} development images; columns p00..p39 follow vocabulary_v1 order")


def load_full() -> list[dict]:
    with open(FULL, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{**r, "probs": [float(r[f"p{i:02d}"]) for i in range(40)]} for r in rows]


def select():
    from stage10.plant_id.protocol_v3 import methods as M
    lk = verify_lock()
    if OP.exists():
        raise SystemExit("v3 operating point exists; refusing to overwrite")
    recs = load_full()
    ids = {r["plan_id"] for r in dev_rows()}
    if {r["plan_id"] for r in recs} != ids:
        raise SystemExit("v3 scores do not match the development manifest")
    res = M.evaluate_all(recs, vocab_names())
    op = {"protocol": "v3", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": "development only",
          "sealed_test_read": False, "prereg_lock_sha256": sha(LOCK), "prereg_locked_utc": lk["locked_utc"],
          "dev_full_softmax_sha256": sha(FULL), "n_records": len(recs), **res}
    if res["adopted"]:
        a = res["candidates"][res["adopted"]]
        op.update({k: a[k] for k in ("tau", "tau_g", "tau_c") if k in a})
    OP.write_text(json.dumps(op, indent=1, default=str) + "\n", encoding="utf-8")
    for c, r in res["candidates"].items():
        print(c, r["status"], r.get("reason"))
    print("adopted:", res["adopted"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lock", action="store_true")
    g.add_argument("--score-dev", action="store_true")
    g.add_argument("--select", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    write_lock() if a.lock else score_dev(a.device) if a.score_dev else select()


if __name__ == "__main__":
    main()
