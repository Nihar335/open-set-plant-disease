"""Read-only near-duplicate check against all Stage 1-9 images (owner approval 2026-10-15).

  python -m stage10.plant_id.neardup --build-index     # hash every image in data/splits/manifest_all.csv (read-only)
  python -m stage10.plant_id.neardup --calibrate       # transformed Stage 1-9 copies vs the index -> thresholds
  python -m stage10.plant_id.neardup --check-pilot     # the 157 stored pilot images vs the index -> detected pairs

Stage 1-9 files are opened for READING only. No Stage 1-9 file, manifest or inventory is written. The index lives in
stage10_local/neardup_index/ (outside the frozen directories). Reports are written to stage10/plant_id/d2/neardup/.

Method: two 64-bit perceptual hashes per image - a difference hash (dHash, 9x8 grey gradients) and a DCT hash (pHash,
32x32 grey, low 8x8 DCT block vs its median). Index images are hashed as stored; each query image is hashed under all 8
rotations/reflections, and the minimum Hamming distance is used. A pair is FLAGGED for visual review when either distance
is at or below its calibrated threshold. A flag is a candidate, not a confirmed copy; a non-flag does not prove independence
(crops, collages, heavy edits, screenshots and visually similar but different photos are not reliably detected), and
nothing here can check BioCLIP 2's training data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "splits" / "manifest_all.csv"
INDEX_DIR = ROOT / "stage10_local" / "neardup_index"
INDEX = INDEX_DIR / "stage1_9_hashes.npz"
REPORTS = Path(__file__).resolve().parent / "d2" / "neardup"
SEED = 20261015
FALSE_FLAG_TARGET = 0.01          # threshold = largest distance whose per-query false-flag rate on impostors is <= 1%


# ----------------------------------------------------------------------------------------------------------- hashes
def _grey(im, size):
    from PIL import Image
    return np.asarray(im.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.float64)


def dhash(im) -> int:
    g = _grey(im, (9, 8))
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash(im) -> int:
    from scipy.fft import dctn
    d = dctn(_grey(im, (32, 32)), norm="ortho")[:8, :8].flatten()
    med = np.median(d[1:])
    return int("".join("1" if v > med else "0" for v in d), 2)


def dihedral(im) -> list:
    from PIL import Image
    out = []
    for base in (im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for rot in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270):
            out.append(base if rot is None else base.transpose(rot))
    return out


def query_hashes(im) -> tuple[np.ndarray, np.ndarray]:
    v = dihedral(im.convert("RGB"))
    return (np.array([dhash(x) for x in v], dtype=np.uint64), np.array([phash(x) for x in v], dtype=np.uint64))


def min_distances(q: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per index row: min over the query's 8 variants of the Hamming distance."""
    best = np.full(index.shape[0], 64, dtype=np.int64)
    for h in q:
        x = np.bitwise_xor(index, np.uint64(h))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        best = np.minimum(best, d)
    return best


def open_readonly(path: Path):
    from PIL import Image
    with open(path, "rb") as f:                       # read-only handle; nothing is ever written back
        b = f.read()
    im = Image.open(io.BytesIO(b))
    im.load()
    return im.convert("RGB")


# ------------------------------------------------------------------------------------------------------- index
def build_index():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths, dh, ph, failed = [], [], [], []
    for i, r in enumerate(rows):
        try:
            im = open_readonly(ROOT / r["image_path"])
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            failed.append({"image_path": r["image_path"], "error": type(e).__name__})
            continue
        paths.append(r["image_path"])
        dh.append(dhash(im))
        ph.append(phash(im))
        if i % 5000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX, paths=np.array(paths), dhash=np.array(dh, dtype=np.uint64), phash=np.array(ph, dtype=np.uint64))
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "manifest_rows": len(rows), "indexed": len(paths), "failed": failed,
            "index_sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest()}
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"indexed {len(paths)} of {len(rows)}; failed {len(failed)}")


def load_index():
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8"))
    if hashlib.sha256(INDEX.read_bytes()).hexdigest() != meta["index_sha256"]:
        raise SystemExit("near-duplicate index hash mismatch; rebuild")
    z = np.load(INDEX)
    return list(z["paths"]), z["dhash"], z["phash"], meta


# --------------------------------------------------------------------------------------------------- calibration
def transforms():
    from PIL import Image, ImageEnhance

    def jpeg(im, q):
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=q)
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")

    def resize(im, s):
        return im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))), Image.Resampling.BILINEAR)

    def crop(im, k):
        dx, dy = int(im.width * (1 - k) / 2), int(im.height * (1 - k) / 2)
        return im.crop((dx, dy, im.width - dx, im.height - dy))
    return {
        "jpeg_q60": lambda im: jpeg(im, 60),
        "resize_50pct_jpeg_q85": lambda im: jpeg(resize(im, 0.5), 85),
        "upscale_200pct_jpeg_q90": lambda im: jpeg(resize(im, 2.0), 90),
        "rotate90_flip": lambda im: im.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "brightness_115pct": lambda im: ImageEnhance.Brightness(im).enhance(1.15),
        "contrast_85pct_jpeg_q80": lambda im: jpeg(ImageEnhance.Contrast(im).enhance(0.85), 80),
        "centre_crop_95pct": lambda im: crop(im, 0.95),
        "centre_crop_85pct": lambda im: crop(im, 0.85),
    }


def calibrate(n: int = 300):
    paths, D, P, meta = load_index()
    rng = random.Random(SEED)
    sample = rng.sample(range(len(paths)), n)
    T = transforms()
    genuine = {k: {"d": [], "p": []} for k in T}
    impostor = {"d": [], "p": []}
    for j, i in enumerate(sample):
        im = open_readonly(ROOT / paths[i])
        for k, fn in T.items():
            qd, qp = query_hashes(fn(im))
            dd, pp = min_distances(qd, D), min_distances(qp, P)
            genuine[k]["d"].append(int(dd[i]))
            genuine[k]["p"].append(int(pp[i]))
            if k == "jpeg_q60":                       # impostor = nearest OTHER image (excluding exact-hash twins of self)
                same = (D == D[i]) & (P == P[i])
                impostor["d"].append(int(dd[~same].min()))
                impostor["p"].append(int(pp[~same].min()))
        if j % 50 == 0:
            print(f"  {j}/{n}", flush=True)
    th = {}
    for h in ("d", "p"):
        imp = np.array(impostor[h])
        ok = [t for t in range(0, 33) if (imp <= t).mean() <= FALSE_FLAG_TARGET]
        th[h] = max(ok) if ok else 0
    rec = {k: {"dhash_recall": float(np.mean(np.array(v["d"]) <= th["d"])),
               "phash_recall": float(np.mean(np.array(v["p"]) <= th["p"])),
               "either_recall": float(np.mean((np.array(v["d"]) <= th["d"]) | (np.array(v["p"]) <= th["p"]))),
               "dhash_median": float(np.median(v["d"])), "phash_median": float(np.median(v["p"]))} for k, v in genuine.items()}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED, "n_sampled_stage1_9_images": n,
           "index_sha256": meta["index_sha256"], "false_flag_target_per_query": FALSE_FLAG_TARGET,
           "thresholds": {"dhash_max_distance": th["d"], "phash_max_distance": th["p"]},
           "impostor_nearest_other_distance": {
               h: {"min": int(min(impostor[h])), "p01": float(np.percentile(impostor[h], 1)),
                   "median": float(np.median(impostor[h]))} for h in ("d", "p")},
           "recall_by_transform_at_thresholds": rec,
           "caveat": "Impostors are Stage 1-9 images against other Stage 1-9 images (mostly uniform-background leaf photos). "
                     "False-flag rates for other image types can differ; every flag is reviewed visually."}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "calibration.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["thresholds"]), json.dumps(rec, indent=0))


def check_pilot():
    paths, D, P, meta = load_index()
    cal = json.loads((REPORTS / "calibration.json").read_text(encoding="utf-8"))
    td, tp = cal["thresholds"]["dhash_max_distance"], cal["thresholds"]["phash_max_distance"]
    pilot = Path(__file__).resolve().parent / "pilot"
    rows = []
    for name in ("p1p2_dev_manifest.csv", "p1p2_sealed_manifest.csv", "p1p2_review_excluded_manifest.csv",
                 "p1p2_quarantined_manifest.csv"):
        with open(pilot / "manifests" / name, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    flagged, nearest = [], []
    for r in rows:
        im = open_readonly(ROOT / r["stored_path"])
        qd, qp = query_hashes(im)
        dd, pp = min_distances(qd, D), min_distances(qp, P)
        nearest.append((int(dd.min()), int(pp.min())))
        for i in np.where((dd <= td) | (pp <= tp))[0]:
            flagged.append({"pilot_record_id": r["record_id"], "pilot_taxon": r["taxon"], "pilot_stored_path": r["stored_path"],
                            "stage1_9_path": paths[i], "dhash_distance": int(dd[i]), "phash_distance": int(pp[i])})
    nd = np.array(nearest)
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "index_sha256": meta["index_sha256"],
           "thresholds": cal["thresholds"], "pilot_images_checked": len(rows), "stage1_9_images_indexed": meta["indexed"],
           "flagged_pairs": flagged,
           "nearest_distance_summary": {"dhash_min": int(nd[:, 0].min()), "dhash_median": float(np.median(nd[:, 0])),
                                        "phash_min": int(nd[:, 1].min()), "phash_median": float(np.median(nd[:, 1]))},
           "note": "Retired pilot images are checked only to exercise the method; this does not re-score or re-open the sealed evaluation."}
    (REPORTS / "pilot_vs_stage1_9.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} pilot images checked; {len(flagged)} flagged pairs; nearest {out['nearest_distance_summary']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-index", action="store_true")
    g.add_argument("--calibrate", action="store_true")
    g.add_argument("--check-pilot", action="store_true")
    a = ap.parse_args()
    if a.build_index:
        build_index()
    elif a.calibrate:
        calibrate()
    else:
        check_pilot()


if __name__ == "__main__":
    main()
