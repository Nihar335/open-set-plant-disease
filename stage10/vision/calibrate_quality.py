"""Calibrate the blur and exposure thresholds of the image quality gate (M5-A2) on DEVELOPMENT images only.

  python -m stage10.vision.calibrate_quality            # writes stage10/vision/quality_calibration.json (refuses to overwrite)

Data: a deterministic sample of Stage 1-9 known-class VALIDATION images (data/splits/val.csv, split == val). The final-test
manifest is never read except by the data guard; held-out unknown images are not used.
Degradations (synthetic): Gaussian blur (sigma 2 and 4), darkening (x0.15), over-exposure (x3 + 120, clipped).
Rule (pre-declared): keep at least 99% of the original validation images, i.e.
  blur threshold = 1st percentile of original sharpness; brightness range = [0.5th, 99.5th] percentiles of originals,
and report how many degraded copies each threshold catches. Limitation: PlantVillage images are laboratory photos on plain
backgrounds; thresholds are conservative (only extreme blur / exposure is rejected) and must be re-checked on field photos.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage10.guards import assert_not_final_test
from stage10.vision.quality import measures

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "quality_calibration.json"
N = 400
SALT = "agrovaidya-m5-a2-quality-v1"


def sample() -> list[str]:
    with open(ROOT / "data/splits/val.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val" and r["role"] == "known" and r["excluded"] == "False"]
    rows.sort(key=lambda r: hashlib.sha256(f"{SALT}|{r['image_path']}".encode()).hexdigest())
    return [r["image_path"] for r in rows[:N]]


def degrade(im, kind):
    from PIL import ImageEnhance, ImageFilter
    if kind == "blur2":
        return im.filter(ImageFilter.GaussianBlur(2))
    if kind == "blur4":
        return im.filter(ImageFilter.GaussianBlur(4))
    if kind == "dark":
        return ImageEnhance.Brightness(im).enhance(0.15)
    a = np.clip(np.asarray(im, dtype=np.float32) * 3 + 120, 0, 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(a)


def main():
    from PIL import Image
    if OUT.exists():
        raise SystemExit("calibration exists; refusing to overwrite")
    paths = sample()
    res = {k: [] for k in ("orig", "blur2", "blur4", "dark", "bright")}
    for p in paths:
        assert_not_final_test(ROOT / p)
        im = Image.open(ROOT / p).convert("RGB")
        res["orig"].append(measures(im))
        for k in ("blur2", "blur4", "dark", "bright"):
            res[k].append(measures(degrade(im, k)))
    o = np.array(res["orig"])
    blur_t = float(np.percentile(o[:, 0], 1))
    lo, hi = float(np.percentile(o[:, 1], 0.5)), float(np.percentile(o[:, 1], 99.5))
    caught = {k: {"blurry": float(np.mean(np.array(v)[:, 0] < blur_t)),
                  "too_dark": float(np.mean(np.array(v)[:, 1] < lo)), "too_bright": float(np.mean(np.array(v)[:, 1] > hi))}
              for k, v in res.items()}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": "Stage 1-9 known-class validation images",
           "n_images": len(paths), "sample_salt": SALT, "sample_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
           "rule": "keep >= 99% of originals: blur = P1 of sharpness; brightness = [P0.5, P99.5]",
           "blur_laplacian_var_min": round(blur_t, 1), "brightness_range": [round(lo, 1), round(hi, 1)],
           "fraction_flagged": caught,
           "limitation": "Laboratory photos on plain backgrounds; conservative thresholds; re-check on licence-verified field photos."}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
