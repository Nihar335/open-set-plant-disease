"""Calibrate the visible-sign analyser (vision/symptoms.py) on Stage 1-9 VALIDATION images only.

  python -m stage10.vision.calibrate_symptoms [--per-class 60]

Threshold for each sign = the 95th percentile of that measure on HEALTHY validation leaves (healthy leaves show the
sign at most 5% of the time by construction). The record also reports, per disease class, how often each sign was seen
on a separate sample of validation images, so the analyser's behaviour is documented, not assumed. No test or held-out
image is read (the script refuses any row whose split is not 'val').
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from stage10.vision import symptoms as SY

PCT = 95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    a = ap.parse_args()
    v = pd.read_csv("data/splits/val.csv", low_memory=False)
    v = v[(v.split == "val") & (v.role == "known") & (~v.excluded.astype(bool))]
    assert set(v.split) == {"val"}
    rng = np.random.default_rng(20261015)
    rows = []
    for cls, g in v.groupby("class_name"):
        idx = rng.permutation(len(g))[: 2 * a.per_class]
        for j, k in enumerate(idx):
            r = g.iloc[k]
            m, lf, _ = SY.measures(r.image_path)
            rows.append({"class_name": cls, "half": "cal" if j % 2 == 0 else "check", "leaf_fraction": lf, **m})
    df = pd.DataFrame(rows)
    healthy = df[df.class_name.str.contains("healthy") & (df.half == "cal")]
    th = {s: float(np.percentile(healthy[s], PCT)) for s in SY.SIGNS}
    # a sign that healthy leaves never show still needs a floor, so pixel noise is not "seen"
    floors = {"yellowing": 0.02, "brown_spots": 0.01, "dark_spots": 0.005, "yellow_halo": 0.10, "elongated_lesions": 0.005,
              "small_spots": 0.5, "white_coating": 0.01, "margin_scorch": 0.03, "large_blight": 0.25, "bronzing": 0.02}
    th = {s: round(max(th[s], floors[s]), 4) for s in SY.SIGNS}
    min_leaf = round(float(np.percentile(df.leaf_fraction, 1)) * 0.5, 3)
    chk = df[df.half == "check"]
    rates = {cls: {s: round(float((g[s] > th[s]).mean()), 3) for s in SY.SIGNS} for cls, g in chk.groupby("class_name")}
    rec = {"analyser_version": SY.ANALYSER_VERSION, "created": str(date.today()), "split_used": "val (Stage 1-9 validation) only",
           "rule": f"threshold = max(P{PCT} of the measure on healthy validation leaves, fixed noise floor)",
           "healthy_images": int(len(healthy)), "images_measured": int(len(df)), "per_class_sample": 2 * a.per_class,
           "min_leaf_fraction": min_leaf, "thresholds": th, "noise_floors": floors,
           "seen_rate_on_check_half": rates,
           "limitations": "Laboratory PlantVillage leaves on plain backgrounds, six crops only. Not validated on field photos."}
    SY.CALIBRATION.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(th, indent=1), "min_leaf", min_leaf)
    print(pd.DataFrame(rates).T.to_string())


if __name__ == "__main__":
    main()
