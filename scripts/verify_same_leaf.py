"""Level-4b duplicate audit: geometric verification of "same physical leaf" candidates.

Embedding cosine cannot separate "another photo of the same leaf" from "a similar leaf on the
same grey board" in PlantVillage (Stage-2 finding). A re-photographed leaf, however, shows the
*same* vein/lesion pattern, so local features match under one geometric transform.

For each PlantVillage candidate pair from scripts/find_near_duplicates.py:
  SIFT keypoints restricted to the leaf (HSV-saturation mask, so the shared photo board
  cannot produce matches) -> Lowe ratio test -> RANSAC homography -> inlier count.
  The second image is also tried mirrored; the larger inlier count is kept.

Calibration. Stage-2 audit showed the leaf map itself sometimes gives one physical leaf two
neighbouring ids (e.g. Septoria leaf 87 vs 88, codes 2619/2622), so "different leaf id" is NOT a
safe negative. Negatives are therefore pairs that cannot be the same leaf:
  * high-cosine candidate pairs from DIFFERENT classes, and
  * same-class pairs whose leaf-map numbers differ by >= `negative_min_leaf_gap` (both mapped).
Positives (same leaf id) are used only to report recall: many are different views of a leaf
(other side / pose) that no geometric matcher should be expected to align.
The threshold is the smallest inlier count whose negative false-positive rate is <= the
configured maximum. Pairs at/above it are written as strong same-leaf evidence and are kept
in one split by prepare_dataset.py. Nothing is deleted.

Outputs: data/processed/same_leaf_pairs.csv, results/metrics/same_leaf_verification.json,
         results/figures/dataset/dup_audit_sift_calibration.png

Usage:  python scripts/verify_same_leaf.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config, resolve  # noqa: E402


def leaf_features(path: str, sat_min: int):
    img = cv2.imread(str(ROOT / path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= sat_min).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # fill lesion/dark holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))          # drop the leaf/board boundary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1500)
    return gray, mask, sift


def inliers(a: str, b: str, sat_min: int, ratio: float, ransac_px: float) -> int:
    ga, ma, sift = leaf_features(a, sat_min)
    gb, mb, _ = leaf_features(b, sat_min)
    ka, da = sift.detectAndCompute(ga, ma)
    if da is None or len(ka) < 8:
        return 0
    best = 0
    for flip in (False, True):
        g, m = (cv2.flip(gb, 1), cv2.flip(mb, 1)) if flip else (gb, mb)
        kb, db = sift.detectAndCompute(g, m)
        if db is None or len(kb) < 8:
            continue
        good = [p[0] for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
                if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        if len(good) < 8:
            continue
        src = np.float32([ka[x.queryIdx].pt for x in good])
        dst = np.float32([kb[x.trainIdx].pt for x in good])
        _, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
        if inl is not None:
            best = max(best, int(inl.sum()))
    return best


def _job(args):
    try:
        return inliers(*args)
    except cv2.error:
        return -1


def main() -> None:
    cfg = load_config()
    sv = cfg["same_leaf_verification"]
    proc, metrics = resolve(cfg["paths"]["processed"]), resolve(cfg["paths"]["metrics"])
    rng = np.random.default_rng(cfg["seed"])
    inv = pd.read_csv(proc / "inventory.csv", low_memory=False, keep_default_na=False)
    mapped = ~inv["leaf_match"].str.startswith("fallback")
    e = pd.read_csv(proc / "embedding_pairs.csv")
    e = e[(e["dataset_i"] == "plantvillage") & (e["dataset_j"] == "plantvillage")].copy()
    e["both_mapped"] = mapped.iloc[e["i"]].to_numpy() & mapped.iloc[e["j"]].to_numpy()

    pos = e[(e["category"] == "PV: same class, same leaf group") & e["both_mapped"]]
    num = inv["maintainer_leaf_id"].str.extract(r":::(\d+(?:\.\d+)?)$")[0].astype(float)
    gap = (num.iloc[e["i"]].to_numpy() - num.iloc[e["j"]].to_numpy())
    far_same_class = (e["category"] == "PV: same class, different leaf group") & e["both_mapped"] &         (np.abs(gap) >= sv["negative_min_leaf_gap"])
    neg = e[(e["category"] == "PV: different class") | far_same_class]
    pos = pos.iloc[rng.permutation(len(pos))[: sv["calibration_pairs"]]]
    neg = neg.iloc[rng.permutation(len(neg))[: sv["calibration_pairs"]]]
    targets = e[e["category"] != "PV: same class, same leaf group"]       # pairs that could cross groups
    print(f"  calibration: {len(pos)} positives, {len(neg)} negatives "
          f"({int((neg['category'] == 'PV: different class').sum())} cross-class); targets: {len(targets)}")

    todo = pd.concat([pos.assign(set="calib_pos"), neg.assign(set="calib_neg"), targets.assign(set="target")])
    params = (sv["saturation_min"], sv["lowe_ratio"], sv["ransac_px"])
    with Pool(cfg["inspection"]["num_workers"]) as pool:
        todo["inliers"] = list(tqdm(pool.imap(_job, [(a, b, *params) for a, b in zip(todo["path_i"], todo["path_j"])],
                                              chunksize=16), total=len(todo), desc="  SIFT+RANSAC"))

    cp, cn = todo.loc[todo["set"] == "calib_pos", "inliers"], todo.loc[todo["set"] == "calib_neg", "inliers"]
    grid = list(range(5, 201))
    fpr = {t: float((cn >= t).mean()) for t in grid}
    tpr = {t: float((cp >= t).mean()) for t in grid}
    thr = next(t for t in grid if fpr[t] <= sv["max_false_positive_rate"])
    t = todo[todo["set"] == "target"]
    verified = t[t["inliers"] >= thr]
    out = verified.assign(method=f"L4b_SIFT_RANSAC_inliers>={thr}", strength="strong", score=verified["inliers"])
    out.drop(columns=["set", "both_mapped", "inliers"]).to_csv(proc / "same_leaf_pairs.csv", index=False)

    stats = {
        "calibration": {"positives": int(len(cp)), "negatives": int(len(cn)),
                        "threshold_inliers": thr, "false_positive_rate_at_threshold": fpr[thr],
                        "recall_on_known_same_leaf_pairs": tpr[thr],
                        "pos_inliers_quantiles": cp.quantile([0.1, 0.25, 0.5, 0.75]).to_dict(),
                        "neg_inliers_quantiles": cn.quantile([0.5, 0.9, 0.99, 1.0]).to_dict()},
        "targets_checked": int(len(t)),
        "verified_same_leaf_pairs": int(len(verified)),
        "verified_by_category": verified["category"].value_counts().to_dict(),
        "verified_by_class": verified["class_i"].value_counts().to_dict(),
        "verified_where_one_image_lacks_leaf_map_id": int((~verified["both_mapped"]).sum()),
        "verified_where_both_have_leaf_map_ids": int(verified["both_mapped"].sum()),
        "errors": int((todo["inliers"] < 0).sum()),
    }
    (metrics / "same_leaf_verification.json").write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps(stats, indent=2, default=float))

    from src.visualization.style import SERIES, apply_style, save
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.arange(0, 201, 5)
    ax.hist(cn.clip(upper=200), bins=bins, histtype="step", lw=2, color=SERIES[0], density=True,
            label=f"cannot be same leaf: cross-class / leaf-id gap (n={len(cn)})")
    ax.hist(cp.clip(upper=200), bins=bins, histtype="step", lw=2, color=SERIES[1], density=True,
            label=f"same leaf id in leaf map (n={len(cp)})")
    ax.axvline(thr, color="#52514e", lw=1, ls="--")
    ax.text(thr + 3, ax.get_ylim()[1] * 0.9, f"threshold = {thr}\nFPR {fpr[thr]:.3%}, recall {tpr[thr]:.1%}",
            fontsize=8, va="top")
    ax.set_xlabel("RANSAC-consistent SIFT matches on the leaf (clipped at 200)")
    ax.set_ylabel("Density")
    ax.set_title("Same-leaf verifier calibration (high-cosine candidate pairs)")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, resolve(cfg["paths"]["figures"]) / "dataset" / "dup_audit_sift_calibration.png")


if __name__ == "__main__":
    main()
