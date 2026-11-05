"""Import the extra apple field datasets (downloaded from Hugging Face, CC BY 4.0) into the Stage 11 index.

* AppleLeaf9 (Yang et al. 2022, Agronomy 12:2784; Project-AgML/appleleaf9_disease_classification):
  PlantVillage + ATLDSD + Plant Pathology 2020/2021 apple leaves, 9 classes.
* Kashmir indigenous apple dataset (Yatoo & Sharma 2024, Data in Brief 53:110165;
  Project-AgML/apple_leaf_disease_classification): Alternaria, mosaic, healthy.

Leakage control
* AppleLeaf9 re-uses PlantVillage images. Every imported image whose 64-bit dHash is within 6 bits of any
  PlantVillage image already in our index is dropped, so PlantVillage test photos cannot enter training.
* Near-duplicates (augmented copies, re-shots; dHash distance <= 6) are linked into groups with union-find, and
  the train / val / test split is drawn per group (hash(group) % 10: 0 test, 1 val, else train).
* Exact duplicates across classes are dropped.

Run:  python -P stage11/import_extra.py      -> stage11/results/extra_index.csv, stage11_local/extra_384/
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "stage11_local" / "downloads"
OUT_IMG = ROOT / "stage11_local" / "extra_384"
OUT_CSV = ROOT / "stage11" / "results" / "extra_index.csv"
MAX_DIST = 6

# dataset label name -> Stage 11 class key
APPLELEAF9 = {
    "Alternaria leaf spot": "apple_alternaria_leaf_spot", "Brown spot": "apple_marssonina_blotch",
    "Frogeye leaf spot": "apple_black_rot", "Grey spot": "apple_grey_spot", "Health": "apple_healthy",
    "Mosaic": "apple_mosaic", "Powdery mildew": "apple_powdery_mildew", "Rust": "apple_cedar_rust",
    "Scab": "apple_scab",
}
KASHMIR = {"Alternaria": "apple_alternaria_leaf_spot", "Apple_Mosaic": "apple_mosaic", "Healthy": "apple_healthy"}


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


def near(hashes: np.ndarray, h: int) -> np.ndarray:
    return popcount(np.bitwise_xor(hashes, np.uint64(h))) <= MAX_DIST


def label_names(path: Path) -> list:
    meta = pq.read_schema(path).metadata or {}
    import json
    info = json.loads(meta[b"huggingface"])["info"]["features"]["label"]["names"]
    return info


def read_dataset(folder: Path, mapping: dict, source: str):
    files = sorted(folder.glob("*.parquet"))
    names = label_names(files[0])
    for f in files:
        t = pq.read_table(f)
        imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
        for im, lab in zip(imgs, labels):
            yield source, mapping[names[lab]], im["bytes"]


def main():
    idx = pd.read_csv(ROOT / "stage11" / "results" / "image_index.csv")
    pv_apple = idx[(idx.source == "plantvillage") & idx.class_key.str.startswith("apple_")]
    print("hashing", len(pv_apple), "PlantVillage apple images")
    pv_hashes = np.array([dhash(Image.open(ROOT / p)) for p in pv_apple.path], dtype=np.uint64)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    rows, hashes, md5_class = [], [], {}
    dropped_pv = dropped_bad = dropped_conflict = 0
    for source, key, data in [*read_dataset(DL / "appleleaf9", APPLELEAF9, "appleleaf9"),
                              *read_dataset(DL / "kashmir_apple", KASHMIR, "kashmir_apple")]:
        md5 = hashlib.md5(data).hexdigest()
        if md5 in md5_class:
            if md5_class[md5] != key:
                dropped_conflict += 1
            continue
        md5_class[md5] = key
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            dropped_bad += 1
            continue
        h = dhash(img)
        if near(pv_hashes, h).any():
            dropped_pv += 1
            continue
        img.thumbnail((384, 384))
        rel = f"stage11_local/extra_384/{source}/{md5}.jpg"
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(ROOT / rel, quality=92)
        rows.append({"path": rel, "source": source, "class_key": key, "w": img.width, "h": img.height, "md5": md5})
        hashes.append(h)
    print(f"kept {len(rows)}; dropped PlantVillage copies {dropped_pv}, unreadable {dropped_bad}, "
          f"cross-class duplicates {dropped_conflict}")

    # union-find near-duplicate groups
    H = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(H)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(H)):
        for j in np.nonzero(near(H[i + 1:], int(H[i])))[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    df = pd.DataFrame(rows)
    df["group"] = [f"x{find(i)}" for i in range(len(df))]
    # drop groups that mix classes (ambiguous near-duplicates)
    mixed = df.groupby("group").class_key.nunique()
    df = df[~df.group.isin(mixed[mixed > 1].index)]
    b = df.group.map(lambda g: int(hashlib.md5(g.encode()).hexdigest(), 16) % 10)
    df["split"] = np.where(b == 0, "test", np.where(b == 1, "val", "train"))
    df.to_csv(OUT_CSV, index=False)
    print("mixed-class near-dup groups dropped:", int((mixed > 1).sum()))
    print(df.groupby(["class_key", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
