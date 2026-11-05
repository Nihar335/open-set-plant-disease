"""Builds the Stage 11 image index (train / val / test) from the frozen Stage 2 manifest.

* PlantVillage (all 38 classes): split by `split_group_id` (same-leaf groups from Stage 2), so re-shots of one
  leaf never cross splits. hash(group) % 10: 0 -> test, 1 -> val, otherwise train.
* PlantDoc (field photos): official test folder -> test; official train folder -> 10% val (by path hash), rest train.
  Images whose md5 appears under more than one class are dropped (known PlantDoc label conflicts).
  PlantDoc images are resized once to max side 384 into stage11_local/plantdoc_384 to keep loading fast.

Run:  python -P stage11/data_index.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage11.kb import load_kb  # noqa: E402

OUT = ROOT / "stage11" / "results" / "image_index.csv"
PD_CACHE = ROOT / "stage11_local" / "plantdoc_384"


def _bucket(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10


def build() -> pd.DataFrame:
    kb = load_kb()
    cls_index = {d["model_class"]: i for i, d in enumerate(kb.model_classes)}
    pd_index = {name: i for i, d in enumerate(kb.model_classes) for name in d.get("plantdoc", [])}

    m = pd.read_csv(ROOT / "data/splits/manifest_all.csv", low_memory=False)
    rows = []

    pv = m[(m.dataset == "plantvillage") & (m.excluded.astype(str) != "True")]
    for r in pv.itertuples():
        b = _bucket(r.split_group_id if isinstance(r.split_group_id, str) else r.leaf_group_id)
        split = "test" if b == 0 else "val" if b == 1 else "train"
        rows.append((r.image_path, "plantvillage", cls_index[r.class_name], split))

    pdoc = m[m.dataset == "plantdoc"].copy()
    conflict = pdoc.groupby("md5").class_name.nunique()
    pdoc = pdoc[~pdoc.md5.isin(conflict[conflict > 1].index)].drop_duplicates("md5")
    PD_CACHE.mkdir(parents=True, exist_ok=True)
    for r in pdoc.itertuples():
        if r.class_name not in pd_index:
            continue
        src = ROOT / r.image_path
        dst = PD_CACHE / (hashlib.md5(r.image_path.encode()).hexdigest() + ".jpg")
        if not dst.exists():
            try:
                im = Image.open(src).convert("RGB")
            except Exception as e:  # unreadable file: skip, record nothing
                print("skip", src, e)
                continue
            im.thumbnail((384, 384))
            im.save(dst, quality=92)
        if r.source_split == "test":
            split = "test"
        else:
            split = "val" if _bucket(r.image_path) == 0 else "train"
        rows.append((str(dst.relative_to(ROOT)).replace("\\", "/"), "plantdoc", pd_index[r.class_name], split))

    # Extra apple field datasets (stage11/import_extra.py: PlantVillage copies removed, near-duplicate-grouped splits).
    extra_csv = ROOT / "stage11" / "results" / "extra_index.csv"
    if extra_csv.exists():
        key_index = {d["key"]: i for i, d in enumerate(kb.model_classes)}
        for r in pd.read_csv(extra_csv).itertuples():
            rows.append((r.path, r.source, key_index[r.class_key], r.split))

    df = pd.DataFrame(rows, columns=["path", "source", "label", "split"])
    df["class_key"] = [kb.model_classes[i]["key"] for i in df.label]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(df.groupby(["source", "split"]).size().unstack())
    print("classes:", df.label.nunique(), "->", OUT)
