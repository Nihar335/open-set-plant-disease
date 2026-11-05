"""Build a file-level inventory of every downloaded image (no pixel reads here)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.plantvillage import leaf_group_key, load_leaf_map, maintainer_leaf_id, parse_class
from src.utils.config import PROJECT_ROOT, resolve

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def _files(d: Path) -> tuple[list[Path], list[Path]]:
    """(image files, non-image files) under d, excluding our own marker files."""
    imgs, other = [], []
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.name.startswith(".download_complete"):
            continue
        (imgs if p.suffix.lower() in IMAGE_EXT else other).append(p)
    return imgs, other


def plantvillage_inventory(cfg: dict) -> tuple[pd.DataFrame, dict]:
    c = cfg["datasets"]["plantvillage"]
    base = resolve(c["dir"])
    color = base / c["variant"]
    leaf_map = load_leaf_map(base / "meta" / "leaf-map.json")

    official = {}
    for split in ("train", "test"):
        for line in (base / "meta" / f"{c['variant']}_{split}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.strip().split("/")
                official[f"{parts[-2]}/{parts[-1]}"] = split

    imgs, other = _files(color)
    rows = []
    for p in imgs:
        cls = p.parent.name
        crop, disease = parse_class(cls)
        leaf_id, how = maintainer_leaf_id(cls, p.name, leaf_map)
        rows.append({"dataset": "plantvillage", "source_split": official.get(f"{cls}/{p.name}", "unlisted"),
                     "class_name": cls, "crop": crop, "disease": disease,
                     "is_healthy": disease.lower() == "healthy", "rel_path": _rel(p), "file_name": p.name,
                     "maintainer_leaf_id": leaf_id, "leaf_group": leaf_group_key(cls, leaf_id), "leaf_match": how,
                     "leaf_map_other_class": how == "map_unique" and leaf_id.split(":::")[0] != cls,
                     "name_has_copy": "copy" in p.name.lower()})
    df = pd.DataFrame(rows)
    on_disk = set(df["class_name"] + "/" + df["file_name"])
    extra = {"non_image_files": [_rel(p) for p in other],
             "listed_but_missing": sorted(set(official) - on_disk),
             "unlisted_on_disk": int((df["source_split"] == "unlisted").sum())}
    return df, extra


def plantdoc_inventory(cfg: dict) -> tuple[pd.DataFrame, dict]:
    base = resolve(cfg["datasets"]["plantdoc"]["dir"])
    cmap = cfg["plantdoc_class_map"]
    imgs, other = [], []
    for split in ("train", "test"):
        i, o = _files(base / split)
        imgs += i
        other += o
    folders = sorted({p.parent.name for p in imgs})
    unmapped = [f for f in folders if f not in cmap]
    if unmapped:
        raise KeyError(f"PlantDoc folders missing from config.plantdoc_class_map: {unmapped}")
    rows = []
    for p in imgs:
        m = cmap[p.parent.name]
        rows.append({"dataset": "plantdoc", "source_split": p.parent.parent.name,
                     "class_name": p.parent.name, "crop": m["crop"], "disease": m["disease"],
                     "is_healthy": m["disease"] == "healthy", "pv_equivalent": m["pv"],
                     "rel_path": _rel(p), "file_name": p.name, "maintainer_leaf_id": "",
                     "leaf_group": "", "leaf_match": "n/a", "leaf_map_other_class": False,
                     "name_has_copy": "copy" in p.name.lower()})
    return pd.DataFrame(rows), {"non_image_files": [_rel(p) for p in other],
                                "config_classes_absent_on_disk": sorted(set(cmap) - set(folders))}


def dtd_inventory(cfg: dict) -> tuple[pd.DataFrame, dict]:
    base = resolve(cfg["datasets"]["dtd"]["dir"]) / "images"
    imgs, other = _files(base)
    rows = [{"dataset": "dtd", "source_split": "all", "class_name": f"dtd/{p.parent.name}",
             "crop": "none (texture)", "disease": p.parent.name, "is_healthy": False,
             "rel_path": _rel(p), "file_name": p.name, "maintainer_leaf_id": "",
             "leaf_group": "", "leaf_match": "n/a", "leaf_map_other_class": False,
             "name_has_copy": "copy" in p.name.lower()} for p in imgs]
    return pd.DataFrame(rows), {"non_image_files": [_rel(p) for p in other]}
