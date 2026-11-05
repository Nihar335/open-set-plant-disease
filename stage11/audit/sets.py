"""Evaluation sets for the Stage 11 multi-crop audit (read-only use of existing images; nothing is copied or trained on).

Each set is a DataFrame with columns path, source, label (-1 = no model class), crop (true plant, '' if not a plant),
split ('calib' = used to fit the decision rule, 'eval' = scored once).

* lab / field / apple_field : stage11/results/image_index.csv (val -> calib, test -> eval)
* dtd_texture   : DTD images that Stages 1-9 EXCLUDED (never used there); md5(path) halves -> calib / eval
* unsupported_plant : Stage 10 D2 *dev* photos of 40 Indian trees / garden plants (iNaturalist, CC BY-NC, internal
                  evaluation only). Citrus is dropped because citrus is a model crop. md5(path) halves -> calib / eval.
                  The sealed D2 test set is never read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "stage11" / "results" / "image_index.csv"
SOURCE_SET = {"plantvillage": "lab", "plantdoc": "field", "appleleaf9": "apple_field", "kashmir_apple": "apple_field"}


def _half(p: str) -> str:
    return "calib" if int(hashlib.md5(p.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def known_sets(kb, index=INDEX) -> pd.DataFrame:
    idx = pd.read_csv(index)
    idx = idx[idx.split.isin(["val", "test"])].copy()
    crops = [d["crop"] for d in kb.model_classes]
    return pd.DataFrame({"path": idx.path, "source": idx.source.map(SOURCE_SET), "label": idx.label,
                         "crop": [crops[i] for i in idx.label],
                         "split": idx.split.map({"val": "calib", "test": "eval"})}).reset_index(drop=True)


def ood_sets() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/splits/manifest_all.csv", low_memory=False, usecols=["image_path", "dataset", "split"])
    dtd = m[(m.dataset == "dtd") & (m.split == "excluded")].image_path.tolist()
    d2 = pd.read_csv(ROOT / "stage10/plant_id/d2/manifests/d2_dev_manifest.csv")
    d2 = d2[~d2.taxon.str.startswith("Citrus")]
    rows = [(p, "dtd_texture", -1, "", _half(p)) for p in dtd]
    rows += [(p, "unsupported_plant", -1, t, _half(p)) for p, t in zip(d2.stored_path, d2.taxon)]
    return pd.DataFrame(rows, columns=["path", "source", "label", "crop", "split"])
