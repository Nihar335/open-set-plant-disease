"""Stage-6 embeddings: dimension, finiteness, determinism, alignment with locked manifests, head consistency."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.utils.config import PROJECT_ROOT

EMB = PROJECT_ROOT / "results" / "embeddings"
have = pytest.mark.skipif(not (EMB / "test_embeddings.npz").exists(), reason="run scripts/feature_osr.py extract first")


@pytest.mark.parametrize("split", ["train", "val", "test"])
@have
def test_dimension_finite_and_alignment(split):
    z = np.load(EMB / f"{split}_embeddings.npz", allow_pickle=False)
    E, L, paths = z["embeddings"], z["logits"], list(z["image_path"])
    assert E.ndim == 2 and E.shape[1] == 2048 and L.shape == (len(E), 21)
    assert np.isfinite(E).all() and np.isfinite(L).all()
    meta = pd.read_csv(EMB / f"{split}_meta.csv", low_memory=False, keep_default_na=False)
    man = pd.read_csv(PROJECT_ROOT / "data" / "splits" / f"{split}.csv", low_memory=False, keep_default_na=False)
    man = man[man["split"] == split]
    assert paths == meta["image_path"].tolist() == man["image_path"].tolist()          # same rows, same order
    assert (meta["known_unknown"].tolist() == man["known_unknown"].tolist())
    assert (meta["class_id"].astype(int).tolist() == man["class_id"].astype(int).tolist())
    assert (meta.loc[meta["known_unknown"] == "unknown", "class_id"].astype(int) == -1).all()
    if split == "train":
        assert set(meta["role"]) == {"known"}                                          # stats can only use known train


@have
def test_extraction_checks_recorded():
    c = json.loads((EMB / "extraction_checks.json").read_text())
    assert c["embedding_dim"] == 2048
    assert c["determinism_rerun_max_abs_diff"] <= 1e-5
    for split, s in c["splits"].items():
        assert s["finite"] and s["head_consistency_max_abs"] < 1e-3
        if "vs_stage4_logits_max_abs" in s:
            assert s["vs_stage4_argmax_mismatches"] == 0


def test_live_determinism_and_head_consistency():
    """Same images twice through the frozen model -> identical embeddings; head(embedding) == forward logits."""
    from scripts.osr_msp import load_frozen, manifest
    from src.data.dataset import ManifestDataset
    from src.osr.embeddings import extract
    from src.utils.config import load_config
    cfg = load_config()
    model, classes, tf, device, s3, ck = load_frozen(cfg)
    df = manifest("data/splits/val.csv", "val").head(6)
    dl = DataLoader(ManifestDataset(df, tf), batch_size=6, shuffle=False, num_workers=0)
    E1, L1, _ = extract(model, dl, device)
    E2, L2, _ = extract(model, dl, device)
    assert E1.shape == (6, 2048) and np.abs(E1 - E2).max() <= 1e-5
    x = torch.stack([tf(__import__("PIL.Image", fromlist=["Image"]).open(PROJECT_ROOT / p).convert("RGB")) for p in df["image_path"]])
    with torch.no_grad():
        direct = model(x.to(device)).float().cpu().numpy()
    assert np.abs(direct - L1).max() < 1e-3
