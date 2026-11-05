"""Stage-3 pre-training checks (run before any long training):

    python -m pytest tests/test_baseline_pipeline.py -q -s
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.data.dataset import ManifestDataset, build_transforms
from src.data.protocol import known_classes, load_manifest, verify_lock
from src.models.classifier import Classifier
from src.training.trainer import class_weights
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def classes(cfg):
    return known_classes(cfg)


@pytest.fixture(scope="module")
def frames(cfg, classes):
    d = cfg["baseline"]["dataset"]
    return {s: load_manifest(d[f"{s}_manifest"], d["row_filter"], s, classes) for s in ("train", "val", "test")}


def test_protocol_lock_and_manifests(cfg, frames):                       # 1 manifest loading
    verify_lock(cfg)
    n = {s: len(f) for s, f in frames.items()}
    print("known rows:", n)
    assert n == {"train": 19126, "val": 4116, "test": 4091}               # Stage-2 locked counts
    for s, f in frames.items():
        assert set(f["role"]) == {"known"} and set(f["dataset"]) == {"plantvillage"}, s
    assert not set(frames["train"]["image_path"]) & set(frames["test"]["image_path"])


def test_class_mapping(cfg, classes, frames):                            # 2 class mapping, 7 n classes
    assert len(classes) == cfg["baseline"]["model"]["num_classes"] == 21
    for f in frames.values():
        assert sorted(f["class_id"].unique()) == list(range(21))           # every class in every split


def test_dataset_and_transforms(cfg, frames):                            # 3 dataset, 4 transforms
    b = cfg["baseline"]
    ds = ManifestDataset(frames["train"].head(4), build_transforms(b["training"], b["augmentation"], True))
    x, y, i = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and 0 <= y < 21 and i == 0
    ev = ManifestDataset(frames["val"].head(2), build_transforms(b["training"], b["augmentation"], False))
    assert torch.equal(ev[0][0], ev[0][0])                                 # eval transform deterministic


def test_batch_forward_backward(cfg, frames):                            # 5 batch, 6 labels, 8 cuda, 9-10
    b = cfg["baseline"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    ds = ManifestDataset(frames["train"].sample(b["training"]["batch_size"], random_state=0),
                         build_transforms(b["training"], b["augmentation"], True))
    loader = torch.utils.data.DataLoader(ds, batch_size=b["training"]["batch_size"], num_workers=0)
    x, y, _ = next(iter(loader))
    assert x.shape == (b["training"]["batch_size"], 3, 224, 224)
    assert int(y.min()) >= 0 and int(y.max()) < 21
    model = Classifier(b["model"]["architecture"], 21, pretrained=True).to(device)
    model.freeze_backbone_except(b["training"]["phase_b"]["unfreeze"])
    model.train()
    w = class_weights(frames["train"]["class_id"].to_numpy(), 21, b["training"]["class_weight_power"])
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(x.to(device))
        loss = crit(out, y.to(device))
    print(f"Input: {list(x.shape)}  Output: {list(out.shape)}  loss {loss.item():.3f}")
    assert out.shape == (b["training"]["batch_size"], 21)
    loss.backward()
    assert model.head.weight.grad is not None and model.backbone.layer4[0].conv1.weight.grad is not None
    assert model.backbone.conv1.weight.grad is None                        # frozen stem gets no gradient
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def test_class_weights_from_train_only(frames):
    w = class_weights(frames["train"]["class_id"].to_numpy(), 21, 0.5)
    assert abs(float(w.mean()) - 1) < 1e-6 and float(w.max() / w.min()) > 1
