"""Stage-7 tests: SupCon loss, multiple positives, batch construction, unknown exclusion, prototypes, thresholds."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.evaluation.osr_metrics import ranking_metrics, threshold_metrics
from src.training.supcon import ClassBalancedBatchSampler, SupConLoss


def reference_supcon(z, y, t):
    """Literal double-loop implementation of Khosla et al. Eq. 2 (L_out) for checking."""
    n, total, count = len(z), 0.0, 0
    for i in range(n):
        P = [p for p in range(n) if p != i and y[p] == y[i]]
        if not P:
            continue
        denom = sum(np.exp(z[i] @ z[a] / t) for a in range(n) if a != i)
        total += -np.mean([np.log(np.exp(z[i] @ z[p] / t) / denom) for p in P])
        count += 1
    return total / count


def test_supcon_matches_reference_with_multiple_positives():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(12, 8)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2])            # several positives per anchor
    got = SupConLoss(0.1)(torch.tensor(z), torch.tensor(y)).item()
    assert abs(got - reference_supcon(z, y, 0.1)) < 1e-5


def test_supcon_prefers_clustered_embeddings():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = F.normalize(torch.eye(3).repeat_interleave(2, 0) + 0.01, dim=1)
    scrambled = F.normalize(torch.eye(3)[[0, 1, 2, 0, 1, 2]] + 0.01, dim=1)
    loss = SupConLoss(0.07)
    assert loss(clustered, y) < loss(scrambled, y)


def test_supcon_finite_and_skips_anchors_without_positives():
    z = F.normalize(torch.randn(5, 4), dim=1)
    y = torch.tensor([0, 0, 1, 2, 3])                              # only anchors 0,1 have positives
    v = SupConLoss(0.07)(z, y)
    assert torch.isfinite(v)
    z.requires_grad_(True)
    SupConLoss(0.07)(z, y).backward()
    assert torch.isfinite(z.grad).all()


def test_batch_sampler_structure_and_determinism():
    labels = np.repeat(np.arange(21), [104 if c == 13 else 300 for c in range(21)])
    s = ClassBalancedBatchSampler(labels, list(range(21)), 16, 4, batches_per_epoch=20, seed=1)
    batches = list(s)
    assert len(batches) == 20
    for b in batches:
        cls = labels[b]
        assert len(b) == 64 and len(set(cls)) == 16
        assert all((cls == c).sum() == 4 for c in set(cls))       # K images per class -> >= 2*K-1 positives/anchor with 2 views
    assert batches == list(ClassBalancedBatchSampler(labels, list(range(21)), 16, 4, 20, seed=1))
    s.set_epoch(1)
    assert list(s) != batches                                     # new epoch -> new batches


def test_sampler_rejects_unknown_labels():
    with pytest.raises(ValueError):
        ClassBalancedBatchSampler(np.array([0, 1, -1]), [0, 1], 2, 1, 3)


def test_training_manifest_excludes_unknowns():
    from src.data.protocol import known_classes, load_manifest
    from src.utils.config import load_config
    cfg = load_config()
    classes = known_classes(cfg)
    tr = load_manifest("data/splits/train.csv", cfg["baseline"]["dataset"]["row_filter"], "train", classes)
    forbidden = set(cfg["osr_protocol"]["unknown_near"]) | set(cfg["osr_protocol"]["unknown_val"])
    assert not set(tr["class_name"]) & forbidden and set(tr["role"]) == {"known"}
    assert set(tr["dataset"]) == {"plantvillage"} and len(set(tr["class_id"])) == 21


def test_prototype_distance_direction_and_threshold():
    protos = F.normalize(torch.eye(4)[:3], dim=1).numpy()          # 3 known prototypes
    def score(x):                                                 # 1 - max cosine similarity
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return 1 - (x @ protos.T).max(1)
    known = score(np.array([[1, 0.05, 0, 0], [0, 1, 0.05, 0]], float))
    unknown = score(np.array([[0, 0, 0, 1], [0.3, 0.3, 0.3, 1]], float))
    assert ranking_metrics(known, unknown)["auroc"] == 1.0
    m = threshold_metrics(known, np.array([0, 1]), np.array([0, 1]), unknown, t=0.5, n_classes=3)
    assert m["known_acceptance_rate"] == 1.0 and m["unknown_recall"] == 1.0   # accept iff score <= t
