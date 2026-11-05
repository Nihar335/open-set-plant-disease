"""Supervised contrastive learning components (Khosla et al., NeurIPS 2020).

* SupConLoss: L = sum_i  -1/|P(i)| sum_{p in P(i)} log( exp(z_i.z_p/t) / sum_{a != i} exp(z_i.z_a/t) )
  over all 2N views of a batch; positives P(i) = every OTHER view with the same label (the second
  view of the same image AND views of other images of the same known class). Anchors without a
  positive are skipped. Only known classes ever appear, so no "unknown" negative class exists.
* ClassBalancedBatchSampler: each batch = P distinct known classes x K images per class
  (sampled with replacement only if a class has < K images). Seeded, reproducible.
* TwoViewDataset: returns two independently augmented views of the SAME source image, so all
  views of an image belong to the split of that image (no augmentation leakage across splits).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """z: [M, d] L2-normalised embeddings of all views; labels: [M]."""
        z = z.float()
        n = z.shape[0]
        sim = z @ z.T / self.t
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, -1e9)                         # exclude i == a from the denominator
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (labels[:, None] == labels[None, :]) & ~self_mask
        n_pos = pos.sum(1)
        valid = n_pos > 0
        mean_log_pos = (log_prob * pos).sum(1)[valid] / n_pos[valid]
        return -mean_log_pos.mean()


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, allowed_classes, classes_per_batch: int, samples_per_class: int,
                 batches_per_epoch: int, seed: int = 0):
        labels = np.asarray(labels)
        bad = set(np.unique(labels)) - set(allowed_classes)
        if bad:
            raise ValueError(f"labels outside the known classes: {sorted(bad)}")
        self.by_class = {c: np.flatnonzero(labels == c) for c in allowed_classes}
        if any(len(v) == 0 for v in self.by_class.values()):
            raise ValueError("a known class has no training images")
        if classes_per_batch > len(self.by_class):
            raise ValueError("classes_per_batch exceeds the number of classes")
        self.P, self.K, self.n, self.seed, self.epoch = classes_per_batch, samples_per_class, batches_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        classes = np.array(sorted(self.by_class))
        for _ in range(self.n):
            batch = []
            for c in rng.choice(classes, self.P, replace=False):
                pool = self.by_class[c]
                batch += list(rng.choice(pool, self.K, replace=len(pool) < self.K))
            yield [int(i) for i in batch]


class TwoViewDataset(Dataset):
    """Wraps a ManifestDataset WITHOUT transform; returns (view1, view2, label, index)."""

    def __init__(self, base, transform, views: int = 2):
        self.base, self.transform, self.views = base, transform, views

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y, idx = self.base[i]                           # base has transform=None -> PIL image
        return torch.stack([self.transform(img) for _ in range(self.views)]), y, idx


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden=2048, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
