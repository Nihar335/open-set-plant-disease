"""Class-conditional Gaussian / Mahalanobis OOD score with a tied, shrinkage covariance.

Lee et al. (NeurIPS 2018) "A simple unified framework for detecting OOD samples": class means
mu_c and ONE shared covariance Sigma from the training features; score(x) = min_c (x-mu_c)^T
Sigma^-1 (x-mu_c). Higher = farther from every known class = more likely unknown.

Stability: Sigma is the Ledoit-Wolf shrinkage estimate of the class-centred features
(well-conditioned even when n < d), inverted through a symmetric eigendecomposition with
eigenvalues floored at `eig_floor_rel * max_eigenvalue`. Distances are computed in a whitened
space z = x W (W = V diag(lambda^-1/2)), so no explicit inverse is formed. Inputs are validated:
non-finite features raise instead of silently producing NaN distances.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class TiedMahalanobis:
    def __init__(self, eig_floor_rel: float = 1e-8):
        self.eig_floor_rel = eig_floor_rel

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("features must be a finite 2-D array")
        return X

    def fit(self, X, y, n_classes: int):
        X = self._check(X)
        y = np.asarray(y)
        if set(np.unique(y)) != set(range(n_classes)):
            raise ValueError("every class must have training samples")
        self.means_ = np.stack([X[y == c].mean(0) for c in range(n_classes)])
        centred = X - self.means_[y]
        lw = LedoitWolf(assume_centered=True).fit(centred)
        self.shrinkage_ = float(lw.shrinkage_)
        cov = (lw.covariance_ + lw.covariance_.T) / 2
        lam, V = np.linalg.eigh(cov)
        self.eig_raw_min_, self.eig_max_ = float(lam.min()), float(lam.max())
        floor = self.eig_floor_rel * lam.max()
        self.n_floored_ = int((lam < floor).sum())
        lam = np.maximum(lam, floor)
        self.condition_number_ = float(lam.max() / lam.min())
        self.W_ = V / np.sqrt(lam)                       # whitening: z = x @ W
        self.mu_w_ = self.means_ @ self.W_
        self.mu_sq_ = (self.mu_w_ ** 2).sum(1)
        self.n_features_, self.n_classes_ = X.shape[1], n_classes
        return self

    def distances(self, X, chunk: int = 8192) -> np.ndarray:
        """Squared Mahalanobis distance to every class mean, shape [N, C]."""
        X = self._check(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("feature dimension mismatch")
        out = []
        for s in range(0, len(X), chunk):
            z = X[s:s + chunk] @ self.W_
            d = (z ** 2).sum(1, keepdims=True) - 2 * z @ self.mu_w_.T + self.mu_sq_
            out.append(np.maximum(d, 0.0))                # clamp tiny negative round-off
        D = np.concatenate(out)
        if not np.isfinite(D).all():
            raise FloatingPointError("non-finite Mahalanobis distance")
        return D

    def score(self, X):
        """Returns (min squared distance = unknown-ness score, index of nearest class)."""
        D = self.distances(X)
        return D.min(1), D.argmin(1)

    def summary(self) -> dict:
        return {"method": "tied covariance, Ledoit-Wolf shrinkage, eigen-floored whitening",
                "shrinkage": self.shrinkage_, "eig_min_raw": self.eig_raw_min_, "eig_max": self.eig_max_,
                "eig_floor_rel": self.eig_floor_rel, "n_eigenvalues_floored": self.n_floored_,
                "condition_number_after_floor": self.condition_number_, "n_features": self.n_features_,
                "n_classes": self.n_classes_}
