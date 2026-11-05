"""Frozen research-prototype predictor: Stage-3 classifier + OSR novelty scores with the frozen validation thresholds.

Nothing here is trained or tuned. It loads:
  * models/baseline/best_model.pt            (Stage 3, sha256-checked)  -> known-disease prediction, MSP, Energy
  * models/contrastive/best_model.pt         (Stage 7, sha256-checked)  -> 2048-d contrastive features
  * models/contrastive/linear_head.npz       (Stage-7 linear head, exported by scripts/export_linear_head.py)
  * thresholds from results/metrics/{osr_msp,osr_energy,contrastive}_threshold(s).json (known-validation, 95% acceptance)

Decision rule (per score): accept as KNOWN iff score <= tau, else "potentially unseen / out-of-distribution".
The headline novelty score is contrastive Energy (best measured single detector, chosen post hoc on the test set;
its gain over original Energy is mostly due to the linear head). The original Energy decision is shown alongside.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.data.dataset import build_transforms
from src.models.classifier import Classifier
from src.osr.scores import energy_score, msp_scores
from src.training.supcon import ProjectionHead
from src.utils.config import load_config, resolve

UNKNOWN_MSG = "Potentially unseen / out-of-distribution input"
KNOWN_MSG = "Known disease (within the 21 trained classes)"


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class _ContrastiveEncoder(nn.Module):
    """Same module layout as scripts/train_contrastive.ContrastiveNet, built without downloading ImageNet weights."""

    def __init__(self, proj):
        super().__init__()
        self.enc = Classifier("resnet50", 21, pretrained=False)
        self.proj = ProjectionHead(self.enc.feature_dim, proj["hidden"], proj["out"])

    def features(self, x):
        return self.enc.extract_features(x)


class OSRPredictor:
    def __init__(self, device: str | None = None):
        cfg = load_config()
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        M = resolve(cfg["paths"]["metrics"])
        # Stage-3 classifier
        p3 = resolve(cfg["osr_msp"]["checkpoint"])
        if _sha256(p3) != cfg["osr_msp"]["checkpoint_sha256"]:
            raise RuntimeError("Stage-3 checkpoint hash differs from the frozen value")
        ck = torch.load(p3, map_location="cpu", weights_only=False)
        self.classes = list(ck["class_names"])
        self.m3 = Classifier(ck["architecture"], ck["num_classes"], pretrained=False)
        self.m3.load_state_dict(ck["state_dict"]); self.m3.to(self.device).eval()
        s3 = yaml.safe_load(resolve(cfg["osr_msp"]["stage3_config"]).read_text(encoding="utf-8"))["baseline"]
        self.tf = build_transforms(s3["training"], s3["augmentation"], train=False)
        # Stage-7 contrastive encoder + linear head
        env7 = json.loads((M / "contrastive_environment.json").read_text())
        p7 = resolve("models/contrastive/best_model.pt")
        if _sha256(p7) != env7["checkpoint_sha256"]:
            raise RuntimeError("Contrastive checkpoint hash differs from the frozen value")
        ck7 = torch.load(p7, map_location="cpu", weights_only=False)
        self.m7 = _ContrastiveEncoder(cfg["contrastive"]["projection"])
        self.m7.load_state_dict(ck7["state_dict"]); self.m7.to(self.device).eval()
        lh = resolve("models/contrastive/linear_head.npz")
        if not lh.exists():
            raise RuntimeError("models/contrastive/linear_head.npz missing — run: python -P scripts/export_linear_head.py")
        z = np.load(lh)
        assert list(z["classes"]) == self.classes
        self.coef, self.intercept = z["coef"].astype(np.float64), z["intercept"].astype(np.float64)
        # frozen thresholds (known validation, 95% acceptance)
        self.t_msp_u = json.loads((M / "osr_msp_threshold.json").read_text())["threshold_u"]
        self.t_energy = json.loads((M / "osr_energy_threshold.json").read_text())["primary"]["threshold_energy"]
        self.t_cenergy = json.loads((M / "contrastive_thresholds.json").read_text())["thresholds"]["contrastive::energy_linear_head"]
        ref = np.load(M / "fusion_normalisation.npz")
        self.ref_cenergy = np.sort(ref["contrastive_energy"])        # known-validation reference (oriented: higher = unknown)
        self.ref_energy = np.sort(ref["orig_energy"])

    @staticmethod
    def _pct(ref, s):
        lo, hi = np.searchsorted(ref, s, "left"), np.searchsorted(ref, s, "right")
        return float((lo + 0.5 * (hi - lo)) / len(ref))

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path) -> dict:
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.m3(x).float().cpu().numpy().astype(np.float64)
        ms = msp_scores(logits)
        e = float(energy_score(logits, 1.0)[0])
        f = self.m7.features(x).float().cpu().numpy().astype(np.float64)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        L = f @ self.coef.T + self.intercept
        ce = float(energy_score(L, 1.0)[0])
        probs = ms["probs"][0]
        top = np.argsort(-probs)[:3]
        unknown = ce > self.t_cenergy
        return {
            "predicted_known_class": self.classes[int(ms["pred"][0])],
            "top3": [(self.classes[i], float(probs[i])) for i in top],
            "msp": float(ms["msp"][0]), "msp_threshold": 1 - self.t_msp_u, "msp_says_unknown": bool(ms["u"][0] > self.t_msp_u),
            "energy": e, "energy_threshold": self.t_energy, "energy_says_unknown": bool(e > self.t_energy),
            "energy_percentile_known_val": self._pct(self.ref_energy, e),
            "contrastive_energy": ce, "contrastive_energy_threshold": self.t_cenergy, "contrastive_energy_says_unknown": bool(unknown),
            "contrastive_energy_percentile_known_val": self._pct(self.ref_cenergy, ce),
            "decision": "UNKNOWN / NOVEL" if unknown else "KNOWN DISEASE",
            "message": UNKNOWN_MSG if unknown else KNOWN_MSG,
        }
