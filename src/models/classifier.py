"""Pretrained torchvision classifier with staged fine-tuning helpers.

`model.gradcam_layer` points at the last convolutional stage (ResNet `layer4`) so Grad-CAM
can hook it later; `extract_features` returns the pooled penultimate embedding for OSR methods.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

WEIGHTS = {"resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
           "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1}


class Classifier(nn.Module):
    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if architecture not in WEIGHTS:
            raise ValueError(f"unsupported architecture {architecture}")
        net = getattr(torchvision.models, architecture)(weights=WEIGHTS[architecture] if pretrained else None)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Linear(self.feature_dim, num_classes)
        self.architecture = architecture
        self._frozen: list[nn.Module] = []

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.backbone.layer4

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ---------------- staged fine-tuning
    def freeze_backbone_except(self, trainable: list[str]) -> None:
        """Freeze every backbone child except those named (e.g. ['layer3','layer4']); head stays trainable."""
        self._frozen = []
        for name, child in self.backbone.named_children():
            train = name in trainable
            for p in child.parameters():
                p.requires_grad = train
            if not train:
                self._frozen.append(child)
        for p in self.head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._frozen:          # frozen stages keep ImageNet BatchNorm statistics
            m.eval()
        return self

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": list(self.head.parameters()), "lr": head_lr, "name": "head"}]
        if bb:
            groups.append({"params": bb, "lr": backbone_lr, "name": "backbone"})
        return groups
