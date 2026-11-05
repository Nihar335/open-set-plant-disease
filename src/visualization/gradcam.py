"""Minimal Grad-CAM (Selvaraju et al., 2017) on a model's `gradcam_layer`."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.acts, self.grads = model, None, None
        self._h = [layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o)),
                   layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """x: (1,3,H,W). Returns (heat-map in [0,1] at input resolution, class used)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        c = int(logits.argmax(1)) if class_idx is None else class_idx
        self.model.zero_grad(set_to_none=True)
        logits[0, c].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), c

    def remove(self) -> None:
        for h in self._h:
            h.remove()
