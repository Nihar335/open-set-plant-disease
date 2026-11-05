"""Stage 11 multi-task network: ResNet-50 backbone + disease head + symptom head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def letterbox(img, size: int = IMG_SIZE, fill=(124, 116, 104)):
    """Whole photo scaled to fit a size x size square, padded with the ImageNet mean colour (nothing cut off)."""
    from PIL import Image
    img = img.copy()
    img.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


# Second view used with EVAL_TF: phone photos are often portrait or off-centre, and the centre crop cuts the leaf.
WHOLE_TF = transforms.Compose([
    transforms.Lambda(letterbox),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class LeafNet(nn.Module):
    """Shared ResNet-50 features -> (disease logits, symptom logits)."""

    def __init__(self, n_classes: int, n_symptoms: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-2])  # up to layer4, keeps spatial map for Grad-CAM
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.disease_head = nn.Linear(2048, n_classes)
        self.symptom_head = nn.Linear(2048, n_symptoms)

    def forward(self, x, return_map: bool = False):
        fmap = self.features(x)
        z = self.dropout(torch.flatten(self.pool(fmap), 1))
        out = (self.disease_head(z), self.symptom_head(z))
        return (*out, fmap) if return_map else out
