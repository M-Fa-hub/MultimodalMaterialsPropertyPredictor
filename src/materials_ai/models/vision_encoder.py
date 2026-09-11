"""Pretrained vision backbone → fixed-size image embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VisionEncoder(nn.Module):
    """Lightweight pretrained vision encoder with projection head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = True,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        weights = "DEFAULT" if pretrained else None

        if backbone == "resnet18":
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = net
        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.encoder = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return self.proj(feats)

    def unfreeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
