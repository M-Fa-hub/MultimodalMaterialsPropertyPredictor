"""Multimodal band-gap regression model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from materials_ai.models.fusion import build_fusion
from materials_ai.models.tabular_encoder import TabularEncoder
from materials_ai.models.text_encoder import TextEncoder
from materials_ai.models.vision_encoder import VisionEncoder


class MultimodalMaterialsModel(nn.Module):
    """Encode each modality, fuse embeddings, regress property."""

    def __init__(
        self,
        tabular_input_dim: int,
        use_image: bool = True,
        use_tabular: bool = True,
        use_text: bool = False,
        vision_cfg: dict[str, Any] | None = None,
        tabular_cfg: dict[str, Any] | None = None,
        text_cfg: dict[str, Any] | None = None,
        fusion_cfg: dict[str, Any] | None = None,
        head_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not any([use_image, use_tabular, use_text]):
            raise ValueError("At least one modality must be enabled")

        vision_cfg = vision_cfg or {}
        tabular_cfg = tabular_cfg or {}
        text_cfg = text_cfg or {}
        fusion_cfg = fusion_cfg or {}
        head_cfg = head_cfg or {}

        self.use_image = use_image
        self.use_tabular = use_tabular
        self.use_text = use_text

        self.vision: VisionEncoder | None = None
        self.tabular: TabularEncoder | None = None
        self.text: TextEncoder | None = None
        dims: list[int] = []

        if use_image:
            self.vision = VisionEncoder(**vision_cfg)
            dims.append(self.vision.embedding_dim)
        if use_tabular:
            self.tabular = TabularEncoder(input_dim=tabular_input_dim, **tabular_cfg)
            dims.append(self.tabular.embedding_dim)
        if use_text:
            self.text = TextEncoder(**text_cfg)
            dims.append(self.text.embedding_dim)

        self.fusion = build_fusion(
            method=fusion_cfg.get("method", "concat"),
            input_dims=dims,
            hidden_dim=fusion_cfg.get("hidden_dim", 256),
            dropout=fusion_cfg.get("dropout", 0.3),
        )

        hidden = head_cfg.get("hidden_dims", [128, 64])
        dropout = head_cfg.get("dropout", 0.2)
        layers: list[nn.Module] = []
        prev = self.fusion.output_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def encode(
        self,
        image: torch.Tensor | None = None,
        tabular: torch.Tensor | None = None,
        text: list[str] | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        if self.use_image:
            if image is None or self.vision is None:
                raise ValueError("Image modality enabled but image is missing")
            embeddings.append(self.vision(image))
        if self.use_tabular:
            if tabular is None or self.tabular is None:
                raise ValueError("Tabular modality enabled but tabular is missing")
            embeddings.append(self.tabular(tabular))
        if self.use_text:
            if text is None or self.text is None:
                raise ValueError("Text modality enabled but text is missing")
            embeddings.append(self.text(text))
        return self.fusion(embeddings)

    def forward(
        self,
        image: torch.Tensor | None = None,
        tabular: torch.Tensor | None = None,
        text: list[str] | None = None,
    ) -> torch.Tensor:
        fused = self.encode(image=image, tabular=tabular, text=text)
        return self.head(fused).squeeze(-1)

    @property
    def inputs_used(self) -> list[str]:
        used = []
        if self.use_image:
            used.append("image")
        if self.use_tabular:
            used.append("tabular")
        if self.use_text:
            used.append("text")
        return used
