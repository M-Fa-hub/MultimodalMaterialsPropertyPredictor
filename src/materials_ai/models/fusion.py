"""Multimodal fusion layers (concat now; gated/attention-ready interface)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Base fusion interface."""

    output_dim: int

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ConcatFusion(FusionModule):
    """Concatenate modality embeddings then project."""

    def __init__(self, input_dims: list[int], hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        total = sum(input_dims)
        self.net = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(embeddings, dim=-1)
        return self.net(x)


class GatedFusion(FusionModule):
    """Learned gates over modality embeddings (extensibility demo)."""

    def __init__(self, input_dims: list[int], hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, hidden_dim) for d in input_dims])
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, hidden_dim), nn.Sigmoid()) for d in input_dims]
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.output_dim = hidden_dim

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        fused = 0
        for emb, proj, gate in zip(embeddings, self.projs, self.gates, strict=True):
            fused = fused + proj(emb) * gate(emb)
        return self.out(fused)  # type: ignore[arg-type]


def build_fusion(method: str, input_dims: list[int], hidden_dim: int, dropout: float) -> FusionModule:
    method = method.lower()
    if method == "concat":
        return ConcatFusion(input_dims, hidden_dim=hidden_dim, dropout=dropout)
    if method == "gated":
        return GatedFusion(input_dims, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown fusion method: {method}")
