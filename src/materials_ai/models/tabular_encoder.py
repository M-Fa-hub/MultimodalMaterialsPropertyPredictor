"""MLP tabular encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """input features → MLP → tabular embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
