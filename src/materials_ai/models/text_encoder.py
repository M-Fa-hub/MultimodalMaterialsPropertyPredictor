"""Optional lightweight text encoder (architecture-ready for Version 2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HashTextEncoder(nn.Module):
    """
    Deterministic bag-of-character hash embedding fallback.

    Used when sentence-transformers is not installed so image+tabular V1
    remains runnable while keeping a text pathway in the architecture.
    """

    def __init__(self, embedding_dim: int = 128, n_buckets: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_buckets = n_buckets
        self.proj = nn.Sequential(
            nn.Linear(n_buckets, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _hash_batch(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            vec = torch.zeros(self.n_buckets, dtype=torch.float32)
            for ch in text.lower():
                vec[hash(ch) % self.n_buckets] += 1.0
            if vec.sum() > 0:
                vec = vec / vec.norm(p=2).clamp_min(1e-6)
            rows.append(vec)
        return torch.stack(rows, dim=0)

    def forward(self, texts: list[str] | torch.Tensor) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            # Already embedded externally
            return self.proj(texts)
        device = next(self.parameters()).device
        hashed = self._hash_batch(texts).to(device)
        return self.proj(hashed)


class TextEncoder(nn.Module):
    """
    Text encoder wrapper.

    Tries sentence-transformers when available; otherwise uses HashTextEncoder.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 128,
        freeze: bool = True,
        use_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backend = "hash"
        self.transformer = None
        self.proj: nn.Module

        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self.transformer = SentenceTransformer(model_name)
                in_dim = self.transformer.get_sentence_embedding_dimension()
                if freeze:
                    for p in self.transformer.parameters():
                        p.requires_grad = False
                self.proj = nn.Linear(in_dim, embedding_dim)
                self.backend = "sentence_transformers"
            except Exception:
                self.proj = HashTextEncoder(embedding_dim=embedding_dim)
                self.backend = "hash"
        else:
            self.proj = HashTextEncoder(embedding_dim=embedding_dim)

    def forward(self, texts: list[str]) -> torch.Tensor:
        if self.backend == "sentence_transformers" and self.transformer is not None:
            device = next(self.proj.parameters()).device
            with torch.set_grad_enabled(self.transformer.training):
                emb = self.transformer.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=str(device),
                )
            return self.proj(emb)
        return self.proj(texts)
