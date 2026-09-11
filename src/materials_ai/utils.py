"""Reproducibility and small shared utilities."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return repository root (two levels above this package file's parent)."""
    return Path(__file__).resolve().parents[2]


def git_commit_hash() -> str | None:
    """Best-effort git commit hash for MLflow tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=project_root(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        return None
    return None


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
