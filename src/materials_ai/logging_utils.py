"""Structured logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def setup_logging(level: str | None = None, name: str = "materials_ai") -> logging.Logger:
    """Configure root logger for the package."""
    raw_level = level or os.getenv("MATERIALS_AI_LOG_LEVEL") or "INFO"
    log_level = raw_level.upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False
    return logger


def get_logger(name: str = "materials_ai") -> logging.Logger:
    """Return a named logger, ensuring handlers exist."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name=name)
    return logger


def log_dict(logger: logging.Logger, title: str, payload: dict[str, Any]) -> None:
    """Log a flat dictionary as key=value pairs."""
    parts = [f"{k}={v}" for k, v in payload.items()]
    logger.info("%s | %s", title, " ".join(parts))
