"""Regression metrics and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R²."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def format_metrics_table(rows: list[dict[str, Any]]) -> str:
    """Render a markdown comparison table."""
    header = "| Model | MAE | RMSE | R² |\n| --- | ---: | ---: | ---: |"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} |"
        )
    return "\n".join(lines)
