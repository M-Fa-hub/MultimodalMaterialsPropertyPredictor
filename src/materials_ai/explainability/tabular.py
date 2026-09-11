"""Tabular explainability: feature importance and optional SHAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from materials_ai.data.descriptors import feature_names
from materials_ai.logging_utils import get_logger
from materials_ai.utils import ensure_dir

logger = get_logger(__name__)


def tree_feature_importance(
    model_path: str | Path,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Load a sklearn tree model and return top feature importances."""
    model = joblib.load(model_path)
    names = feature_names()
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_")
    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1][:top_k]
    return [{"feature": names[i], "importance": float(importances[i])} for i in order]


def shap_explanation(
    model: Any,
    x_row: np.ndarray,
    feature_cols: list[str] | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Compute feature attributions for a single row.

    Prefers tree feature importances weighted by |feature value|, then SHAP,
    then magnitude proxy.
    """
    feature_cols = feature_cols or feature_names()
    x = np.asarray(x_row, dtype=float).ravel()

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        scores = importances * (np.abs(x) + 1e-6)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "feature": feature_cols[i],
                "importance": float(importances[i]),
                "contribution": float(scores[i]),
            }
            for i in order
            if i < len(feature_cols)
        ]

    try:
        import shap

        if hasattr(model, "predict"):
            explainer = shap.Explainer(model.predict, x.reshape(1, -1))
            values = explainer(x.reshape(1, -1))
            shap_vals = np.asarray(values.values).ravel()
            order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
            return [
                {"feature": feature_cols[i], "contribution": float(shap_vals[i])}
                for i in order
                if i < len(feature_cols)
            ]
    except Exception as exc:
        logger.warning("SHAP unavailable (%s); using magnitude proxy", exc)

    shap_vals = np.abs(x)
    order = np.argsort(shap_vals)[::-1][:top_k]
    return [
        {"feature": feature_cols[i], "contribution": float(shap_vals[i])}
        for i in order
        if i < len(feature_cols)
    ]


def write_tabular_report(
    out_path: str | Path,
    prediction: float,
    true_value: float | None,
    factors: list[dict[str, Any]],
    unit: str = "eV",
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lines = [
        f"Predicted band gap: {prediction:.2f} {unit}",
    ]
    if true_value is not None:
        lines.append(f"True band gap: {true_value:.2f} {unit}")
        lines.append(f"Absolute error: {abs(prediction - true_value):.2f} {unit}")
    lines.append("")
    lines.append("Important tabular factors:")
    for item in factors:
        name = item.get("feature", "?")
        val = item.get("contribution", item.get("importance", 0.0))
        lines.append(f"  - {name}: {val:.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
