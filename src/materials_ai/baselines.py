"""Baseline tabular models for band-gap regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

from materials_ai.data.dataset import load_processed_frame
from materials_ai.data.descriptors import feature_names
from materials_ai.logging_utils import get_logger
from materials_ai.mlflow_utils import setup_mlflow
from materials_ai.training.evaluation import regression_metrics
from materials_ai.utils import ensure_dir, git_commit_hash, set_seed, to_jsonable

logger = get_logger(__name__)


def _xy(df: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    cols = [f"scaled_{c}" for c in feature_names()]
    if not all(c in df.columns for c in cols):
        cols = feature_names()
    sub = df.loc[df["split"] == split]
    return sub[cols].to_numpy(dtype=np.float32), sub["band_gap"].to_numpy(dtype=np.float32)


class TabularMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_tabular_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[TabularMLPRegressor, dict[str, float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLPRegressor(x_train.shape[1], hidden_dims, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train.copy()), torch.from_numpy(y_train.copy())),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = None
    best_mae = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
        mae = regression_metrics(y_val, pred)["mae"]
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
    return model, regression_metrics(y_val, val_pred)


def run_baselines(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "outputs/models/baselines",
    metrics_path: str | Path = "outputs/figures/baseline_metrics.json",
    seed: int = 42,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train dummy / RF / ExtraTrees / XGBoost / tabular MLP baselines."""
    set_seed(seed)
    cfg = cfg or {}
    df = load_processed_frame(processed_dir)
    x_train, y_train = _xy(df, "train")
    x_val, y_val = _xy(df, "val")
    x_test, y_test = _xy(df, "test")
    out_dir = ensure_dir(output_dir)

    setup_mlflow(cfg.get("mlflow_experiment", "materials-bandgap-baselines"))

    results: list[dict[str, Any]] = []

    def _eval_and_store(name: str, predict_fn, artifact: Any | None = None) -> None:
        pred = predict_fn(x_test)
        metrics = regression_metrics(y_test, pred)
        row = {"model": name, **metrics}
        results.append(row)
        logger.info("%s | MAE=%.4f RMSE=%.4f R2=%.4f", name, metrics["mae"], metrics["rmse"], metrics["r2"])
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("git_commit", git_commit_hash())
            mlflow.log_metrics(metrics)
        if artifact is not None:
            joblib.dump(artifact, out_dir / f"{name.replace(' ', '_').lower()}.joblib")

    # 1. Dummy mean
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    _eval_and_store("Mean baseline", dummy.predict, dummy)

    # 2. Random Forest
    rf_cfg = cfg.get("random_forest", {})
    rf = RandomForestRegressor(random_state=seed, **rf_cfg)
    rf.fit(x_train, y_train)
    _eval_and_store("Random Forest", rf.predict, rf)

    # 3. Extra Trees
    et_cfg = cfg.get("extra_trees", {})
    et = ExtraTreesRegressor(random_state=seed, **et_cfg)
    et.fit(x_train, y_train)
    _eval_and_store("Extra Trees", et.predict, et)

    # 4. XGBoost if available
    try:
        from xgboost import XGBRegressor

        xgb_cfg = cfg.get("xgboost", {})
        xgb = XGBRegressor(random_state=seed, objective="reg:squarederror", **xgb_cfg)
        xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        _eval_and_store("XGBoost", xgb.predict, xgb)
    except Exception as exc:
        logger.warning("XGBoost skipped: %s", exc)

    # 5. Tabular MLP
    mlp_cfg = cfg.get("tabular_mlp", {})
    mlp, _ = train_tabular_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dims=mlp_cfg.get("hidden_dims", [256, 128, 64]),
        dropout=mlp_cfg.get("dropout", 0.2),
        epochs=mlp_cfg.get("epochs", 50),
        batch_size=mlp_cfg.get("batch_size", 64),
        lr=mlp_cfg.get("lr", 1e-3),
        weight_decay=mlp_cfg.get("weight_decay", 1e-4),
        patience=mlp_cfg.get("early_stopping_patience", 10),
        seed=seed,
    )
    device = next(mlp.parameters()).device

    def mlp_predict(x: np.ndarray) -> np.ndarray:
        mlp.eval()
        with torch.no_grad():
            return mlp(torch.from_numpy(x).to(device)).cpu().numpy()

    torch.save(mlp.state_dict(), out_dir / "tabular_mlp.pt")
    _eval_and_store("Tabular MLP", mlp_predict)

    metrics_path = Path(metrics_path)
    ensure_dir(metrics_path.parent)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results
