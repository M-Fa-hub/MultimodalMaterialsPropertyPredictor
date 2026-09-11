"""PyTorch training loop for multimodal regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from materials_ai.logging_utils import get_logger
from materials_ai.mlflow_utils import setup_mlflow
from materials_ai.training.checkpointing import load_checkpoint, save_checkpoint
from materials_ai.training.evaluation import regression_metrics
from materials_ai.utils import ensure_dir, git_commit_hash, set_seed, to_jsonable

logger = get_logger(__name__)


@dataclass
class TrainResult:
    best_val_mae: float
    history: list[dict[str, float]]
    checkpoint_path: Path
    test_metrics: dict[str, float]


class EarlyStopping:
    def __init__(self, patience: int = 8) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if metric < self.best - 1e-6:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _batch_forward(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device) if getattr(model, "use_image", False) else None
    tabular = batch["tabular"].to(device) if getattr(model, "use_tabular", False) else None
    text = batch["text"] if getattr(model, "use_text", False) else None
    return model(image=image, tabular=tabular, text=text)


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        out = _batch_forward(model, batch, device)
        preds.append(out.detach().cpu().numpy())
        targets.append(batch["target"].numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_multimodal(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    *,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    mixed_precision: bool = True,
    early_stopping_patience: int = 8,
    scheduler_name: str = "cosine",
    checkpoint_dir: str | Path = "outputs/models",
    checkpoint_name: str = "multimodal_best.pt",
    resume: str | None = None,
    seed: int = 42,
    mlflow_experiment: str = "materials-bandgap-multimodal",
    run_name: str = "multimodal",
    extra_params: dict[str, Any] | None = None,
) -> TrainResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val = float("inf")
    ckpt_path = ensure_dir(checkpoint_dir) / checkpoint_name

    if resume:
        payload = load_checkpoint(resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_metric") or best_val)
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    if scheduler_name == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(patience=early_stopping_patience)
    history: list[dict[str, float]] = []

    setup_mlflow(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        params = {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": loaders["train"].batch_size,
            "device": str(device),
            "git_commit": git_commit_hash(),
            **(extra_params or {}),
        }
        mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

        for epoch in range(start_epoch, epochs):
            model.train()
            losses: list[float] = []
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                target = batch["target"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = _batch_forward(model, batch, device)
                    loss = criterion(pred, target)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))

            train_loss = float(np.mean(losses)) if losses else float("nan")
            val_pred, val_true = predict_loader(model, loaders["val"], device)
            val_metrics = regression_metrics(val_true, val_pred)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            mlflow.log_metrics(to_jsonable(row), step=epoch)
            logger.info(
                "Epoch %d | train_loss=%.4f val_mae=%.4f val_rmse=%.4f val_r2=%.4f",
                epoch,
                train_loss,
                val_metrics["mae"],
                val_metrics["rmse"],
                val_metrics["r2"],
            )

            if val_metrics["mae"] < best_val:
                best_val = val_metrics["mae"]
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    best_metric=best_val,
                    meta={"inputs_used": getattr(model, "inputs_used", [])},
                )

            if scheduler is not None:
                scheduler.step()
            if stopper.step(val_metrics["mae"]):
                logger.info("Early stopping at epoch %d", epoch)
                break

        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model, map_location=device)
        test_pred, test_true = predict_loader(model, loaders["test"], device)
        test_metrics = regression_metrics(test_true, test_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info("Test metrics: %s", test_metrics)

    return TrainResult(
        best_val_mae=best_val,
        history=history,
        checkpoint_path=ckpt_path,
        test_metrics=test_metrics,
    )
