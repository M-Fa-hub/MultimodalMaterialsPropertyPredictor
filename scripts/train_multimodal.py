#!/usr/bin/env python3
"""Train the multimodal band-gap model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materials_ai.config import load_config
from materials_ai.data.dataset import create_dataloaders
from materials_ai.logging_utils import setup_logging
from materials_ai.model_factory import build_multimodal_model, checkpoint_meta_from_cfg
from materials_ai.training.checkpointing import save_checkpoint
from materials_ai.training.trainer import train_multimodal
from materials_ai.utils import to_jsonable


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default="image_tabular")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    loaders = create_dataloaders(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
    )
    model = build_multimodal_model(cfg)
    meta = checkpoint_meta_from_cfg(cfg, model)

    result = train_multimodal(
        model,
        loaders,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        mixed_precision=cfg.training.mixed_precision,
        early_stopping_patience=cfg.training.early_stopping_patience,
        scheduler_name=cfg.training.scheduler,
        checkpoint_dir=cfg.training.checkpoint_dir,
        checkpoint_name=cfg.training.checkpoint_name,
        resume=cfg.training.resume,
        seed=cfg.seed,
        mlflow_experiment=cfg.output.mlflow_experiment,
        run_name=args.run_name,
        extra_params={
            "dataset_version": cfg.data.dataset_version,
            "modalities": ",".join(model.inputs_used),
            "fusion": cfg.model.fusion.method,
        },
    )

    # Re-save with full architecture meta for inference reconstruction
    save_checkpoint(
        result.checkpoint_path,
        model,
        epoch=0,
        best_metric=result.best_val_mae,
        meta=meta,
    )

    metrics_path = Path(cfg.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Image + Tabular",
        **result.test_metrics,
        "checkpoint": str(result.checkpoint_path),
        "inputs_used": model.inputs_used,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
