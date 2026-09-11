#!/usr/bin/env python3
"""Run modality ablation experiments and write a comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materials_ai.config import load_config
from materials_ai.data.dataset import create_dataloaders
from materials_ai.logging_utils import setup_logging
from materials_ai.model_factory import build_multimodal_model, checkpoint_meta_from_cfg
from materials_ai.training.checkpointing import save_checkpoint
from materials_ai.training.evaluation import format_metrics_table
from materials_ai.training.trainer import train_multimodal
from materials_ai.utils import ensure_dir, to_jsonable


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ablation study across modalities")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for faster runs")
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

    runs = cfg.ablation.get("runs", [])
    # Version 1 default: skip text-enabled run unless explicitly wanted and small
    results: list[dict] = []
    for run in runs:
        mods = run.get("modalities", {})
        # Skip text run in V1 unless text modality dependencies desired
        if mods.get("text") and not cfg.modalities.text:
            # Still allow architectural demo with hash text encoder
            pass
        name = run["name"]
        print(f"\n=== Ablation: {name} ===")
        model = build_multimodal_model(cfg, modalities=mods)
        ckpt_name = f"ablation_{name}.pt"
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
            checkpoint_name=ckpt_name,
            seed=cfg.seed,
            mlflow_experiment=cfg.output.mlflow_experiment + "-ablation",
            run_name=name,
            extra_params={"ablation": name, "modalities": ",".join(model.inputs_used)},
        )
        save_checkpoint(
            result.checkpoint_path,
            model,
            best_metric=result.best_val_mae,
            meta=checkpoint_meta_from_cfg(cfg, model),
        )
        label = {
            "tabular_only": "Tabular only",
            "image_only": "Image only",
            "image_tabular": "Image + Tabular",
            "image_tabular_text": "Image + Tabular + Text",
        }.get(name, name)
        results.append({"model": label, **result.test_metrics, "checkpoint": str(result.checkpoint_path)})

    out = Path(cfg.output.ablation_path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print("\n" + format_metrics_table(results))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
