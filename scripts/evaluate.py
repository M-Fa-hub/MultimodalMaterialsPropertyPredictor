#!/usr/bin/env python3
"""Evaluate a trained multimodal checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from materials_ai.config import load_config
from materials_ai.data.dataset import create_dataloaders
from materials_ai.logging_utils import setup_logging
from materials_ai.model_factory import build_multimodal_model
from materials_ai.training.checkpointing import load_checkpoint
from materials_ai.training.evaluation import regression_metrics
from materials_ai.training.trainer import predict_loader
from materials_ai.utils import ensure_dir, to_jsonable


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate multimodal checkpoint")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/multimodal_best.pt")
    parser.add_argument("--out", default="outputs/figures/eval_scatter.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loaders = create_dataloaders(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    inputs = meta.get("inputs_used", ["image", "tabular"])
    model = build_multimodal_model(
        cfg,
        modalities={
            "image": "image" in inputs,
            "tabular": "tabular" in inputs,
            "text": "text" in inputs,
        },
    )
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)

    preds, targets = predict_loader(model, loaders["test"], device)
    metrics = regression_metrics(targets, preds)
    print(json.dumps(to_jsonable(metrics), indent=2))

    out = Path(args.out)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.7, edgecolor="k", linewidth=0.3)
    lims = [float(min(targets.min(), preds.min())), float(max(targets.max(), preds.max()))]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("True band gap (eV)")
    ax.set_ylabel("Predicted band gap (eV)")
    ax.set_title(f"Test set | MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
