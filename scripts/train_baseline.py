#!/usr/bin/env python3
"""Train tabular baseline models."""

from __future__ import annotations

import argparse

from materials_ai.baselines import run_baselines
from materials_ai.config import load_yaml
from materials_ai.logging_utils import setup_logging
from materials_ai.training.evaluation import format_metrics_table


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    results = run_baselines(
        processed_dir=cfg.get("data", {}).get("processed_dir", "data/processed"),
        output_dir=cfg.get("output", {}).get("models_dir", "outputs/models/baselines"),
        metrics_path=cfg.get("output", {}).get("metrics_path", "outputs/figures/baseline_metrics.json"),
        seed=cfg.get("seed", 42),
        cfg={**cfg.get("models", {}), "mlflow_experiment": cfg.get("output", {}).get("mlflow_experiment")},
    )
    print(format_metrics_table(results))


if __name__ == "__main__":
    main()
