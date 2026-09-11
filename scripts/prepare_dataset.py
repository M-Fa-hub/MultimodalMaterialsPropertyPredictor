#!/usr/bin/env python3
"""Acquire and prepare the multimodal materials dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from materials_ai.data.download import acquire_dataset
from materials_ai.data.preprocessing import prepare_dataset
from materials_ai.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Prepare materials band-gap dataset")
    parser.add_argument("--source", default="demo", choices=["demo", "mp", "materials_project"])
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument(
        "--split-strategy",
        default=None,
        choices=["random", "composition"],
        help=(
            "composition reduces chemical leakage (preferred for Materials Project). "
            "For the small demo corpus, random is used by default because there are "
            "few unique element-set groups."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    split_strategy = args.split_strategy
    if split_strategy is None:
        split_strategy = "random" if args.source == "demo" else "composition"

    manifest = acquire_dataset(
        raw_dir=args.raw_dir,
        source=args.source,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    paths = prepare_dataset(
        raw_manifest=manifest,
        processed_dir=args.processed_dir,
        images_dir=args.images_dir,
        split_strategy=split_strategy,
        seed=args.seed,
        image_size=args.image_size,
    )
    print("Dataset ready:")
    for k, v in paths.items():
        print(f"  {k}: {Path(v)}")


if __name__ == "__main__":
    main()
