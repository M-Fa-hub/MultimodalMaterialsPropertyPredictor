#!/usr/bin/env python3
"""Generate explainability reports for sample predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch

from materials_ai.config import load_config
from materials_ai.data.dataset import MaterialsMultimodalDataset, load_processed_frame
from materials_ai.explainability.tabular import shap_explanation, write_tabular_report
from materials_ai.explainability.vision import save_gradcam_overlay
from materials_ai.logging_utils import setup_logging
from materials_ai.model_factory import build_multimodal_model
from materials_ai.training.checkpointing import load_checkpoint
from materials_ai.utils import ensure_dir


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Explain sample predictions")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/multimodal_best.pt")
    parser.add_argument("--material-id", default=None)
    parser.add_argument("--out-dir", default="outputs/explanations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_processed_frame(cfg.data.processed_dir)
    if args.material_id:
        subset = df.loc[df["material_id"] == args.material_id]
        if subset.empty:
            raise SystemExit(f"material_id not found: {args.material_id}")
    else:
        subset = df.loc[df["split"] == "test"].head(1)
    row = subset.iloc[0]
    ds = MaterialsMultimodalDataset(subset, split=None, image_size=cfg.data.image_size)
    sample = ds[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inputs = payload.get("meta", {}).get("inputs_used", ["image", "tabular"])
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
    model.eval()

    with torch.no_grad():
        pred = model(
            image=sample["image"].unsqueeze(0).to(device) if model.use_image else None,
            tabular=sample["tabular"].unsqueeze(0).to(device) if model.use_tabular else None,
            text=[sample["text"]] if model.use_text else None,
        )
    prediction = float(pred.cpu().item())
    true_value = float(sample["target"].item())

    out_dir = ensure_dir(args.out_dir)
    mid = str(row["material_id"])

    # Tabular explanation via RF importance / SHAP if baseline exists
    rf_path = Path("outputs/models/baselines/random_forest.joblib")
    factors = []
    if rf_path.exists():
        rf = joblib.load(rf_path)
        factors = shap_explanation(rf, sample["tabular"].numpy())
    report_path = write_tabular_report(
        out_dir / f"{mid}_report.txt",
        prediction=prediction,
        true_value=true_value,
        factors=factors,
        unit=cfg.unit,
    )

    cam_path = None
    if model.use_image:
        cam_path = save_gradcam_overlay(
            model,
            sample["image"].to(device),
            original_image_path=row["image_path"],
            out_path=out_dir / f"{mid}_gradcam.png",
            tabular=sample["tabular"] if model.use_tabular else None,
            text=[sample["text"]] if model.use_text else None,
        )

    print(f"Wrote report: {report_path}")
    if cam_path:
        print(f"Wrote Grad-CAM: {cam_path}")


if __name__ == "__main__":
    main()
