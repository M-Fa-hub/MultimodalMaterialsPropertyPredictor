"""End-to-end smoke: demo data → preprocess → short train → predict."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from materials_ai.config import AppConfig
from materials_ai.data.dataset import create_dataloaders, load_processed_frame
from materials_ai.data.download import acquire_dataset
from materials_ai.data.preprocessing import prepare_dataset
from materials_ai.inference.predictor import MaterialsPredictor
from materials_ai.model_factory import build_multimodal_model, checkpoint_meta_from_cfg
from materials_ai.training.checkpointing import save_checkpoint
from materials_ai.training.trainer import train_multimodal


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    images = tmp_path / "images"
    models = tmp_path / "models"

    manifest = acquire_dataset(raw_dir=raw, source="demo", n_samples=36, seed=42)
    prepare_dataset(
        raw_manifest=manifest,
        processed_dir=processed,
        images_dir=images,
        split_strategy="random",  # tiny demo: composition groups can empty a split
        seed=42,
        image_size=64,
    )
    df = load_processed_frame(processed)
    assert len(df) >= 30
    assert set(df["split"]) >= {"train", "val", "test"}

    cfg = AppConfig()
    cfg.data.processed_dir = str(processed)
    cfg.data.images_dir = str(images)
    cfg.data.image_size = 64
    cfg.training.epochs = 1
    cfg.training.batch_size = 4
    cfg.training.mixed_precision = False
    cfg.training.checkpoint_dir = str(models)
    cfg.training.checkpoint_name = "smoke.pt"
    cfg.training.early_stopping_patience = 2
    cfg.model.vision.pretrained = False
    cfg.model.vision.embedding_dim = 32
    cfg.model.tabular.hidden_dims = [32]
    cfg.model.tabular.embedding_dim = 16
    cfg.model.fusion.hidden_dim = 32
    cfg.model.head.hidden_dims = [16]
    cfg.modalities.text = False

    loaders = create_dataloaders(
        processed_dir=processed,
        batch_size=4,
        image_size=64,
        num_workers=0,
    )
    model = build_multimodal_model(cfg)
    result = train_multimodal(
        model,
        loaders,
        epochs=1,
        lr=1e-3,
        mixed_precision=False,
        early_stopping_patience=2,
        checkpoint_dir=models,
        checkpoint_name="smoke.pt",
        seed=42,
        mlflow_experiment="smoke",
        run_name="smoke",
        extra_params={"dataset_version": "smoke"},
    )
    assert result.checkpoint_path.exists()
    save_checkpoint(
        result.checkpoint_path,
        model,
        best_metric=result.best_val_mae,
        meta=checkpoint_meta_from_cfg(cfg, model),
    )

    predictor = MaterialsPredictor(
        checkpoint_path=result.checkpoint_path,
        processed_dir=processed,
        model_version="smoke",
    )
    mid = str(df.iloc[0]["material_id"])
    pred = predictor.predict(material_id=mid, n_uncertainty=3)
    assert pred.prediction == pytest.approx(pred.prediction)  # finite
    assert pred.uncertainty is not None
    assert "tabular" in pred.inputs_used or "image" in pred.inputs_used
    assert torch.isfinite(torch.tensor(pred.prediction))
