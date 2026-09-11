"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    processed_dir: str = "data/processed"
    images_dir: str = "data/images"
    split_strategy: str = "composition"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    dataset_version: str = "v1"
    image_size: int = 224
    num_workers: int = 0


class VisionConfig(BaseModel):
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze: bool = True
    embedding_dim: int = 256


class TabularEncoderConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128])
    embedding_dim: int = 128
    dropout: float = 0.2


class TextEncoderConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 128
    freeze: bool = True


class FusionConfig(BaseModel):
    method: str = "concat"
    hidden_dim: int = 256
    dropout: float = 0.3


class HeadConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.2


class ModalitiesConfig(BaseModel):
    image: bool = True
    tabular: bool = True
    text: bool = False


class ModelConfig(BaseModel):
    vision: VisionConfig = Field(default_factory=VisionConfig)
    tabular: TabularEncoderConfig = Field(default_factory=TabularEncoderConfig)
    text: TextEncoderConfig = Field(default_factory=TextEncoderConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)


class TrainingConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 8
    scheduler: str = "cosine"
    resume: str | None = None
    checkpoint_dir: str = "outputs/models"
    checkpoint_name: str = "multimodal_best.pt"


class UncertaintyConfig(BaseModel):
    method: str = "mc_dropout"
    n_samples: int = 20
    dropout_rate: float = 0.2


class OutputConfig(BaseModel):
    metrics_path: str = "outputs/figures/multimodal_metrics.json"
    ablation_path: str = "outputs/figures/ablation_metrics.json"
    models_dir: str = "outputs/models/baselines"
    mlflow_experiment: str = "materials-bandgap"


class AppConfig(BaseModel):
    seed: int = 42
    target: str = "band_gap"
    unit: str = "eV"
    data: DataConfig = Field(default_factory=DataConfig)
    modalities: ModalitiesConfig = Field(default_factory=ModalitiesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    models: dict[str, Any] = Field(default_factory=dict)
    ablation: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config file into a validated AppConfig."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load raw YAML as a dictionary."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
