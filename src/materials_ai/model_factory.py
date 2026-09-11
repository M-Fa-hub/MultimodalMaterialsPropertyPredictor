"""Factory helpers for constructing multimodal models from config."""

from __future__ import annotations

from typing import Any

from materials_ai.config import AppConfig
from materials_ai.data.descriptors import feature_names
from materials_ai.models.multimodal_model import MultimodalMaterialsModel


def build_multimodal_model(cfg: AppConfig, modalities: dict[str, bool] | None = None) -> MultimodalMaterialsModel:
    mods = modalities or {
        "image": cfg.modalities.image,
        "tabular": cfg.modalities.tabular,
        "text": cfg.modalities.text,
    }
    vision_cfg = {
        "backbone": cfg.model.vision.backbone,
        "pretrained": cfg.model.vision.pretrained,
        "freeze": cfg.model.vision.freeze,
        "embedding_dim": cfg.model.vision.embedding_dim,
    }
    tabular_cfg = {
        "hidden_dims": cfg.model.tabular.hidden_dims,
        "embedding_dim": cfg.model.tabular.embedding_dim,
        "dropout": cfg.model.tabular.dropout,
    }
    text_cfg = {
        "model_name": cfg.model.text.model_name,
        "embedding_dim": cfg.model.text.embedding_dim,
        "freeze": cfg.model.text.freeze,
        "use_transformer": False,
    }
    fusion_cfg = {
        "method": cfg.model.fusion.method,
        "hidden_dim": cfg.model.fusion.hidden_dim,
        "dropout": cfg.model.fusion.dropout,
    }
    head_cfg = {
        "hidden_dims": cfg.model.head.hidden_dims,
        "dropout": cfg.model.head.dropout,
    }
    return MultimodalMaterialsModel(
        tabular_input_dim=len(feature_names()),
        use_image=bool(mods.get("image", False)),
        use_tabular=bool(mods.get("tabular", False)),
        use_text=bool(mods.get("text", False)),
        vision_cfg=vision_cfg,
        tabular_cfg=tabular_cfg,
        text_cfg=text_cfg,
        fusion_cfg=fusion_cfg,
        head_cfg=head_cfg,
    )


def checkpoint_meta_from_cfg(cfg: AppConfig, model: MultimodalMaterialsModel) -> dict[str, Any]:
    return {
        "inputs_used": model.inputs_used,
        "vision_cfg": {
            "backbone": cfg.model.vision.backbone,
            "pretrained": False,  # inference does not need to re-download weights
            "freeze": cfg.model.vision.freeze,
            "embedding_dim": cfg.model.vision.embedding_dim,
        },
        "tabular_cfg": {
            "hidden_dims": cfg.model.tabular.hidden_dims,
            "embedding_dim": cfg.model.tabular.embedding_dim,
            "dropout": cfg.model.tabular.dropout,
        },
        "text_cfg": {
            "model_name": cfg.model.text.model_name,
            "embedding_dim": cfg.model.text.embedding_dim,
            "freeze": cfg.model.text.freeze,
            "use_transformer": False,
        },
        "fusion_cfg": {
            "method": cfg.model.fusion.method,
            "hidden_dim": cfg.model.fusion.hidden_dim,
            "dropout": cfg.model.fusion.dropout,
        },
        "head_cfg": {
            "hidden_dims": cfg.model.head.hidden_dims,
            "dropout": cfg.model.head.dropout,
        },
        "dataset_version": cfg.data.dataset_version,
        "target": cfg.target,
    }
