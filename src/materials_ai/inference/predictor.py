"""Inference + MC-dropout uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from pymatgen.core import Structure
from torchvision import transforms

from materials_ai.data.descriptors import (
    descriptors_to_vector,
    extract_descriptors,
    feature_names,
    generate_text_description,
)
from materials_ai.data.image_generation import render_structure_image
from materials_ai.logging_utils import get_logger
from materials_ai.models.multimodal_model import MultimodalMaterialsModel
from materials_ai.utils import ensure_dir, project_root

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    prediction: float
    unit: str
    uncertainty: float | None
    model_version: str
    inputs_used: list[str]
    material_id: str | None = None
    explanation: dict[str, Any] | None = None


def build_model_from_meta(
    tabular_input_dim: int,
    meta: dict[str, Any],
) -> MultimodalMaterialsModel:
    """Reconstruct model using checkpoint meta / defaults."""
    inputs = meta.get("inputs_used", ["image", "tabular"])
    return MultimodalMaterialsModel(
        tabular_input_dim=tabular_input_dim,
        use_image="image" in inputs,
        use_tabular="tabular" in inputs,
        use_text="text" in inputs,
        vision_cfg=meta.get(
            "vision_cfg",
            {"backbone": "resnet18", "pretrained": False, "freeze": True, "embedding_dim": 256},
        ),
        tabular_cfg=meta.get(
            "tabular_cfg",
            {"hidden_dims": [256, 128], "embedding_dim": 128, "dropout": 0.2},
        ),
        text_cfg=meta.get(
            "text_cfg",
            {"embedding_dim": 128, "use_transformer": False},
        ),
        fusion_cfg=meta.get("fusion_cfg", {"method": "concat", "hidden_dim": 256, "dropout": 0.3}),
        head_cfg=meta.get("head_cfg", {"hidden_dims": [128, 64], "dropout": 0.2}),
    )


class MaterialsPredictor:
    """End-to-end predictor for API / Streamlit."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        processed_dir: str | Path = "data/processed",
        model_version: str = "v1",
        unit: str = "eV",
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.processed_dir = Path(processed_dir)
        self.model_version = model_version
        self.unit = unit
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_cols = feature_names()
        scaler_path = self.processed_dir / "scaler.joblib"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = self._load_model()

    def _load_model(self) -> MultimodalMaterialsModel:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        # Peek meta without full load of weights into wrong architecture
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        model = build_model_from_meta(len(self.feature_cols), meta)
        model.load_state_dict(payload["model_state"])
        model.to(self.device)
        model.eval()
        return model

    def _scale(self, vec: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return vec.astype(np.float32)
        return self.scaler.transform(vec.reshape(1, -1)).astype(np.float32).ravel()

    def featurize_structure(
        self,
        structure: Structure,
        material_id: str = "runtime",
        crystal_system: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        feats = extract_descriptors(structure, crystal_system=crystal_system)
        vec = descriptors_to_vector(feats)
        scaled = self._scale(vec)
        text = generate_text_description(
            formula=structure.composition.reduced_formula,
            crystal_system=crystal_system,
            n_atoms=int(feats["n_atoms"]),
        )
        cache_dir = ensure_dir(cache_dir or (project_root() / "outputs" / "tmp_images"))
        image_path = Path(cache_dir) / f"{material_id}.png"
        render_structure_image(structure, image_path)
        image = Image.open(image_path).convert("RGB")
        image_t = self.transform(image)
        return {
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "tabular": torch.from_numpy(scaled),
            "image": image_t,
            "text": text,
            "image_path": str(image_path),
            "descriptors": feats,
        }

    def featurize_from_material_id(self, material_id: str) -> dict[str, Any]:

        from materials_ai.data.dataset import load_processed_frame

        df = load_processed_frame(self.processed_dir)
        rows = df.loc[df["material_id"] == material_id]
        if rows.empty:
            raise KeyError(f"Unknown material_id: {material_id}")
        row = rows.iloc[0]
        structure = Structure.from_file(row["structure_path"])
        return self.featurize_structure(
            structure,
            material_id=material_id,
            crystal_system=row.get("crystal_system"),
        )

    @torch.no_grad()
    def _forward_once(self, bundle: dict[str, Any]) -> float:
        image = bundle["image"].unsqueeze(0).to(self.device) if self.model.use_image else None
        tabular = bundle["tabular"].unsqueeze(0).to(self.device) if self.model.use_tabular else None
        text = [bundle["text"]] if self.model.use_text else None
        pred = self.model(image=image, tabular=tabular, text=text)
        return float(pred.detach().cpu().item())

    def predict_mc_dropout(
        self,
        bundle: dict[str, Any],
        n_samples: int = 20,
    ) -> tuple[float, float]:
        """
        Monte Carlo dropout uncertainty.

        Note: returned ±uncertainty is the predictive std under MC dropout,
        not a calibrated scientific confidence interval.
        """
        self.model.train()  # enable dropout
        preds: list[float] = []
        for _ in range(n_samples):
            preds.append(self._forward_once(bundle))
        self.model.eval()
        arr = np.asarray(preds, dtype=float)
        return float(arr.mean()), float(arr.std())

    def predict(
        self,
        *,
        material_id: str | None = None,
        structure_path: str | Path | None = None,
        n_uncertainty: int = 20,
    ) -> PredictionResult:
        if material_id:
            bundle = self.featurize_from_material_id(material_id)
        elif structure_path:
            structure = Structure.from_file(str(structure_path))
            bundle = self.featurize_structure(structure, material_id=Path(structure_path).stem)
        else:
            raise ValueError("Provide material_id or structure_path")

        mean, std = self.predict_mc_dropout(bundle, n_samples=n_uncertainty)
        return PredictionResult(
            prediction=mean,
            unit=self.unit,
            uncertainty=std,
            model_version=self.model_version,
            inputs_used=self.model.inputs_used,
            material_id=bundle.get("material_id"),
            explanation={"descriptors": bundle.get("descriptors"), "image_path": bundle.get("image_path")},
        )
