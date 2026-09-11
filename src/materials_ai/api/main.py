"""FastAPI inference service for Multimodal Materials Property Predictor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from materials_ai import __version__
from materials_ai.logging_utils import setup_logging
from materials_ai.utils import ensure_dir, project_root

logger = setup_logging()

app = FastAPI(
    title="Multimodal Materials Property Predictor",
    description="Band-gap regression from structure images + tabular descriptors",
    version=__version__,
)

_predictor = None


class PredictRequest(BaseModel):
    material_id: str | None = None
    structure_path: str | None = None
    n_uncertainty: int = Field(default=20, ge=1, le=100)


class PredictResponse(BaseModel):
    prediction: float
    unit: str
    uncertainty: float | None
    model_version: str
    inputs_used: list[str]
    material_id: str | None = None
    note: str = (
        "Uncertainty is MC-dropout predictive std, not a calibrated confidence interval."
    )


def get_predictor():
    global _predictor
    if _predictor is None:
        root = project_root()
        ckpt = Path(os.getenv("MODEL_CHECKPOINT", root / "outputs/models/multimodal_best.pt"))
        processed = Path(os.getenv("MATERIALS_AI_DATA_DIR", root / "data")) / "processed"
        if not processed.exists():
            processed = root / "data" / "processed"
        from materials_ai.inference.predictor import MaterialsPredictor

        if not ckpt.exists():
            raise RuntimeError(
                f"Model checkpoint missing at {ckpt}. Train first with scripts/train_multimodal.py"
            )
        _predictor = MaterialsPredictor(checkpoint_path=ckpt, processed_dir=processed)
        logger.info("Loaded predictor from %s", ckpt)
    return _predictor


@app.get("/health")
def health() -> dict[str, Any]:
    ckpt = Path(os.getenv("MODEL_CHECKPOINT", project_root() / "outputs/models/multimodal_best.pt"))
    return {
        "status": "ok",
        "version": __version__,
        "checkpoint_present": ckpt.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not req.material_id and not req.structure_path:
        raise HTTPException(status_code=400, detail="Provide material_id or structure_path")
    try:
        result = predictor.predict(
            material_id=req.material_id,
            structure_path=req.structure_path,
            n_uncertainty=req.n_uncertainty,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        prediction=result.prediction,
        unit=result.unit,
        uncertainty=result.uncertainty,
        model_version=result.model_version,
        inputs_used=result.inputs_used,
        material_id=result.material_id,
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: Annotated[UploadFile, File()],
    n_uncertainty: int = 20,
) -> PredictResponse:
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    suffix = Path(file.filename or "upload.cif").suffix or ".cif"
    tmp_dir = ensure_dir(project_root() / "outputs" / "uploads")
    tmp_path = tmp_dir / f"upload{suffix}"
    content = await file.read()
    tmp_path.write_bytes(content)
    try:
        result = predictor.predict(structure_path=tmp_path, n_uncertainty=n_uncertainty)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(
        prediction=result.prediction,
        unit=result.unit,
        uncertainty=result.uncertainty,
        model_version=result.model_version,
        inputs_used=result.inputs_used,
        material_id=result.material_id,
    )
