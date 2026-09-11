"""Unit tests for descriptors, validation, metrics, and model forward pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from materials_ai.data.descriptors import (
    descriptors_to_vector,
    extract_descriptors,
    feature_names,
    generate_text_description,
)
from materials_ai.data.download import build_demo_dataset, structure_from_dict
from materials_ai.data.image_generation import render_structure_image
from materials_ai.data.validation import validate_records
from materials_ai.models.multimodal_model import MultimodalMaterialsModel
from materials_ai.training.evaluation import regression_metrics


@pytest.fixture
def simple_structure() -> Structure:
    lattice = Lattice.cubic(4.0)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_feature_names_stable() -> None:
    names = feature_names()
    assert len(names) == len(set(names))
    assert "electronegativity_mean" in names
    assert "band_gap" not in names  # no target leakage


def test_extract_descriptors(simple_structure: Structure) -> None:
    feats = extract_descriptors(simple_structure, crystal_system="cubic")
    vec = descriptors_to_vector(feats)
    assert vec.shape == (len(feature_names()),)
    assert feats["n_atoms"] == 2
    assert feats["cs_cubic"] == 1.0
    assert "band_gap" not in feats


def test_text_description_deterministic() -> None:
    a = generate_text_description("Si", "cubic", 2)
    b = generate_text_description("Si", "cubic", 2)
    assert a == b
    assert "cubic" in a
    assert "2 atoms" in a


def test_validation_removes_bad_rows(tmp_path: Path, simple_structure: Structure) -> None:
    cif = tmp_path / "ok.cif"
    simple_structure.to(filename=str(cif), fmt="cif")
    records = [
        {
            "material_id": "a",
            "formula": "Si",
            "band_gap": 1.1,
            "structure_path": str(cif),
        },
        {
            "material_id": "b",
            "formula": "Si",
            "band_gap": None,
            "structure_path": str(cif),
        },
        {
            "material_id": "a",
            "formula": "Si",
            "band_gap": 1.2,
            "structure_path": str(cif),
        },
    ]
    cleaned, report = validate_records(records)
    assert len(cleaned) == 1
    assert report.removals.get("missing_target", 0) == 1
    assert report.removals.get("duplicate_material_id", 0) == 1


def test_render_structure_image(tmp_path: Path, simple_structure: Structure) -> None:
    out = tmp_path / "si.png"
    render_structure_image(simple_structure, out, image_size=128)
    assert out.exists()
    assert out.stat().st_size > 0


def test_regression_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.1, 1.9, 3.2])
    m = regression_metrics(y, pred)
    assert set(m) == {"mae", "rmse", "r2"}
    assert m["mae"] > 0


def test_multimodal_forward() -> None:
    model = MultimodalMaterialsModel(
        tabular_input_dim=16,
        use_image=True,
        use_tabular=True,
        use_text=False,
        vision_cfg={"backbone": "resnet18", "pretrained": False, "freeze": True, "embedding_dim": 32},
        tabular_cfg={"hidden_dims": [32], "embedding_dim": 16, "dropout": 0.1},
        fusion_cfg={"method": "concat", "hidden_dim": 32, "dropout": 0.1},
        head_cfg={"hidden_dims": [16], "dropout": 0.1},
    )
    image = torch.randn(2, 3, 64, 64)
    tabular = torch.randn(2, 16)
    out = model(image=image, tabular=tabular)
    assert out.shape == (2,)


def test_demo_dataset_smoke(tmp_path: Path) -> None:
    manifest = build_demo_dataset(tmp_path / "raw", n_samples=16, seed=0)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_samples"] == 16
    assert Path(payload["records"][0]["structure_path"]).exists()
    structure = structure_from_dict(
        {
            "lattice": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "species": ["C"],
            "coords": [[0, 0, 0]],
            "coords_are_cartesian": False,
        }
    )
    assert structure.num_sites == 1
