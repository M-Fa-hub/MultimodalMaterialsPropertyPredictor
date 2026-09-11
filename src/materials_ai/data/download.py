"""Dataset acquisition from Materials Project or a reproducible demo corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure

from materials_ai.logging_utils import get_logger
from materials_ai.utils import ensure_dir

logger = get_logger(__name__)

# Lightweight demo materials with realistic-ish band gaps for offline / CI use.
# Structures are small unit cells; targets are illustrative, not claims of discovery.
DEMO_MATERIALS: list[dict[str, Any]] = [
    {
        "material_id": "demo-Si",
        "formula": "Si",
        "band_gap": 1.12,
        "crystal_system": "cubic",
        "spacegroup": "Fd-3m",
        "structure": {
            "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            "species": ["Si", "Si"],
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-GaAs",
        "formula": "GaAs",
        "band_gap": 1.42,
        "crystal_system": "cubic",
        "spacegroup": "F-43m",
        "structure": {
            "lattice": [[5.65, 0, 0], [0, 5.65, 0], [0, 0, 5.65]],
            "species": ["Ga", "As"],
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-GaN",
        "formula": "GaN",
        "band_gap": 3.40,
        "crystal_system": "hexagonal",
        "spacegroup": "P63mc",
        "structure": {
            "lattice": [[3.19, 0, 0], [-1.595, 2.762, 0], [0, 0, 5.19]],
            "species": ["Ga", "N", "Ga", "N"],
            "coords": [
                [0.333, 0.667, 0.0],
                [0.333, 0.667, 0.375],
                [0.667, 0.333, 0.5],
                [0.667, 0.333, 0.875],
            ],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-ZnO",
        "formula": "ZnO",
        "band_gap": 3.37,
        "crystal_system": "hexagonal",
        "spacegroup": "P63mc",
        "structure": {
            "lattice": [[3.25, 0, 0], [-1.625, 2.815, 0], [0, 0, 5.21]],
            "species": ["Zn", "O", "Zn", "O"],
            "coords": [
                [0.333, 0.667, 0.0],
                [0.333, 0.667, 0.38],
                [0.667, 0.333, 0.5],
                [0.667, 0.333, 0.88],
            ],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-TiO2",
        "formula": "TiO2",
        "band_gap": 3.20,
        "crystal_system": "tetragonal",
        "spacegroup": "P42/mnm",
        "structure": {
            "lattice": [[4.59, 0, 0], [0, 4.59, 0], [0, 0, 2.96]],
            "species": ["Ti", "Ti", "O", "O", "O", "O"],
            "coords": [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.3, 0.3, 0],
                [0.7, 0.7, 0],
                [0.2, 0.8, 0.5],
                [0.8, 0.2, 0.5],
            ],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-MoS2",
        "formula": "MoS2",
        "band_gap": 1.80,
        "crystal_system": "hexagonal",
        "spacegroup": "P63/mmc",
        "structure": {
            "lattice": [[3.16, 0, 0], [-1.58, 2.736, 0], [0, 0, 12.3]],
            "species": ["Mo", "S", "S"],
            "coords": [[0.333, 0.667, 0.25], [0.333, 0.667, 0.12], [0.333, 0.667, 0.38]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-CdTe",
        "formula": "CdTe",
        "band_gap": 1.50,
        "crystal_system": "cubic",
        "spacegroup": "F-43m",
        "structure": {
            "lattice": [[6.48, 0, 0], [0, 6.48, 0], [0, 0, 6.48]],
            "species": ["Cd", "Te"],
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-InP",
        "formula": "InP",
        "band_gap": 1.34,
        "crystal_system": "cubic",
        "spacegroup": "F-43m",
        "structure": {
            "lattice": [[5.87, 0, 0], [0, 5.87, 0], [0, 0, 5.87]],
            "species": ["In", "P"],
            "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-AlN",
        "formula": "AlN",
        "band_gap": 6.20,
        "crystal_system": "hexagonal",
        "spacegroup": "P63mc",
        "structure": {
            "lattice": [[3.11, 0, 0], [-1.555, 2.693, 0], [0, 0, 4.98]],
            "species": ["Al", "N", "Al", "N"],
            "coords": [
                [0.333, 0.667, 0.0],
                [0.333, 0.667, 0.38],
                [0.667, 0.333, 0.5],
                [0.667, 0.333, 0.88],
            ],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-Cu2O",
        "formula": "Cu2O",
        "band_gap": 2.10,
        "crystal_system": "cubic",
        "spacegroup": "Pn-3m",
        "structure": {
            "lattice": [[4.27, 0, 0], [0, 4.27, 0], [0, 0, 4.27]],
            "species": ["Cu", "Cu", "O", "O"],
            "coords": [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-Fe2O3",
        "formula": "Fe2O3",
        "band_gap": 2.20,
        "crystal_system": "trigonal",
        "spacegroup": "R-3c",
        "structure": {
            "lattice": [[5.04, 0, 0], [-2.52, 4.365, 0], [0, 0, 13.75]],
            "species": ["Fe", "Fe", "O", "O", "O"],
            "coords": [
                [0, 0, 0.355],
                [0, 0, 0.145],
                [0.306, 0, 0.25],
                [0, 0.306, 0.25],
                [0.694, 0.694, 0.25],
            ],
            "coords_are_cartesian": False,
        },
    },
    {
        "material_id": "demo-WO3",
        "formula": "WO3",
        "band_gap": 2.60,
        "crystal_system": "monoclinic",
        "spacegroup": "P21/n",
        "structure": {
            "lattice": [[7.3, 0, 0], [0, 7.53, 0], [-1.2, 0, 7.69]],
            "species": ["W", "W", "O", "O", "O", "O", "O", "O"],
            "coords": [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.0, 0.25, 0.25],
                [0.5, 0.25, 0.25],
                [0.25, 0.0, 0.25],
                [0.25, 0.5, 0.25],
                [0.25, 0.25, 0.0],
                [0.25, 0.25, 0.5],
            ],
            "coords_are_cartesian": False,
        },
    },
]


def structure_from_dict(payload: dict[str, Any]) -> Structure:
    """Build a pymatgen Structure from a compact dict."""
    lattice = Lattice(payload["lattice"])
    return Structure(
        lattice,
        payload["species"],
        payload["coords"],
        coords_are_cartesian=payload.get("coords_are_cartesian", False),
    )


def _augment_demo(base: list[dict[str, Any]], n_extra: int, seed: int) -> list[dict[str, Any]]:
    """Create additional samples by small lattice/gap perturbations of base materials."""
    rng = np.random.default_rng(seed)
    out = list(base)
    for i in range(n_extra):
        src = base[i % len(base)]
        scale = float(1.0 + rng.normal(0, 0.02))
        gap_noise = float(rng.normal(0, 0.08))
        struct = src["structure"]
        lattice = (np.array(struct["lattice"], dtype=float) * scale).tolist()
        band_gap = max(0.05, float(src["band_gap"]) + gap_noise)
        out.append(
            {
                **src,
                "material_id": f"{src['material_id']}-aug{i:03d}",
                "band_gap": round(band_gap, 4),
                "structure": {**struct, "lattice": lattice},
                "is_augmented": True,
            }
        )
    return out


def build_demo_dataset(
    raw_dir: str | Path,
    n_samples: int = 120,
    seed: int = 42,
) -> Path:
    """
    Write a reproducible demo Materials-like dataset for offline development.

    Notes
    -----
    Demo targets are curated / lightly perturbed literature-like values for
    engineering the pipeline. They are not a substitute for Materials Project
    DFT data when claiming scientific performance.
    """
    raw_path = ensure_dir(raw_dir)
    n_extra = max(0, n_samples - len(DEMO_MATERIALS))
    records = _augment_demo(DEMO_MATERIALS, n_extra=n_extra, seed=seed)

    structures_dir = ensure_dir(raw_path / "structures")
    rows: list[dict[str, Any]] = []
    for rec in records:
        structure = structure_from_dict(rec["structure"])
        cif_path = structures_dir / f"{rec['material_id']}.cif"
        structure.to(filename=str(cif_path), fmt="cif")
        rows.append(
            {
                "material_id": rec["material_id"],
                "formula": rec["formula"],
                "band_gap": rec["band_gap"],
                "crystal_system": rec["crystal_system"],
                "spacegroup": rec["spacegroup"],
                "structure_path": str(cif_path.as_posix()),
                "is_augmented": rec.get("is_augmented", False),
                "source": "demo",
            }
        )

    manifest_path = raw_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"source": "demo", "n_samples": len(rows), "records": rows}, f, indent=2)
    logger.info("Wrote demo dataset with %d samples to %s", len(rows), manifest_path)
    return manifest_path


def download_materials_project(
    raw_dir: str | Path,
    n_samples: int = 500,
    api_key: str | None = None,
) -> Path:
    """
    Download a filtered band-gap subset from Materials Project (requires MP_API_KEY).

    Filters:
    - band_gap is not None and >= 0
    - excludes metals (band_gap == 0) optionally kept if > 0 only
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError(
            "mp-api is required for Materials Project download. "
            "Install with: pip install 'materials-ai[mp]'"
        ) from exc

    key = api_key or os.getenv("MP_API_KEY")
    if not key:
        raise OSError(
            "MP_API_KEY not set. Use --source demo or set MP_API_KEY in .env"
        )

    raw_path = ensure_dir(raw_dir)
    structures_dir = ensure_dir(raw_path / "structures")
    rows: list[dict[str, Any]] = []

    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(0.1, 8.0),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "symmetry",
                "structure",
                "volume",
                "density",
                "nsites",
            ],
            num_chunks=max(1, n_samples // 100 + 1),
            chunk_size=min(100, n_samples),
        )
        for doc in docs:
            if len(rows) >= n_samples:
                break
            if doc.band_gap is None or doc.structure is None:
                continue
            mid = str(doc.material_id)
            cif_path = structures_dir / f"{mid}.cif"
            doc.structure.to(filename=str(cif_path), fmt="cif")
            sym = doc.symmetry
            rows.append(
                {
                    "material_id": mid,
                    "formula": doc.formula_pretty,
                    "band_gap": float(doc.band_gap),
                    "crystal_system": getattr(sym, "crystal_system", None),
                    "spacegroup": getattr(sym, "symbol", None),
                    "structure_path": str(cif_path.as_posix()),
                    "is_augmented": False,
                    "source": "materials_project",
                    "volume": float(doc.volume) if doc.volume is not None else None,
                    "density": float(doc.density) if doc.density is not None else None,
                    "nsites": int(doc.nsites) if doc.nsites is not None else None,
                }
            )

    manifest_path = raw_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"source": "materials_project", "n_samples": len(rows), "records": rows},
            f,
            indent=2,
        )
    logger.info("Downloaded %d Materials Project samples to %s", len(rows), manifest_path)
    return manifest_path


def acquire_dataset(
    raw_dir: str | Path = "data/raw",
    source: str = "demo",
    n_samples: int = 120,
    seed: int = 42,
    api_key: str | None = None,
) -> Path:
    """Acquire raw dataset from the requested source."""
    source = source.lower()
    if source == "demo":
        return build_demo_dataset(raw_dir, n_samples=n_samples, seed=seed)
    if source in {"mp", "materials_project"}:
        return download_materials_project(raw_dir, n_samples=n_samples, api_key=api_key)
    raise ValueError(f"Unknown source: {source}. Use 'demo' or 'mp'.")
