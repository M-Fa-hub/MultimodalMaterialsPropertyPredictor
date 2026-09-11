"""Scientific descriptor extraction without target leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import Element, Structure

# Common elements for composition fraction vectors (deterministic order).
ELEMENT_VOCAB = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Ba", "W", "Au", "Pb", "Bi",
]

CRYSTAL_SYSTEMS = [
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
    "trigonal",
    "monoclinic",
    "triclinic",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _element_stats(structure: Structure) -> dict[str, float]:
    zs: list[float] = []
    ens: list[float] = []
    radii: list[float] = []
    for site in structure:
        el = Element(site.specie.symbol)
        zs.append(float(el.Z))
        ens.append(float(el.X) if el.X is not None else 0.0)
        radii.append(float(el.atomic_radius) if el.atomic_radius is not None else 0.0)
    arr_z = np.asarray(zs, dtype=float)
    arr_en = np.asarray(ens, dtype=float)
    arr_r = np.asarray(radii, dtype=float)
    return {
        "n_atoms": float(len(structure)),
        "n_elements": float(len({s.specie.symbol for s in structure})),
        "atomic_number_mean": float(arr_z.mean()),
        "atomic_number_std": float(arr_z.std()) if len(arr_z) > 1 else 0.0,
        "atomic_number_max": float(arr_z.max()),
        "electronegativity_mean": float(arr_en.mean()),
        "electronegativity_std": float(arr_en.std()) if len(arr_en) > 1 else 0.0,
        "atomic_radius_mean": float(arr_r.mean()),
        "atomic_radius_std": float(arr_r.std()) if len(arr_r) > 1 else 0.0,
    }


def _composition_fractions(structure: Structure) -> dict[str, float]:
    comp = structure.composition.fractional_composition
    total = float(sum(comp[el] for el in comp))
    fracs = {f"frac_{sym}": 0.0 for sym in ELEMENT_VOCAB}
    if total <= 0:
        return fracs
    for el, amt in comp.items():
        key = f"frac_{el.symbol}"
        if key in fracs:
            fracs[key] = float(amt) / total
    return fracs


def _crystal_system_one_hot(crystal_system: str | None) -> dict[str, float]:
    cs = (crystal_system or "unknown").lower()
    return {f"cs_{name}": 1.0 if cs == name else 0.0 for name in CRYSTAL_SYSTEMS}


def extract_descriptors(
    structure: Structure,
    crystal_system: str | None = None,
    spacegroup: str | None = None,
) -> dict[str, float]:
    """
    Extract tabular descriptors from structure + symmetry metadata.

    Explicitly excluded (target leakage):
    - band_gap, formation_energy, energy_above_hull, and related targets
    """
    lattice = structure.lattice
    feats: dict[str, float] = {}
    feats.update(_element_stats(structure))
    feats.update(
        {
            "volume": float(structure.volume),
            "density": float(structure.density),
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
        }
    )
    feats.update(_composition_fractions(structure))
    feats.update(_crystal_system_one_hot(crystal_system))
    # Space group number if parseable from symbol is skipped; use hashed length proxy avoided.
    # Keep a simple numeric encoding of spacegroup string length is not scientific —
    # instead store nothing from target fields.
    _ = spacegroup  # retained for API symmetry; not used as a leaky dense ID
    return feats


def feature_names() -> list[str]:
    """Deterministic ordered feature names matching extract_descriptors output."""
    names = [
        "n_atoms",
        "n_elements",
        "atomic_number_mean",
        "atomic_number_std",
        "atomic_number_max",
        "electronegativity_mean",
        "electronegativity_std",
        "atomic_radius_mean",
        "atomic_radius_std",
        "volume",
        "density",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
    ]
    names.extend([f"frac_{sym}" for sym in ELEMENT_VOCAB])
    names.extend([f"cs_{name}" for name in CRYSTAL_SYSTEMS])
    return names


def descriptors_to_vector(feats: dict[str, float]) -> np.ndarray:
    """Convert descriptor dict to a fixed-order float32 vector."""
    names = feature_names()
    return np.asarray([_safe_float(feats.get(n, 0.0)) for n in names], dtype=np.float32)


def generate_text_description(
    formula: str,
    crystal_system: str | None,
    n_atoms: int,
) -> str:
    """Deterministic short text metadata derived only from input features."""
    elements = " and ".join(
        sorted({tok for tok in _split_formula_elements(formula)})
    ) or formula
    cs = crystal_system or "unknown"
    return (
        f"Crystal material containing {elements} with {cs} symmetry "
        f"and {n_atoms} atoms in the unit cell."
    )


def _split_formula_elements(formula: str) -> list[str]:
    import re

    return re.findall(r"[A-Z][a-z]?", formula)
