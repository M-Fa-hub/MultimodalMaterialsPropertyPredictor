"""Automatic 2D structure image generation from CIF / pymatgen structures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from pymatgen.core import Element, Structure

from materials_ai.utils import ensure_dir

# Simple CPK-inspired colors for common elements.
CPK_COLORS: dict[str, str] = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Fe": "#E06633",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Mo": "#54B5B5",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Te": "#D47A00",
    "Al": "#BFA6A6",
    "Ti": "#BFC2C7",
    "W": "#2194D6",
}


def _element_color(symbol: str) -> str:
    return CPK_COLORS.get(symbol, "#78A2CC")


def _element_radius(symbol: str, scale: float = 0.35) -> float:
    el = Element(symbol)
    r = float(el.atomic_radius) if el.atomic_radius is not None else 1.0
    return max(0.25, r * scale)


def render_structure_image(
    structure: Structure,
    out_path: str | Path,
    image_size: int = 224,
    dpi: int = 100,
    max_bond: float = 2.8,
) -> Path:
    """
    Render a standardized 2D projection of a crystal structure.

    - Atoms as colored spheres (scatter)
    - Neighbor bonds drawn when distance < max_bond
    - Fixed square canvas and white background
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Standardize orientation via Niggli reduction + PCA-ish: use lattice matrix projection.
    cart = np.asarray(structure.cart_coords, dtype=float)
    if len(cart) == 0:
        raise ValueError("Empty structure")

    # Center and project to first two principal components for consistent 2D view.
    centered = cart - cart.mean(axis=0, keepdims=True)
    if centered.shape[0] >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:2].T
    else:
        proj = centered[:, :2]

    # Scale to fit nicely in axes.
    span = np.ptp(proj, axis=0)
    span[span < 1e-6] = 1.0
    proj = proj / span.max() * 1.6

    fig_inches = image_size / dpi
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Bonds
    species = [site.specie.symbol for site in structure]
    for i in range(len(cart)):
        for j in range(i + 1, len(cart)):
            dist = float(np.linalg.norm(cart[i] - cart[j]))
            if dist <= max_bond:
                ax.plot(
                    [proj[i, 0], proj[j, 0]],
                    [proj[i, 1], proj[j, 1]],
                    color="#BBBBBB",
                    linewidth=1.0,
                    zorder=1,
                )

    for (x, y), sym in zip(proj, species, strict=True):
        r = _element_radius(sym) * 0.12
        circ = Circle((x, y), radius=r, facecolor=_element_color(sym), edgecolor="#333333", linewidth=0.6, zorder=2)
        ax.add_patch(circ)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_path


def generate_images_for_records(
    records: list[dict],
    images_dir: str | Path,
    image_size: int = 224,
) -> list[dict]:
    """Generate structure images for all records; attach image_path field."""
    images_dir = ensure_dir(images_dir)
    updated: list[dict] = []
    for rec in records:
        structure = Structure.from_file(rec["structure_path"])
        img_path = images_dir / f"{rec['material_id']}.png"
        if not img_path.exists():
            render_structure_image(structure, img_path, image_size=image_size)
        updated.append({**rec, "image_path": str(img_path.as_posix())})
    return updated
