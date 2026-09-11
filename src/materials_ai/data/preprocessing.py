"""Preprocessing, splitting, and model-ready artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from materials_ai.data.descriptors import (
    descriptors_to_vector,
    extract_descriptors,
    feature_names,
    generate_text_description,
)
from materials_ai.data.image_generation import generate_images_for_records
from materials_ai.data.validation import (
    load_manifest,
    save_validation_report,
    validate_records,
)
from materials_ai.logging_utils import get_logger
from materials_ai.utils import ensure_dir, set_seed

logger = get_logger(__name__)

SplitName = Literal["train", "val", "test"]


def _reduced_formula_key(formula: str) -> str:
    """Composition-aware group key using alphabetically sorted element symbols."""
    from materials_ai.data.descriptors import _split_formula_elements

    elems = sorted(set(_split_formula_elements(formula)))
    return "-".join(elems) if elems else formula


def split_dataframe(
    df: pd.DataFrame,
    strategy: str = "composition",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign split labels.

    - random: independent i.i.d. split (may leak near-duplicate chemistries)
    - composition: group by reduced element set so related chemistries stay together
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.copy()
    idx = np.arange(len(df))

    if strategy == "random":
        train_idx, temp_idx = train_test_split(
            idx, test_size=(1 - train_ratio), random_state=seed, shuffle=True
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=relative_test, random_state=seed, shuffle=True
        )
    elif strategy == "composition":
        groups = df["formula"].map(_reduced_formula_key).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
        train_idx, temp_idx = next(gss.split(idx, groups=groups))
        temp_groups = groups[temp_idx]
        relative_test = test_ratio / (val_ratio + test_ratio)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
        val_rel, test_rel = next(gss2.split(temp_idx, groups=temp_groups))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    split = np.array(["train"] * len(df), dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    # train_idx already labeled train
    df["split"] = split
    df["composition_group"] = df["formula"].map(_reduced_formula_key)
    logger.info(
        "Split (%s): train=%d val=%d test=%d",
        strategy,
        int((df["split"] == "train").sum()),
        int((df["split"] == "val").sum()),
        int((df["split"] == "test").sum()),
    )
    return df


def build_feature_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract descriptors, text, and targets for each validated record."""
    rows: list[dict[str, Any]] = []
    names = feature_names()
    for rec in records:
        structure = Structure.from_file(rec["structure_path"])
        feats = extract_descriptors(
            structure,
            crystal_system=rec.get("crystal_system"),
            spacegroup=rec.get("spacegroup"),
        )
        vec = descriptors_to_vector(feats)
        text = generate_text_description(
            formula=str(rec["formula"]),
            crystal_system=rec.get("crystal_system"),
            n_atoms=int(feats["n_atoms"]),
        )
        row: dict[str, Any] = {
            "material_id": rec["material_id"],
            "formula": rec["formula"],
            "band_gap": float(rec["band_gap"]),
            "crystal_system": rec.get("crystal_system"),
            "spacegroup": rec.get("spacegroup"),
            "structure_path": rec["structure_path"],
            "image_path": rec.get("image_path"),
            "text": text,
            "source": rec.get("source", "unknown"),
        }
        for n, v in zip(names, vec, strict=True):
            row[n] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df.loc[df["split"] == "train", feature_cols].to_numpy(dtype=np.float32))
    return scaler


def prepare_dataset(
    raw_manifest: str | Path = "data/raw/manifest.json",
    processed_dir: str | Path = "data/processed",
    images_dir: str | Path = "data/images",
    split_strategy: str = "composition",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    image_size: int = 224,
) -> dict[str, Path]:
    """
    Full pipeline: validate → images → descriptors → split → scale → cache.
    """
    set_seed(seed)
    processed_dir = ensure_dir(processed_dir)
    images_dir = ensure_dir(images_dir)

    records = load_manifest(raw_manifest)
    cleaned, report = validate_records(records)
    save_validation_report(report, processed_dir / "validation_report.json")

    cleaned = generate_images_for_records(cleaned, images_dir, image_size=image_size)
    df = build_feature_table(cleaned)
    df = split_dataframe(
        df,
        strategy=split_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    feat_cols = feature_names()
    scaler = fit_scaler(df, feat_cols)
    scaled = scaler.transform(df[feat_cols].to_numpy(dtype=np.float32))
    for i, col in enumerate(feat_cols):
        df[f"scaled_{col}"] = scaled[:, i]

    parquet_path = processed_dir / "dataset.parquet"
    csv_path = processed_dir / "dataset.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    with (processed_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)

    meta = {
        "n_samples": int(len(df)),
        "split_strategy": split_strategy,
        "seed": seed,
        "target": "band_gap",
        "feature_dim": len(feat_cols),
        "validation": report.to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "notes": [
            "Random split can place chemically near-identical materials in train and test.",
            "Composition split groups by element set to reduce chemical leakage across splits.",
        ],
    }
    meta_path = processed_dir / "dataset_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Prepared dataset with %d samples → %s", len(df), parquet_path)
    return {
        "parquet": parquet_path,
        "csv": csv_path,
        "scaler": processed_dir / "scaler.joblib",
        "meta": meta_path,
        "validation": processed_dir / "validation_report.json",
    }
