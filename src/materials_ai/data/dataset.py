"""PyTorch dataset and dataloaders for multimodal materials samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from materials_ai.data.descriptors import feature_names


class MaterialsMultimodalDataset(Dataset):
    """Dataset yielding image tensor, tabular vector, text, and target."""

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str | None = None,
        image_size: int = 224,
        use_scaled_features: bool = True,
    ) -> None:
        df = frame if split is None else frame.loc[frame["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No samples for split={split}")
        self.df = df
        self.feat_cols = feature_names()
        self.use_scaled = use_scaled_features
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _tabular(self, row: pd.Series) -> np.ndarray:
        if self.use_scaled:
            cols = [f"scaled_{c}" for c in self.feat_cols]
            if all(c in self.df.columns for c in cols):
                return row[cols].to_numpy(dtype=np.float32)
        return row[self.feat_cols].to_numpy(dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image_t = self.transform(image)
        tabular = torch.from_numpy(self._tabular(row))
        target = torch.tensor(float(row["band_gap"]), dtype=torch.float32)
        return {
            "material_id": str(row["material_id"]),
            "image": image_t,
            "tabular": tabular,
            "text": str(row.get("text", "")),
            "target": target,
            "formula": str(row["formula"]),
        }


def load_processed_frame(processed_dir: str | Path) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    parquet = processed_dir / "dataset.parquet"
    csv_path = processed_dir / "dataset.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No processed dataset in {processed_dir}")


def create_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders from processed artifacts."""
    df = load_processed_frame(processed_dir)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = MaterialsMultimodalDataset(df, split=split, image_size=image_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
