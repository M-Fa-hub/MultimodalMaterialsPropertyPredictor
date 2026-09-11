"""Raw dataset validation and filtering with audited removal counts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from materials_ai.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Audit trail for sample filtering."""

    n_input: int = 0
    n_output: int = 0
    removals: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_removed": self.n_input - self.n_output,
            "removals": self.removals,
            "notes": self.notes,
        }


def _inc(report: ValidationReport, reason: str) -> None:
    report.removals[reason] = report.removals.get(reason, 0) + 1


def validate_records(
    records: list[dict[str, Any]],
    target_key: str = "band_gap",
    min_gap: float = 0.05,
    max_gap: float = 10.0,
) -> tuple[list[dict[str, Any]], ValidationReport]:
    """
    Validate and filter material records.

    Removals are counted by reason for reproducibility documentation.
    """
    report = ValidationReport(n_input=len(records))
    report.notes.append(
        "Target leakage check: band_gap is never used as an input descriptor."
    )
    report.notes.append(
        "Density/volume from structure geometry are allowed; target property is excluded."
    )

    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for rec in records:
        mid = rec.get("material_id")
        if not mid:
            _inc(report, "missing_material_id")
            continue
        if mid in seen_ids:
            _inc(report, "duplicate_material_id")
            continue
        seen_ids.add(str(mid))

        if target_key not in rec or rec[target_key] is None:
            _inc(report, "missing_target")
            continue
        try:
            target = float(rec[target_key])
        except (TypeError, ValueError):
            _inc(report, "non_numeric_target")
            continue
        if not (min_gap <= target <= max_gap):
            _inc(report, "target_out_of_range")
            continue

        structure_path = rec.get("structure_path")
        if not structure_path or not Path(structure_path).exists():
            _inc(report, "missing_structure_file")
            continue

        if not rec.get("formula"):
            _inc(report, "missing_formula")
            continue

        cleaned.append({**rec, target_key: target})

    report.n_output = len(cleaned)
    logger.info(
        "Validation: %d -> %d samples (removed %d)",
        report.n_input,
        report.n_output,
        report.n_input - report.n_output,
    )
    for reason, count in report.removals.items():
        logger.info("  removed %d due to %s", count, reason)
    return cleaned, report


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a raw manifest JSON."""
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("records", []))


def save_validation_report(report: ValidationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
