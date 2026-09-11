"""MLflow helpers with SQLite tracking (file store is deprecated)."""

from __future__ import annotations

import os

import mlflow

from materials_ai.utils import ensure_dir, project_root


def setup_mlflow(experiment: str, tracking_uri: str | None = None) -> None:
    """Configure MLflow tracking URI and experiment name."""
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        db_path = ensure_dir(project_root() / "outputs" / "mlflow") / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.as_posix()}"
    # Allow legacy file:// URIs if explicitly requested
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
