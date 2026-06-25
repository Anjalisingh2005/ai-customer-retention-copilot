"""MLflow tracking + model-registry helpers.

We log every model trained in scripts/train_model.py with its params, metrics,
and a serialized artifact. The best model is registered as `churn-prod` and
transitioned to the `Production` stage so the API can `mlflow.pyfunc.load_model`
from the registry in environments where MLflow is the source of truth.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def _client():
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


@contextmanager
def run(name: str, tags: Dict[str, str] | None = None):
    mlflow = _client()
    with mlflow.start_run(run_name=name, tags=tags or {}) as r:
        yield r


def log_params(params: Dict[str, Any]) -> None:
    mlflow = _client()
    mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: Dict[str, float]) -> None:
    mlflow = _client()
    mlflow.log_metrics(metrics)


def log_sklearn(model, artifact_path: str = "model", input_example=None) -> str:
    """Log a sklearn-style estimator (XGB/LightGBM use sklearn wrappers)."""
    mlflow = _client()
    info = mlflow.sklearn.log_model(model, artifact_path, input_example=input_example)
    return info.model_uri


def register_and_promote(model_uri: str, name: str | None = None) -> None:
    """Register the model and move it to Production stage."""
    mlflow = _client()
    name = name or settings.mlflow_registered_model
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=name, version=mv.version, stage="Production", archive_existing_versions=True
        )
        log.info("Registered %s v%s → Production", name, mv.version)
    except Exception as e:
        log.warning("Model registry transition skipped: %s", e)


def log_artifacts(paths: Iterable[str], artifact_path: str | None = None) -> None:
    mlflow = _client()
    for p in paths:
        mlflow.log_artifact(p, artifact_path=artifact_path)
