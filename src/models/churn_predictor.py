"""Train Logistic Regression, Random Forest, XGBoost, LightGBM. Pick best by ROC-AUC.

The chosen winner is what the API loads. Everything is also logged to MLflow
(see src/mlops/tracking.py) so you can see metrics & params per model in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ModelResult:
    name: str
    model: Any
    metrics: Dict[str, float]
    params: Dict[str, Any] = field(default_factory=dict)


def _evaluate(name: str, model, X_test, y_test) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    log.info(
        "%-20s | ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f P=%.4f R=%.4f",
        name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics


def train_logistic(X_train, y_train, X_test, y_test) -> ModelResult:
    params = dict(max_iter=1000, C=1.0, n_jobs=-1, random_state=settings.random_seed)
    model = LogisticRegression(**params).fit(X_train, y_train)
    return ModelResult("logistic_regression", model, _evaluate("logistic_regression", model, X_test, y_test), params)


def train_random_forest(X_train, y_train, X_test, y_test) -> ModelResult:
    params = dict(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        class_weight="balanced",
        random_state=settings.random_seed,
    )
    model = RandomForestClassifier(**params).fit(X_train, y_train)
    return ModelResult("random_forest", model, _evaluate("random_forest", model, X_test, y_test), params)


def train_xgboost(X_train, y_train, X_test, y_test) -> ModelResult:
    from xgboost import XGBClassifier

    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    params = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=settings.random_seed,
    )
    model = XGBClassifier(**params).fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return ModelResult("xgboost", model, _evaluate("xgboost", model, X_test, y_test), params)


def train_lightgbm(X_train, y_train, X_test, y_test) -> ModelResult:
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=700,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=settings.random_seed,
    )
    model = LGBMClassifier(**params).fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return ModelResult("lightgbm", model, _evaluate("lightgbm", model, X_test, y_test), params)


def train_all(X_train, y_train, X_test, y_test) -> List[ModelResult]:
    log.info("Training 4 models on %d rows × %d features", *X_train.shape)
    return [
        train_logistic(X_train, y_train, X_test, y_test),
        train_random_forest(X_train, y_train, X_test, y_test),
        train_xgboost(X_train, y_train, X_test, y_test),
        train_lightgbm(X_train, y_train, X_test, y_test),
    ]


def pick_best(results: List[ModelResult]) -> ModelResult:
    best = max(results, key=lambda r: r.metrics["roc_auc"])
    log.info("Best model = %s (ROC-AUC=%.4f)", best.name, best.metrics["roc_auc"])
    return best


def save_model(result: ModelResult, path: Path | None = None) -> Path:
    path = path or (settings.model_dir / f"{result.name}.joblib")
    joblib.dump({"model": result.model, "name": result.name, "metrics": result.metrics}, path)
    log.info("Saved %s → %s", result.name, path)
    return path


def save_production(result: ModelResult) -> Path:
    """Promote the winner to a stable filename the API loads."""
    path = settings.model_dir / "production.joblib"
    joblib.dump({"model": result.model, "name": result.name, "metrics": result.metrics}, path)
    log.info("Promoted %s → production: %s", result.name, path)
    return path


def load_production() -> Dict[str, Any]:
    path = settings.model_dir / "production.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No production model at {path}. Run scripts/train_model.py first."
        )
    return joblib.load(path)


def predict_proba_one(record_vec: np.ndarray) -> float:
    bundle = load_production()
    proba = bundle["model"].predict_proba(record_vec)[:, 1]
    return float(proba[0])
