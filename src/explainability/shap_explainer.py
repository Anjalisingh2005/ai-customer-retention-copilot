"""SHAP-based per-customer explanations.

For tree models we use shap.TreeExplainer (fast, exact). For LR we fall back
to a LinearExplainer. Output is a ranked list of human-readable drivers like:

    Contract Month-to-Month  +31%
    Low tenure               +22%
    High monthly charges     +15%

so the retention agent can weave them into the prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np

from src.config import settings
from src.data.preprocessor import load_preprocessor
from src.models.churn_predictor import load_production
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ShapDriver:
    feature: str
    pretty: str
    contribution: float  # signed; positive means pushes toward churn
    value: object | None = None


@dataclass
class ShapExplanation:
    base_value: float
    prediction: float
    drivers: List[ShapDriver]


_PRETTY_RULES = [
    (r"^Contract_(.+)", lambda m, v: f"Contract: {m.group(1)}"),
    (r"^InternetService_(.+)", lambda m, v: f"Internet service: {m.group(1)}"),
    (r"^PaymentMethod_(.+)", lambda m, v: f"Payment method: {m.group(1)}"),
    (r"^tenure$", lambda m, v: f"Tenure ({int(v)} months)"),
    (r"^MonthlyCharges$", lambda m, v: f"Monthly charges (${v:.0f})"),
    (r"^TotalCharges$", lambda m, v: f"Total charges (${v:.0f})"),
    (r"^last_login_days$", lambda m, v: f"Days since last login ({int(v)})"),
    (r"^support_ticket_count$", lambda m, v: f"Support tickets ({int(v)})"),
    (r"^avg_response_time$", lambda m, v: f"Avg response time ({v:.1f}h)"),
    (r"^nps_score$", lambda m, v: f"NPS ({int(v)})"),
    (r"^feature_usage_score$", lambda m, v: f"Feature usage ({v:.2f})"),
    (r"^sentiment_(.+)", lambda m, v: f"Sentiment: {m.group(1)}"),
    (r"^tenure_bucket_(.+)", lambda m, v: f"Tenure bucket: {m.group(1)}"),
    (r"^TechSupport_(.+)", lambda m, v: f"Tech support: {m.group(1)}"),
    (r"^OnlineSecurity_(.+)", lambda m, v: f"Online security: {m.group(1)}"),
    (r"^OnlineBackup_(.+)", lambda m, v: f"Online backup: {m.group(1)}"),
    (r"^Paperless.*", lambda m, v: "Paperless billing"),
    (r"^SeniorCitizen$", lambda m, v: "Senior citizen"),
]


def _prettify(feature_name: str, value: object | None) -> str:
    for pat, fn in _PRETTY_RULES:
        m = re.match(pat, feature_name)
        if m:
            try:
                return fn(m, value)
            except Exception:
                return feature_name
    return feature_name.replace("_", " ")


class ChurnExplainer:
    def __init__(self):
        bundle = load_production()
        self.model = bundle["model"]
        self.model_name = bundle["name"]
        self.preprocessor = load_preprocessor()
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self._explainer = self._build_explainer()

    def _build_explainer(self):
        import shap

        name = type(self.model).__name__.lower()
        if "xgb" in name or "lgbm" in name or "lightgbm" in name or "forest" in name:
            return shap.TreeExplainer(self.model)
        # Logistic regression / linear fallback.
        # We need a background sample, but we don't always have one cached;
        # use a small zeros background - fine for LR with standardized inputs.
        background = np.zeros((1, len(self.feature_names)))
        return shap.LinearExplainer(self.model, background)

    def explain_vector(self, X_row: np.ndarray, raw_values: dict | None = None, top_k: int = 5) -> ShapExplanation:
        import shap

        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)

        shap_values = self._explainer.shap_values(X_row)
        # Binary tree models can return a list [neg, pos] or a (n, f) array; normalise to pos-class contribs.
        if isinstance(shap_values, list):
            contribs = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])[0]
            expected = self._explainer.expected_value[1] if isinstance(self._explainer.expected_value, (list, np.ndarray)) else float(self._explainer.expected_value)
        else:
            arr = np.asarray(shap_values)
            contribs = arr[0]
            ev = self._explainer.expected_value
            expected = float(ev[0] if hasattr(ev, "__len__") else ev)

        proba = float(self.model.predict_proba(X_row)[0, 1])

        order = np.argsort(np.abs(contribs))[::-1][:top_k]
        drivers: List[ShapDriver] = []
        for i in order:
            feat = self.feature_names[i]
            raw_val = (raw_values or {}).get(feat.split("_", 1)[0]) if raw_values else None
            drivers.append(
                ShapDriver(
                    feature=feat,
                    pretty=_prettify(feat, raw_val),
                    contribution=float(contribs[i]),
                    value=raw_val,
                )
            )
        return ShapExplanation(base_value=expected, prediction=proba, drivers=drivers)

    def explain_record(self, record: dict, top_k: int = 5) -> ShapExplanation:
        from src.data.preprocessor import transform_one

        X = transform_one(record, self.preprocessor)
        return self.explain_vector(X, raw_values=record, top_k=top_k)


def save_explainer_cache(explainer: ChurnExplainer, path: Path | None = None) -> Path:
    """Persist the wrapper so the API doesn't pay the build cost on every request."""
    path = path or (settings.model_dir / "explainer.joblib")
    joblib.dump(explainer, path)
    log.info("Saved explainer → %s", path)
    return path


def load_explainer_cache(path: Path | None = None) -> ChurnExplainer:
    path = path or (settings.model_dir / "explainer.joblib")
    if path.exists():
        return joblib.load(path)
    log.info("No cached explainer; building fresh.")
    return ChurnExplainer()
