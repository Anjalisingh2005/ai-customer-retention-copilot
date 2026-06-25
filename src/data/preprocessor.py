"""Encoders + train/test split. One source of truth for feature ordering.

We persist the ColumnTransformer so the API loads the exact same encoder used
at training time — no drift between train/predict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns we DROP entirely (id, leakage, raw-but-derived).
DROP_COLS = ["customerID"]

# Categorical columns from the enhanced schema.
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "sentiment",
    "tenure_bucket",
]

NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "last_login_days",
    "support_ticket_count",
    "avg_response_time",
    "nps_score",
    "feature_usage_score",
]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    preprocessor: ColumnTransformer
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges has blank strings for tenure=0 in the Kaggle CSV.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[settings.target_col] = (df[settings.target_col] == "Yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """The encoder used at both train- and serve-time."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                NUMERIC_COLS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare(df: pd.DataFrame) -> PreparedData:
    df = _clean(df)
    y = df[settings.target_col].to_numpy()
    feat_df = df.drop(columns=[settings.target_col, *DROP_COLS])

    raw_train, raw_test, y_train, y_test = train_test_split(
        feat_df,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_seed,
    )

    pre = build_preprocessor()
    X_train = pre.fit_transform(raw_train)
    X_test = pre.transform(raw_test)

    feature_names = list(pre.get_feature_names_out())

    log.info("Train shape %s, test shape %s, %d features", X_train.shape, X_test.shape, len(feature_names))
    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=pre,
        raw_train=raw_train,
        raw_test=raw_test,
    )


def save_preprocessor(pre: ColumnTransformer, path: Path | None = None) -> Path:
    path = path or (settings.encoder_dir / "preprocessor.joblib")
    joblib.dump(pre, path)
    log.info("Saved preprocessor → %s", path)
    return path


def load_preprocessor(path: Path | None = None) -> ColumnTransformer:
    path = path or (settings.encoder_dir / "preprocessor.joblib")
    return joblib.load(path)


def transform_one(record: dict, pre: ColumnTransformer | None = None) -> np.ndarray:
    """Encode a single customer record at predict-time."""
    pre = pre or load_preprocessor()
    df = pd.DataFrame([record])
    # Drop label / id if accidentally passed in.
    for c in [settings.target_col, *DROP_COLS]:
        if c in df.columns:
            df = df.drop(columns=[c])
    # Ensure numeric coercion for TotalCharges.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return pre.transform(df)
