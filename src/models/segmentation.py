"""Customer segmentation: business quadrants + KMeans behavioural clusters.

Two complementary views:
1. **Business quadrants** (value × risk) - what stakeholders ask for.
2. **KMeans on behavioural fields** - what the data tells us, useful for
   discovering segments the business hasn't named yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.features.engineering import attach_value_risk, rfm_scores
from src.utils.logger import get_logger

log = get_logger(__name__)

BEHAVIOURAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "last_login_days",
    "support_ticket_count",
    "avg_response_time",
    "nps_score",
    "feature_usage_score",
]


@dataclass
class SegmentationBundle:
    scaler: StandardScaler
    kmeans: KMeans
    n_clusters: int
    cluster_personas: dict[int, str]


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> SegmentationBundle:
    X = df[BEHAVIOURAL_COLS].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=settings.random_seed).fit(Xs)

    # Auto-name clusters from their centroid behaviour.
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=BEHAVIOURAL_COLS
    )
    personas: dict[int, str] = {}
    for idx, row in centroids.iterrows():
        if row["nps_score"] >= 8 and row["support_ticket_count"] <= 2:
            personas[idx] = "Loyal Promoter"
        elif row["support_ticket_count"] >= 3 and row["nps_score"] <= 6:
            personas[idx] = "Frustrated"
        elif row["tenure"] <= 12 and row["MonthlyCharges"] >= 70:
            personas[idx] = "New & Expensive"
        elif row["feature_usage_score"] <= 0.3:
            personas[idx] = "Disengaged"
        else:
            personas[idx] = "Steady Mainstream"
    log.info("KMeans personas: %s", personas)
    return SegmentationBundle(scaler=scaler, kmeans=km, n_clusters=n_clusters, cluster_personas=personas)


def assign_clusters(df: pd.DataFrame, bundle: SegmentationBundle) -> pd.Series:
    Xs = bundle.scaler.transform(df[BEHAVIOURAL_COLS].to_numpy())
    return pd.Series(bundle.kmeans.predict(Xs), index=df.index, name="kmeans_cluster")


def segment_dataset(df: pd.DataFrame, churn_proba: np.ndarray, bundle: SegmentationBundle) -> pd.DataFrame:
    """Return df + value/risk/segment + rfm scores + kmeans cluster + persona."""
    out = attach_value_risk(df, churn_proba)
    rfm = rfm_scores(df).drop(columns=["recency_raw", "frequency_raw", "monetary_raw"])
    out = pd.concat([out, rfm], axis=1)
    clusters = assign_clusters(df, bundle)
    out["kmeans_cluster"] = clusters.values
    out["persona"] = out["kmeans_cluster"].map(bundle.cluster_personas)
    return out


def save_bundle(bundle: SegmentationBundle, path: Path | None = None) -> Path:
    path = path or (settings.model_dir / "segmentation.joblib")
    joblib.dump(bundle, path)
    log.info("Saved segmentation bundle → %s", path)
    return path


def load_bundle(path: Path | None = None) -> SegmentationBundle:
    path = path or (settings.model_dir / "segmentation.joblib")
    return joblib.load(path)


def segment_summary(seg_df: pd.DataFrame) -> List[dict]:
    """Per-segment summary suitable for the dashboard."""
    g = seg_df.groupby("segment").agg(
        customers=("MonthlyCharges", "size"),
        avg_churn_prob=("churn_probability", "mean"),
        avg_monthly=("MonthlyCharges", "mean"),
        avg_tenure=("tenure", "mean"),
        total_arr=("MonthlyCharges", lambda s: float(s.sum() * 12)),
    ).reset_index()
    return g.to_dict(orient="records")
