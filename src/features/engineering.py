"""Derived features used by segmentation + dashboards.

Kept separate from the model preprocessor so we can use them for human-readable
reporting (e.g., RFM scores in the dashboard) without bloating the model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def customer_lifetime_value(df: pd.DataFrame) -> pd.Series:
    """Naive CLV proxy: monthly_charge * tenure (already in TotalCharges, but
    we recompute so synthetic-only rows are consistent)."""
    total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return total.where(total > 0, df["MonthlyCharges"] * df["tenure"])


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """RFM-style scoring for telco:
    - Recency  = -last_login_days   (higher = better)
    - Frequency = feature_usage_score
    - Monetary = MonthlyCharges
    Each scored 1..5 via quintile binning.
    """
    out = pd.DataFrame(index=df.index)
    out["recency_raw"] = -df["last_login_days"]
    out["frequency_raw"] = df["feature_usage_score"]
    out["monetary_raw"] = df["MonthlyCharges"]

    for col in ["recency", "frequency", "monetary"]:
        raw = out[f"{col}_raw"]
        try:
            out[f"{col}_score"] = pd.qcut(raw.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        except ValueError:
            # qcut fails when there are fewer unique ranks than bins. Fall back to a
            # uniform mid-score, but keep it as an index-aligned Series so the
            # subsequent rfm_score arithmetic stays element-wise.
            out[f"{col}_score"] = pd.Series(3, index=out.index, dtype=int)
    out["rfm_score"] = out["recency_score"] + out["frequency_score"] + out["monetary_score"]
    return out


def value_tier(monthly_charge: float, tenure: int) -> str:
    """Quick value bucket used by the segmentation step."""
    annualised = monthly_charge * 12
    if annualised >= 900 or (monthly_charge >= 70 and tenure >= 24):
        return "High"
    if annualised >= 500:
        return "Mid"
    return "Low"


def risk_tier(churn_probability: float) -> str:
    if churn_probability >= 0.6:
        return "High"
    if churn_probability >= 0.3:
        return "Medium"
    return "Low"


def segment_label(value: str, risk: str) -> str:
    """Map (value, risk) → the four canonical segments."""
    if value == "High" and risk == "High":
        return "High Value + High Risk"
    if value == "High" and risk in {"Medium", "Low"}:
        return "High Value + Low Risk"
    if value in {"Low", "Mid"} and risk == "High":
        return "Low Value + High Risk"
    return "Low Value + Low Risk"


def attach_value_risk(df: pd.DataFrame, churn_proba: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["value_tier"] = [value_tier(m, t) for m, t in zip(df["MonthlyCharges"], df["tenure"])]
    out["churn_probability"] = churn_proba
    out["risk_tier"] = [risk_tier(p) for p in churn_proba]
    out["segment"] = [segment_label(v, r) for v, r in zip(out["value_tier"], out["risk_tier"])]
    return out
