"""Smoke tests for the data layer."""
from __future__ import annotations

import pandas as pd

from src.data.synthetic_generator import (
    TELCO_COLUMNS,
    add_behavioural_fields,
    synthesize_telco,
)


def test_synthesize_schema():
    df = synthesize_telco(n=200)
    assert list(df.columns) == TELCO_COLUMNS
    assert df["Churn"].isin(["Yes", "No"]).all()
    assert (df["tenure"] >= 0).all() and (df["tenure"] <= 72).all()


def test_behavioural_fields_added():
    df = synthesize_telco(n=200)
    enh = add_behavioural_fields(df)
    for col in [
        "last_login_days",
        "support_ticket_count",
        "avg_response_time",
        "nps_score",
        "feature_usage_score",
        "sentiment",
        "tenure_bucket",
    ]:
        assert col in enh.columns
    assert enh["nps_score"].between(0, 10).all()
    assert enh["feature_usage_score"].between(0, 1).all()
    assert enh["sentiment"].isin(["Positive", "Neutral", "Negative"]).all()


def test_churn_signal_present():
    """If our synthetic generator can't even produce a usable signal, the rest is moot."""
    df = add_behavioural_fields(synthesize_telco(n=2000))
    churn = (df["Churn"] == "Yes").astype(int)
    # Month-to-month should have higher churn rate than Two year.
    rate_mtm = churn[df["Contract"] == "Month-to-month"].mean()
    rate_2y = churn[df["Contract"] == "Two year"].mean()
    assert rate_mtm > rate_2y
