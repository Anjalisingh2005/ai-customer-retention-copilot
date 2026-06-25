"""Generate a Telco-Churn-shaped dataset with realistic enhancements.

We don't ship the Kaggle CSV. If `data/raw/telco_churn.csv` is present
(downloaded by the user), we use it as the base and bolt on synthetic
behavioural fields (last_login_days, support_ticket_count, avg_response_time,
nps_score, feature_usage_score, plus a derived sentiment).

If the raw file is absent, we synthesize the *entire* dataset with the same
schema as the Kaggle Telco Customer Churn dataset so every downstream phase
runs end-to-end with zero external dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Kaggle Telco Customer Churn column schema.
TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def _rng() -> np.random.Generator:
    return np.random.default_rng(settings.random_seed)


def _make_id(rng: np.random.Generator) -> str:
    digits = rng.integers(1000, 9999)
    letters = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=5))
    return f"{digits}-{letters}"


def synthesize_telco(n: int = 7043) -> pd.DataFrame:
    """Synthesize a Telco-shaped frame with realistic churn correlations."""
    rng = _rng()
    log.info("Synthesizing %d Telco-shaped customer rows", n)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Tenure: bimodal — many short-tenure (churn risk) and many long-tenure (loyal).
    tenure = np.clip(
        np.where(
            rng.random(n) < 0.45,
            rng.integers(0, 12, n),
            rng.integers(12, 72, n),
        ),
        0,
        72,
    ).astype(int)

    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi = np.where(
        phone == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    internet = rng.choice(
        ["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22]
    )

    def _addon(p_yes: float = 0.40) -> np.ndarray:
        base = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        return np.where(internet == "No", "No internet service", base)

    online_security = _addon(0.36)
    online_backup = _addon(0.40)
    device_protection = _addon(0.40)
    tech_support = _addon(0.37)
    streaming_tv = _addon(0.49)
    streaming_movies = _addon(0.49)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_monthly = np.where(internet == "Fiber optic", 75.0, np.where(internet == "DSL", 45.0, 20.0))
    addon_boost = (
        (online_security == "Yes").astype(float) * 5
        + (online_backup == "Yes").astype(float) * 5
        + (device_protection == "Yes").astype(float) * 5
        + (tech_support == "Yes").astype(float) * 5
        + (streaming_tv == "Yes").astype(float) * 10
        + (streaming_movies == "Yes").astype(float) * 10
    )
    monthly = np.round(base_monthly + addon_boost + rng.normal(0, 4, n), 2).clip(18.25, 119.0)
    total = np.round(monthly * tenure + rng.normal(0, 50, n), 2).clip(min=0)
    # Kaggle quirk: brand-new customers sometimes have blank TotalCharges.
    total_str = np.where(tenure == 0, " ", total.astype(str))

    # ---- Churn label: weighted score so downstream models actually learn signal ----
    score = (
        (contract == "Month-to-month") * 1.2
        + (tenure < 6) * 1.0
        + (monthly > 80) * 0.5
        + (payment == "Electronic check") * 0.6
        + (internet == "Fiber optic") * 0.4
        + (online_security == "No") * 0.3
        + (tech_support == "No") * 0.3
        + (senior == 1) * 0.3
        - (contract == "Two year") * 1.5
        - (tenure > 48) * 0.8
        + rng.normal(0, 1.2, n)
    )
    prob = 1 / (1 + np.exp(-(score - 2.1)))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    ids = [_make_id(rng) for _ in range(n)]
    df = pd.DataFrame(
        {
            "customerID": ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_str,
            "Churn": churn,
        }
    )
    return df[TELCO_COLUMNS]


def add_behavioural_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Bolt on the synthetic behavioural fields that make the project realistic."""
    rng = _rng()
    n = len(df)

    is_churner = (df["Churn"] == "Yes").to_numpy()
    tenure = df["tenure"].to_numpy()

    # last_login_days: churners drift, loyal customers stay engaged.
    # Ranges overlap (3-30) so the classes aren't perfectly separable.
    last_login = np.where(
        is_churner,
        rng.integers(3, 90, n),
        rng.integers(0, 30, n),
    )

    # support_ticket_count: more tickets => more friction.
    base_tickets = rng.poisson(1.0, n)
    extra = np.where(is_churner, rng.poisson(2.0, n), 0)
    tickets = base_tickets + extra

    # avg_response_time (hours): unhappy customers wait longer.
    # Tighter gap so distributions overlap and produce ambiguous cases.
    response = np.round(
        np.where(is_churner, rng.gamma(2.5, 4.0, n), rng.gamma(2.2, 3.0, n)),
        1,
    )

    # nps_score: 0..10 detractor/promoter, overlapping in the 4-7 band.
    nps = np.where(
        is_churner,
        rng.integers(0, 8, n),
        rng.integers(4, 11, n),
    ).astype(int)

    # feature_usage_score 0..1 — overlapping betas so the boundary is fuzzy.
    usage = np.clip(
        np.where(
            is_churner,
            rng.beta(2.5, 4, n),
            rng.beta(4, 2.5, n),
        ),
        0,
        1,
    ).round(3)

    # Derived sentiment from tickets + NPS.
    sentiment = np.where(
        (nps <= 6) | (tickets >= 4),
        "Negative",
        np.where((nps >= 9) & (tickets <= 1), "Positive", "Neutral"),
    )

    enhanced = df.copy()
    enhanced["last_login_days"] = last_login
    enhanced["support_ticket_count"] = tickets
    enhanced["avg_response_time"] = response
    enhanced["nps_score"] = nps
    enhanced["feature_usage_score"] = usage
    enhanced["sentiment"] = sentiment
    # Tag tenure bucket for downstream RFM.
    enhanced["tenure_bucket"] = pd.cut(
        tenure, bins=[-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49+"]
    ).astype(str)
    return enhanced


def build_dataset(n: int | None = None) -> pd.DataFrame:
    """Return base (Kaggle if present, else synthetic) + behavioural fields."""
    raw_path = settings.raw_data_path
    if raw_path.exists():
        log.info("Loading base Telco from %s", raw_path)
        base = pd.read_csv(raw_path)
        missing = [c for c in TELCO_COLUMNS if c not in base.columns]
        if missing:
            raise ValueError(f"Raw Telco CSV missing columns: {missing}")
        base = base[TELCO_COLUMNS]
    else:
        log.warning("No raw Telco CSV at %s — synthesizing one.", raw_path)
        base = synthesize_telco(n or 7043)

    enhanced = add_behavioural_fields(base)
    out = settings.synthetic_data_path
    out.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_csv(out, index=False)
    log.info("Wrote enhanced dataset (%d rows, %d cols) → %s", *enhanced.shape, out)
    return enhanced
