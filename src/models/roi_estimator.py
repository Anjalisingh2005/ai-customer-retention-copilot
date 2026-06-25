"""ROI calculator for retention actions.

Inputs:
- baseline churn probability (from the model)
- expected acceptance rate of the offer (from past campaigns)
- expected retention lift if accepted
- offer cost (from policy table)
- customer ARPU + tenure

Outputs:
- expected_revenue_saved
- offer_cost
- net_value
- roi_multiple
- payback_months
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


# Reference offer cost / lift table. In production this would live in a config
# file or a DB updated by the analytics team.
OFFER_CATALOG: dict[str, dict] = {
    "loyalty_discount_15pct_12mo": {
        "cost": 0.0,  # opportunity cost computed dynamically as 15% * 12 * monthly
        "cost_kind": "pct_of_monthly",
        "pct": 0.15,
        "months": 12,
        "acceptance_rate": 0.55,
        "retention_lift": 0.40,
        "description": "15% discount for 12 months, contract switch to 1-year.",
    },
    "loyalty_discount_10pct_6mo": {
        "cost": 0.0,
        "cost_kind": "pct_of_monthly",
        "pct": 0.10,
        "months": 6,
        "acceptance_rate": 0.50,
        "retention_lift": 0.30,
        "description": "10% discount for 6 months.",
    },
    "free_tech_support_6mo": {
        "cost": 30.0,
        "cost_kind": "flat",
        "acceptance_rate": 0.65,
        "retention_lift": 0.20,
        "description": "Free TechSupport add-on for 6 months.",
    },
    "free_security_bundle_12mo": {
        "cost": 40.0,
        "cost_kind": "flat",
        "acceptance_rate": 0.70,
        "retention_lift": 0.25,
        "description": "Free OnlineSecurity + OnlineBackup for 12 months.",
    },
    "dedicated_csm_priority": {
        "cost": 450.0,
        "cost_kind": "flat",
        "acceptance_rate": 0.60,
        "retention_lift": 0.35,
        "description": "Assigned CSM + priority support queue for 12 months.",
    },
    "email_only_discount_5pct": {
        "cost": 0.0,
        "cost_kind": "pct_of_monthly",
        "pct": 0.05,
        "months": 6,
        "acceptance_rate": 0.18,
        "retention_lift": 0.10,
        "description": "Low-touch 5% email discount, 6 months.",
    },
}


@dataclass
class ROIEstimate:
    offer_key: str
    description: str
    baseline_churn_prob: float
    retained_churn_prob: float
    expected_acceptance_rate: float
    offer_cost: float
    expected_revenue_saved: float
    net_value: float
    roi_multiple: float
    payback_months: float | None
    horizon_months: int


def _offer_cost(offer: dict, monthly_charge: float) -> float:
    if offer["cost_kind"] == "flat":
        return float(offer["cost"])
    pct = offer["pct"]
    months = offer["months"]
    return float(monthly_charge * pct * months)


def estimate(
    offer_key: str,
    *,
    monthly_charge: float,
    baseline_churn_prob: float,
    horizon_months: int = 12,
) -> ROIEstimate:
    if offer_key not in OFFER_CATALOG:
        raise KeyError(f"Unknown offer {offer_key}. Known: {list(OFFER_CATALOG)}")
    o = OFFER_CATALOG[offer_key]
    accept = o["acceptance_rate"]
    lift = o["retention_lift"]
    # Effective reduction in churn = acceptance × lift, clipped at baseline.
    eff_lift = min(accept * lift, baseline_churn_prob)
    retained_p = max(baseline_churn_prob - eff_lift, 0.0)

    revenue_per_month = monthly_charge
    expected_revenue_saved = eff_lift * revenue_per_month * horizon_months
    cost = _offer_cost(o, monthly_charge)
    net = expected_revenue_saved - cost
    roi = (expected_revenue_saved / cost) if cost > 0 else float("inf")
    payback = (cost / (eff_lift * revenue_per_month)) if (eff_lift * revenue_per_month) > 0 else None

    return ROIEstimate(
        offer_key=offer_key,
        description=o["description"],
        baseline_churn_prob=float(baseline_churn_prob),
        retained_churn_prob=float(retained_p),
        expected_acceptance_rate=float(accept),
        offer_cost=round(cost, 2),
        expected_revenue_saved=round(expected_revenue_saved, 2),
        net_value=round(net, 2),
        roi_multiple=(round(roi, 2) if cost > 0 else None),
        payback_months=(round(payback, 1) if payback is not None else None),
        horizon_months=horizon_months,
    )


def rank_offers(
    monthly_charge: float,
    baseline_churn_prob: float,
    horizon_months: int = 12,
    exclude: set[str] | None = None,
) -> list[ROIEstimate]:
    exclude = exclude or set()
    estimates = [
        estimate(k, monthly_charge=monthly_charge, baseline_churn_prob=baseline_churn_prob, horizon_months=horizon_months)
        for k in OFFER_CATALOG
        if k not in exclude
    ]
    estimates.sort(key=lambda e: e.net_value, reverse=True)
    return estimates


def to_dict(e: ROIEstimate) -> dict:
    return asdict(e)
