"""Agent 5 - ROI Agent.

Scores the LLM-chosen offer + ranks all alternatives so the dashboard can show
"this is what we picked, here's the rest of the menu and why."
"""
from __future__ import annotations

import time

from src.agents.state import CopilotState
from src.models.roi_estimator import estimate, rank_offers, to_dict
from src.utils.logger import get_logger

log = get_logger(__name__)


def run(state: CopilotState) -> CopilotState:
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    profile = state.get("profile")
    proba = state.get("churn_probability")
    offer_key = state.get("primary_offer_key")

    if not profile or proba is None or not offer_key:
        errors.append("roi_agent: missing profile / churn_probability / offer_key")
        trace.append({"agent": "roi", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        monthly = float(profile["MonthlyCharges"])
        ranked = rank_offers(monthly_charge=monthly, baseline_churn_prob=proba)
        chosen = estimate(offer_key, monthly_charge=monthly, baseline_churn_prob=proba)
        log.info(
            "ROIAgent → offer=%s, expected_saved=$%.0f, cost=$%.0f, roi=%s",
            offer_key, chosen.expected_revenue_saved, chosen.offer_cost, chosen.roi_multiple,
        )
        trace.append({"agent": "roi", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "recommended_roi": to_dict(chosen),
            "roi_estimates": [to_dict(e) for e in ranked],
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("ROIAgent failed: %s", e)
        errors.append(f"roi_agent: {e}")
        trace.append({"agent": "roi", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}
