"""Agent 1 - Customer Profile Agent.

Loads the enhanced record for a customer and builds a compact summary the
downstream agents will use in their prompts.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from src.agents.state import CopilotState
from src.data.loader import get_customer
from src.utils.logger import get_logger

log = get_logger(__name__)


def _summarise(profile: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod",
        "support_ticket_count",
        "avg_response_time",
        "nps_score",
        "feature_usage_score",
        "last_login_days",
        "sentiment",
    ]
    return {k: profile.get(k) for k in fields if k in profile}


def run(state: CopilotState) -> CopilotState:
    t0 = time.time()
    customer_id = state["customer_id"]
    profile = get_customer(customer_id)
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    if profile is None:
        errors.append(f"customer_id {customer_id} not found")
        trace.append({"agent": "profile", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    summary = _summarise(profile)
    log.info("ProfileAgent → built summary for %s (%d fields)", customer_id, len(summary))
    trace.append({"agent": "profile", "ok": True, "ms": int((time.time() - t0) * 1000)})
    return {
        **state,
        "profile": profile,
        "profile_summary": summary,
        "trace": trace,
        "errors": errors,
    }
