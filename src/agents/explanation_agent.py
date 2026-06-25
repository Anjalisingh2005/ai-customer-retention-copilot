"""Agent 3 — Explanation Agent.

Runs SHAP on the production model for this customer; returns the top-k drivers
and a short LLM-polished plain-English sentence.
"""
from __future__ import annotations

import time

from src.agents.state import CopilotState
from src.explainability.shap_explainer import load_explainer_cache
from src.llm.client import get_llm
from src.llm.prompts import EXPLANATION_SYSTEM
from src.utils.logger import get_logger

log = get_logger(__name__)


def run(state: CopilotState) -> CopilotState:
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    profile = state.get("profile")
    proba = state.get("churn_probability")
    usage = dict(state.get("llm_usage", {"input_tokens": 0, "output_tokens": 0}))

    if not profile or proba is None:
        errors.append("explanation_agent: missing profile or churn_probability")
        trace.append({"agent": "explanation", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        explainer = load_explainer_cache()
        result = explainer.explain_record(profile, top_k=5)
        drivers = [
            {"pretty": d.pretty, "contribution": d.contribution, "value": str(d.value) if d.value is not None else None}
            for d in result.drivers
        ]

        drivers_text = "\n".join(
            f"- {d['pretty']} (impact={d['contribution']:+.3f})" for d in drivers
        )
        user_msg = (
            f"Churn probability: {proba:.0%}.\n"
            f"Top SHAP drivers:\n{drivers_text}\n\n"
            "Write a 2-sentence plain-English explanation a CSM can read at the top of a call brief."
        )
        llm = get_llm()
        resp = llm.complete(system=EXPLANATION_SYSTEM, user=user_msg, max_tokens=200)
        usage["input_tokens"] += resp.input_tokens
        usage["output_tokens"] += resp.output_tokens

        log.info("ExplanationAgent → %d drivers, %d output tokens", len(drivers), resp.output_tokens)
        trace.append({"agent": "explanation", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "shap_drivers": drivers,
            "explanation_text": resp.text.strip(),
            "llm_usage": usage,
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("ExplanationAgent failed: %s", e)
        errors.append(f"explanation_agent: {e}")
        trace.append({"agent": "explanation", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}
