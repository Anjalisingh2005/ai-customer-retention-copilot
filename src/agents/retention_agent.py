"""Agent 4 - Retention Strategy Agent.

The most important node. Retrieves relevant playbook chunks from FAISS, asks
the LLM for a structured retention plan, then maps the chosen action to a
machine-readable offer key the ROI agent can score.
"""
from __future__ import annotations

import re
import time

from src.agents.state import CopilotState
from src.llm.client import get_llm
from src.llm.prompts import RETENTION_SYSTEM, build_retention_user_prompt
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

# Keyword → offer_key in roi_estimator.OFFER_CATALOG.
_OFFER_PATTERNS = [
    (re.compile(r"15\s*%.*12\s*month|loyalty.*lock|1[- ]year.*15\s*%", re.I), "loyalty_discount_15pct_12mo"),
    (re.compile(r"10\s*%.*6\s*month", re.I), "loyalty_discount_10pct_6mo"),
    (re.compile(r"tech\s*support.*free|free.*tech\s*support", re.I), "free_tech_support_6mo"),
    (re.compile(r"online\s*security|security\s*bundle|free.*backup", re.I), "free_security_bundle_12mo"),
    (re.compile(r"dedicated\s*csm|priority\s*support", re.I), "dedicated_csm_priority"),
    (re.compile(r"email.*5\s*%|5\s*%.*email", re.I), "email_only_discount_5pct"),
]


def _pick_offer_key(text: str, segment: str | None) -> str:
    for pat, key in _OFFER_PATTERNS:
        if pat.search(text):
            return key
    # Fallbacks by segment.
    if segment == "High Value + High Risk":
        return "loyalty_discount_15pct_12mo"
    if segment == "Low Value + High Risk":
        return "email_only_discount_5pct"
    return "free_tech_support_6mo"


def _retrieval_query(profile_summary: dict, drivers: list[dict]) -> str:
    parts = [f"segment customer with contract {profile_summary.get('Contract')}"]
    parts.append(f"internet {profile_summary.get('InternetService')}")
    parts.append(f"tenure {profile_summary.get('tenure')} months")
    parts.append(f"sentiment {profile_summary.get('sentiment')}")
    if drivers:
        parts.append("drivers: " + ", ".join(d["pretty"] for d in drivers[:3]))
    return " | ".join(str(p) for p in parts)


def run(state: CopilotState) -> CopilotState:
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    usage = dict(state.get("llm_usage", {"input_tokens": 0, "output_tokens": 0}))

    profile_summary = state.get("profile_summary") or {}
    drivers = state.get("shap_drivers") or []
    segment = state.get("segment")
    proba = state.get("churn_probability")

    if not profile_summary or proba is None:
        errors.append("retention_agent: missing profile_summary or churn_probability")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        retriever = get_retriever()
        query = _retrieval_query(profile_summary, drivers)
        docs = retriever.search(query, k=4)
        retrieved = [{"source": d.source, "score": d.score, "text": d.text} for d in docs]

        # Build LLM prompt.
        customer_summary = {**profile_summary, "segment": segment, "churn_probability": f"{proba:.0%}"}
        driver_strs = [f"{d['pretty']} (impact {d['contribution']:+.2f})" for d in drivers]
        user_prompt = build_retention_user_prompt(
            customer_summary=customer_summary,
            shap_drivers=driver_strs,
            retrieved_context=[d["text"] for d in retrieved],
        )

        llm = get_llm()
        resp = llm.complete(system=RETENTION_SYSTEM, user=user_prompt, max_tokens=900)
        usage["input_tokens"] += resp.input_tokens
        usage["output_tokens"] += resp.output_tokens

        offer_key = _pick_offer_key(resp.text, segment)
        log.info("RetentionAgent → mapped offer=%s, %d retrieved docs", offer_key, len(retrieved))
        trace.append({"agent": "retention", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "retrieved_docs": retrieved,
            "recommendation_markdown": resp.text,
            "primary_offer_key": offer_key,
            "llm_usage": usage,
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("RetentionAgent failed: %s", e)
        errors.append(f"retention_agent: {e}")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}
