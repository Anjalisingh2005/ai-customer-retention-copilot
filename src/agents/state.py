"""Shared LangGraph state that all 5 agents read/write."""
from __future__ import annotations

from typing import Any, List, Optional, TypedDict


class CopilotState(TypedDict, total=False):
    # --- inputs ---
    customer_id: str

    # --- profile agent ---
    profile: dict
    profile_summary: dict          # compact key-value used in prompts

    # --- risk agent ---
    churn_probability: float
    risk_tier: str
    value_tier: str
    segment: str
    persona: str

    # --- explanation agent ---
    shap_drivers: List[dict]       # [{pretty, contribution, value}]
    explanation_text: str          # plain-English 2-sentence rationale

    # --- retention agent ---
    retrieved_docs: List[dict]
    recommendation_markdown: str   # full LLM response
    primary_offer_key: str         # which offer the LLM picked (mapped)

    # --- ROI agent ---
    roi_estimates: List[dict]      # all offers ranked
    recommended_roi: dict          # ROI for the chosen offer

    # --- meta ---
    errors: List[str]
    llm_usage: dict
    trace: List[dict]              # per-node timing/log entries
