"""Pydantic request/response schemas for the FastAPI surface."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerRequest(BaseModel):
    customer_id: str = Field(..., description="Telco customerID, e.g. 7590-VHVEG")


class CustomerRecordRequest(BaseModel):
    """For ad-hoc predictions where the caller passes a raw record (no DB lookup)."""

    record: Dict[str, Any]


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    value_tier: str
    segment: str
    persona: str


class ShapDriver(BaseModel):
    pretty: str
    contribution: float
    value: Optional[str] = None


class CustomerAnalysisResponse(BaseModel):
    customer_id: str
    churn_probability: float
    explanation_text: str
    drivers: List[ShapDriver]


class StrategyRequest(BaseModel):
    customer_id: str


class StrategyResponse(BaseModel):
    customer_id: str
    segment: str
    recommendation_markdown: str
    primary_offer_key: str
    retrieved_sources: List[str]


class ROIRequest(BaseModel):
    customer_id: str
    offer_key: Optional[str] = Field(
        default=None,
        description="If omitted, the API runs the LangGraph pipeline and uses the LLM-chosen offer.",
    )


class ROIResponseItem(BaseModel):
    # ROIEstimate dataclass carries extra fields (baseline_churn_prob, retained_churn_prob,
    # expected_acceptance_rate) we don't want to expose in the API response. Ignore them
    # so `ROIResponseItem(**to_dict(estimate))` doesn't raise.
    model_config = ConfigDict(extra="ignore")

    offer_key: str
    description: str
    expected_revenue_saved: float
    offer_cost: float
    net_value: float
    roi_multiple: Optional[float] = None
    payback_months: Optional[float] = None
    horizon_months: int


class ROIResponse(BaseModel):
    customer_id: str
    chosen: ROIResponseItem
    alternatives: List[ROIResponseItem]


class CopilotResponse(BaseModel):
    customer_id: str
    churn_probability: float
    segment: str
    persona: str
    explanation_text: str
    drivers: List[ShapDriver]
    recommendation_markdown: str
    primary_offer_key: str
    recommended_roi: ROIResponseItem
    alternatives: List[ROIResponseItem]
    retrieved_sources: List[str]
    llm_usage: Dict[str, int]
    errors: List[str]
