"""FastAPI surface.

Endpoints map 1:1 to the project spec:
- POST /predict_churn       — churn probability + segment
- POST /customer_analysis   — SHAP-driven explanation
- POST /generate_strategy   — LLM + RAG recommendation
- POST /customer_roi        — ROI table for actions
- POST /copilot/run         — full multi-agent pipeline

Plus GET /health.
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.orchestrator import run as run_copilot
from src.api.schemas import (
    ChurnResponse,
    CopilotResponse,
    CustomerAnalysisResponse,
    CustomerRequest,
    ROIRequest,
    ROIResponse,
    ROIResponseItem,
    ShapDriver,
    StrategyRequest,
    StrategyResponse,
)
from src.data.loader import get_customer
from src.data.preprocessor import load_preprocessor, transform_one
from src.explainability.shap_explainer import load_explainer_cache
from src.features.engineering import risk_tier, segment_label, value_tier
from src.models.churn_predictor import load_production
from src.models.roi_estimator import estimate, rank_offers, to_dict
from src.models.segmentation import BEHAVIOURAL_COLS, load_bundle
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Customer Retention Copilot",
    version="0.1.0",
    description="Churn prediction + SHAP + LLM/RAG + LangGraph multi-agent retention copilot.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _profile_or_404(customer_id: str) -> dict:
    p = get_customer(customer_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")
    return p


def _persona_for(profile: dict) -> str:
    """Look up the KMeans persona for a single customer. Returns "Unknown" if the
    segmentation bundle is unavailable or any feature is missing — never raises,
    so it can't break /predict_churn."""
    try:
        import numpy as np

        bundle = load_bundle()
        row = np.array([[profile[c] for c in BEHAVIOURAL_COLS]], dtype=float)
        cluster = int(bundle.kmeans.predict(bundle.scaler.transform(row))[0])
        return bundle.cluster_personas.get(cluster, "Unknown")
    except Exception as e:
        log.warning("persona lookup failed: %s", e)
        return "Unknown"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict_churn", response_model=ChurnResponse)
def predict_churn(req: CustomerRequest) -> ChurnResponse:
    profile = _profile_or_404(req.customer_id)
    bundle = load_production()
    pre = load_preprocessor()
    X = transform_one(profile, pre)
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    vt = value_tier(float(profile["MonthlyCharges"]), int(profile["tenure"]))
    rt = risk_tier(proba)
    seg = segment_label(vt, rt)
    return ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=round(proba, 4),
        risk_tier=rt,
        value_tier=vt,
        segment=seg,
        persona=_persona_for(profile),
    )


@app.post("/customer_analysis", response_model=CustomerAnalysisResponse)
def customer_analysis(req: CustomerRequest) -> CustomerAnalysisResponse:
    profile = _profile_or_404(req.customer_id)
    explainer = load_explainer_cache()
    expl = explainer.explain_record(profile, top_k=5)
    drivers = [
        ShapDriver(pretty=d.pretty, contribution=round(d.contribution, 4), value=str(d.value) if d.value is not None else None)
        for d in expl.drivers
    ]
    text = (
        f"This customer has a churn probability of {expl.prediction:.0%}. "
        f"The strongest driver is '{drivers[0].pretty if drivers else 'unknown'}'."
    )
    return CustomerAnalysisResponse(
        customer_id=req.customer_id,
        churn_probability=round(expl.prediction, 4),
        explanation_text=text,
        drivers=drivers,
    )


@app.post("/generate_strategy", response_model=StrategyResponse)
def generate_strategy(req: StrategyRequest) -> StrategyResponse:
    state = run_copilot(req.customer_id)
    if state.get("errors"):
        log.warning("/generate_strategy errors: %s", state["errors"])
    return StrategyResponse(
        customer_id=req.customer_id,
        segment=state.get("segment", "unknown"),
        recommendation_markdown=state.get("recommendation_markdown", ""),
        primary_offer_key=state.get("primary_offer_key", ""),
        retrieved_sources=list({d["source"] for d in state.get("retrieved_docs", [])}),
    )


@app.post("/customer_roi", response_model=ROIResponse)
def customer_roi(req: ROIRequest) -> ROIResponse:
    profile = _profile_or_404(req.customer_id)
    monthly = float(profile["MonthlyCharges"])
    if req.offer_key:
        # Fast path: caller supplied an offer key.
        bundle = load_production()
        pre = load_preprocessor()
        proba = float(bundle["model"].predict_proba(transform_one(profile, pre))[0, 1])
        chosen = estimate(req.offer_key, monthly_charge=monthly, baseline_churn_prob=proba)
        ranked = rank_offers(monthly_charge=monthly, baseline_churn_prob=proba, exclude={req.offer_key})
    else:
        # Slow path: run full copilot so the LLM picks the offer.
        state = run_copilot(req.customer_id)
        offer_key = state.get("primary_offer_key", "free_tech_support_6mo")
        proba = state.get("churn_probability", 0.5)
        chosen = estimate(offer_key, monthly_charge=monthly, baseline_churn_prob=proba)
        ranked = rank_offers(monthly_charge=monthly, baseline_churn_prob=proba, exclude={offer_key})

    return ROIResponse(
        customer_id=req.customer_id,
        chosen=ROIResponseItem(**to_dict(chosen)),
        alternatives=[ROIResponseItem(**to_dict(e)) for e in ranked],
    )


@app.post("/copilot/run", response_model=CopilotResponse)
def copilot_run(req: CustomerRequest) -> CopilotResponse:
    """Full multi-agent flow: Profile → Risk → Explanation → Retention → ROI."""
    state = run_copilot(req.customer_id)
    if "churn_probability" not in state:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {state.get('errors')}")
    drivers = [ShapDriver(**d) for d in state.get("shap_drivers", [])]
    rec_roi = state.get("recommended_roi") or {}
    alt = state.get("roi_estimates") or []
    return CopilotResponse(
        customer_id=req.customer_id,
        churn_probability=round(state["churn_probability"], 4),
        segment=state.get("segment", "unknown"),
        persona=state.get("persona", "unknown"),
        explanation_text=state.get("explanation_text", ""),
        drivers=drivers,
        recommendation_markdown=state.get("recommendation_markdown", ""),
        primary_offer_key=state.get("primary_offer_key", ""),
        recommended_roi=ROIResponseItem(**rec_roi) if rec_roi else ROIResponseItem(
            offer_key="", description="", expected_revenue_saved=0, offer_cost=0,
            net_value=0, horizon_months=12,
        ),
        alternatives=[ROIResponseItem(**a) for a in alt if a.get("offer_key") != state.get("primary_offer_key")],
        retrieved_sources=list({d["source"] for d in state.get("retrieved_docs", [])}),
        llm_usage=state.get("llm_usage", {}),
        errors=state.get("errors", []),
    )
