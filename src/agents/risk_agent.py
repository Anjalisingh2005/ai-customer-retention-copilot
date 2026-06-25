"""Agent 2 - Risk Assessment Agent.

Loads the production model + segmentation bundle. Predicts churn probability,
computes value/risk tiers and the business segment + KMeans persona.
"""
from __future__ import annotations

import time

from src.agents.state import CopilotState
from src.data.preprocessor import load_preprocessor, transform_one
from src.features.engineering import risk_tier, segment_label, value_tier
from src.models.churn_predictor import load_production
from src.models.segmentation import BEHAVIOURAL_COLS, load_bundle
from src.utils.logger import get_logger

log = get_logger(__name__)


def run(state: CopilotState) -> CopilotState:
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    profile = state.get("profile")
    if not profile:
        errors.append("risk_agent: missing profile")
        trace.append({"agent": "risk", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        pre = load_preprocessor()
        bundle = load_production()
        model = bundle["model"]
        X = transform_one(profile, pre)
        proba = float(model.predict_proba(X)[0, 1])

        vt = value_tier(float(profile["MonthlyCharges"]), int(profile["tenure"]))
        rt = risk_tier(proba)
        seg = segment_label(vt, rt)

        # KMeans persona, if available.
        persona = "Unknown"
        try:
            kb = load_bundle()
            import numpy as np

            row = np.array([[profile[c] for c in BEHAVIOURAL_COLS]], dtype=float)
            cluster = int(kb.kmeans.predict(kb.scaler.transform(row))[0])
            persona = kb.cluster_personas.get(cluster, "Unknown")
        except Exception as e:
            log.warning("Persona lookup failed: %s", e)

        log.info(
            "RiskAgent → p_churn=%.3f, value=%s, risk=%s, seg=%s, persona=%s",
            proba, vt, rt, seg, persona,
        )
        trace.append({"agent": "risk", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "churn_probability": proba,
            "value_tier": vt,
            "risk_tier": rt,
            "segment": seg,
            "persona": persona,
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("RiskAgent failed: %s", e)
        errors.append(f"risk_agent: {e}")
        trace.append({"agent": "risk", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}
