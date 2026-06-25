"""All prompt templates. One file so we can iterate on language without hunting."""
from __future__ import annotations

from typing import List

RETENTION_SYSTEM = """\
You are a senior Customer Retention Strategist for a telecom company.
You combine quantitative churn signals (model probability, SHAP drivers, segment)
with qualitative behavioural data (support tickets, NPS, sentiment) to propose
ACTIONABLE retention plans.

Hard rules:
- Always recommend ONE primary action and at most TWO supporting actions.
- Be specific: name the offer, the discount %, the channel, and the owner.
- Quantify expected impact (retention probability lift, ARR saved).
- Use ONLY the information provided in the user message and the retrieved context.
  If something is unknown, say "unknown" — never invent ticket counts, NPS, or history.
- Output must follow the exact section headers in the user template.
"""


def build_retention_user_prompt(
    *,
    customer_summary: dict,
    shap_drivers: List[str],
    retrieved_context: List[str],
) -> str:
    drivers_block = "\n".join(f"- {d}" for d in shap_drivers) or "- (no SHAP drivers)"
    context_block = (
        "\n\n".join(f"[Doc {i+1}]\n{c}" for i, c in enumerate(retrieved_context))
        if retrieved_context
        else "(no retrieved context)"
    )
    summary_lines = "\n".join(f"- {k}: {v}" for k, v in customer_summary.items())

    return f"""\
## Customer
{summary_lines}

## Top churn drivers (from SHAP)
{drivers_block}

## Retrieved retention knowledge (policies, past campaigns, playbooks)
{context_block}

## Required output (use these exact headers)
### Risk Analysis
2-4 sentence diagnosis of WHY this customer is likely to churn, citing the drivers above.

### Recommended Action Plan
1. Primary action — be specific (offer, %, channel, owner).
2. Supporting action (optional).
3. Supporting action (optional).

### Expected Impact
- Estimated retention probability after action: NN%
- Estimated ARR saved if retained for 12 months: $X
- Confidence: Low / Medium / High (with one-line justification)

### Talk Track
A 3-4 sentence script the CSM can read on the call.
"""


CSAT_TICKET_SYSTEM = """\
You classify support ticket text into severity (low/medium/high/critical)
and sentiment (positive/neutral/negative). Reply as compact JSON only.
"""

EXPLANATION_SYSTEM = """\
You translate ML SHAP outputs into a 2-sentence plain-English explanation for
a non-technical CSM. No jargon. No numbers beyond the churn probability itself.
"""
