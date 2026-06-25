"""Run the full multi-agent copilot pipeline for a single customer and pretty-print the result."""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from src.agents.orchestrator import run as run_copilot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer-id", required=True)
    ap.add_argument("--json", action="store_true", help="Print full JSON state.")
    args = ap.parse_args()

    state = run_copilot(args.customer_id)
    if args.json:
        # state contains the full profile dict — keep printable size sane.
        printable = {k: v for k, v in state.items() if k != "profile"}
        print(json.dumps(printable, indent=2, default=str))
        return

    print("=" * 72)
    print(f"Customer:        {args.customer_id}")
    print(f"Segment:         {state.get('segment')}  (persona: {state.get('persona')})")
    print(f"Churn risk:      {state.get('churn_probability', 0):.0%}  ({state.get('risk_tier')})")
    print("-" * 72)
    print("SHAP drivers:")
    for d in state.get("shap_drivers", [])[:5]:
        sign = "+" if d["contribution"] >= 0 else "-"
        print(f"  {sign} {d['pretty']:<45s}  ({d['contribution']:+.3f})")
    print("-" * 72)
    print("Explanation:")
    print(f"  {state.get('explanation_text', '')}")
    print("-" * 72)
    print("Retention recommendation:")
    print(state.get("recommendation_markdown", ""))
    print("-" * 72)
    roi = state.get("recommended_roi") or {}
    if roi:
        print("ROI:")
        print(f"  Offer:                   {roi.get('description')}")
        print(f"  Offer cost:              ${roi.get('offer_cost'):.2f}")
        print(f"  Expected revenue saved:  ${roi.get('expected_revenue_saved'):.2f}")
        print(f"  Net value:               ${roi.get('net_value'):.2f}")
        print(f"  ROI multiple:            {roi.get('roi_multiple')}")
        print(f"  Payback months:          {roi.get('payback_months')}")
    print("-" * 72)
    print(f"LLM usage: {state.get('llm_usage')}")
    if state.get("errors"):
        print("ERRORS:")
        for e in state["errors"]:
            print(f"  - {e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
