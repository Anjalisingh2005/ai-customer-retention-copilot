"""Wire the 5 agents into a LangGraph state machine.

Profile -> Risk -> Explanation -> Retention -> ROI -> END

If LangGraph isn't installed (e.g. lightweight CI), we transparently fall back
to a manual sequential runner with the same node signatures.
"""
from __future__ import annotations

from typing import Callable, List

from src.agents import (
    explanation_agent,
    profile_agent,
    retention_agent,
    risk_agent,
    roi_agent,
)
from src.agents.state import CopilotState
from src.utils.logger import get_logger

log = get_logger(__name__)

_PIPELINE: List[tuple[str, Callable[[CopilotState], CopilotState]]] = [
    ("profile", profile_agent.run),
    ("risk", risk_agent.run),
    ("explanation", explanation_agent.run),
    ("retention", retention_agent.run),
    ("roi", roi_agent.run),
]


def _build_langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    g = StateGraph(CopilotState)
    for name, fn in _PIPELINE:
        g.add_node(name, fn)
    g.add_edge(START, "profile")
    g.add_edge("profile", "risk")
    g.add_edge("risk", "explanation")
    g.add_edge("explanation", "retention")
    g.add_edge("retention", "roi")
    g.add_edge("roi", END)
    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = _build_langgraph()
    return _compiled


def run_sequential(initial: CopilotState) -> CopilotState:
    """Fallback runner — same semantics as the LangGraph compiled flow."""
    state = dict(initial)
    state.setdefault("trace", [])
    state.setdefault("errors", [])
    state.setdefault("llm_usage", {"input_tokens": 0, "output_tokens": 0})
    for name, fn in _PIPELINE:
        log.info("→ running agent: %s", name)
        state = fn(state)  # type: ignore[arg-type]
    return state  # type: ignore[return-value]


def run(customer_id: str) -> CopilotState:
    initial: CopilotState = {
        "customer_id": customer_id,
        "trace": [],
        "errors": [],
        "llm_usage": {"input_tokens": 0, "output_tokens": 0},
    }
    graph = get_graph()
    if graph is None:
        log.warning("LangGraph not available — running sequential fallback.")
        return run_sequential(initial)
    return graph.invoke(initial)
