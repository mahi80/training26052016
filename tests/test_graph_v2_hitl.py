"""WHY THIS EXISTS
---------------
Proof of the v2 wiring on the REAL compiled graph, offline: supervisor ->
worker -> validator -> {END | human_review}, and the interrupt/resume cycle
that makes human review a *pause*, not a poll.

The HITL mechanics under test (CHALLENGES_GUIDE.md §6, now implemented):

1. ``interrupt(payload)`` inside human_review stops the run; ``invoke``
   returns with an ``__interrupt__`` entry carrying the payload.
2. The checkpointer (MemorySaver) has saved the whole state under the
   ``thread_id``; ``get_state(config).next`` shows the paused node.
3. ``invoke(Command(resume=decision), config)`` re-enters human_review;
   ``interrupt`` returns ``decision`` and the graph runs to END.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agents.supervisor_v2 import RouterV2
from src.graph_v2 import ascii_graph_v2, build_graph_v2
from src.state_v2 import WORKERS_V2


class _ScriptedStructuredRunnable:
    """Stand-in for ``llm.with_structured_output(RouterV2)`` — replays a script."""

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self._calls = 0

    def invoke(self, _messages: Any, config: Any = None) -> RouterV2:
        route = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return RouterV2(next=route, reason=f"scripted decision #{self._calls}: {route}")


class FakeRouterV2LLM:
    """Fake chat model exposing only what the v2 supervisor node actually uses."""

    def __init__(self, script: list[str]) -> None:
        self._script = script

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _ScriptedStructuredRunnable:
        assert schema is RouterV2, "v2 supervisor must request the RouterV2 schema"
        return _ScriptedStructuredRunnable(self._script)


def _fake_workers() -> dict[str, Callable[[dict], dict]]:
    """Plain-callable worker nodes, mimicking make_worker_node's output shape."""

    def make(name: str) -> Callable[[dict], dict]:
        def node(state: dict) -> dict:
            return {"messages": [AIMessage(content=f"{name} report: work done.", name=name)]}

        return node

    return {name: make(name) for name in WORKERS_V2}


def _scripted_validator(verdict: str) -> Callable[[dict], dict]:
    def node(state: dict) -> dict:
        return {"verdict": verdict, "verdict_reason": f"scripted: {verdict}"}

    return node


def _build(script: list[str], verdict: str) -> tuple[Any, dict]:
    app = build_graph_v2(
        llm=FakeRouterV2LLM(script),
        workers=_fake_workers(),
        validator=_scripted_validator(verdict),
        checkpointer=MemorySaver(),
    )
    config = {"recursion_limit": 25, "configurable": {"thread_id": "hitl-test"}}
    return app, config


def _invoke(app: Any, config: dict, question: str) -> dict:
    return app.invoke(
        {"messages": [HumanMessage(content=question)], "next_agent": "", "verdict": "", "verdict_reason": ""},
        config=config,
    )


# ---- Auto-complete path: validator says complete -> END, no pause -------------

def test_complete_verdict_runs_to_end_without_interrupt() -> None:
    app, config = _build(["sql_analyst", "FINISH"], verdict="complete")
    result = _invoke(app, config, "How many orders per carrier?")

    assert "__interrupt__" not in result
    assert result["verdict"] == "complete"
    assert app.get_state(config).next == ()  # nothing pending — the run is over


# ---- Human-review path: needs_human -> interrupt -> resume approve ------------

def test_needs_human_pauses_at_human_review_then_resume_approve() -> None:
    app, config = _build(["sql_analyst", "FINISH"], verdict="needs_human")
    result = _invoke(app, config, "How many orders per carrier?")

    assert "__interrupt__" in result, "graph must pause for human review"
    payload = result["__interrupt__"][0].value
    assert "draft_answer" in payload and "validator_reason" in payload
    assert "human_review" in app.get_state(config).next  # paused at the right node

    resumed = app.invoke(Command(resume={"action": "approve"}), config=config)
    assert resumed["verdict"] == "complete"
    assert "approved" in resumed["verdict_reason"]
    assert app.get_state(config).next == ()


# ---- Human-review path: resume edit replaces the answer ------------------------

def test_resume_edit_appends_human_reviewer_message() -> None:
    app, config = _build(["sql_analyst", "FINISH"], verdict="needs_human")
    _invoke(app, config, "How many orders per carrier?")

    resumed = app.invoke(
        Command(resume={"action": "edit", "text": "Corrected by a human: 2,841 orders."}),
        config=config,
    )
    final = resumed["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.name == "human_reviewer"
    assert "Corrected by a human" in final.content
    assert resumed["verdict"] == "complete"


# ---- Routing: the 5th worker is reachable --------------------------------------

def test_routes_through_logistics_coordinator() -> None:
    app, config = _build(["logistics_coordinator", "FINISH"], verdict="complete")
    result = _invoke(app, config, "Any open incidents for SwiftShip right now?")

    visited = [m.name for m in result["messages"] if isinstance(m, AIMessage) and m.name]
    assert visited == ["logistics_coordinator"]


# ---- Topology rendering ---------------------------------------------------------

def test_ascii_graph_v2_mentions_validator_and_human_review() -> None:
    art = ascii_graph_v2()
    assert "validator" in art
    assert "human_review" in art
    assert "logistics_coordinator" in art
