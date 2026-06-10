"""WHY THIS EXISTS
---------------
Proof that the multi-agent graph's *wiring* is correct without any API key:
supervisor -> worker -> supervisor -> FINISH, on the real compiled graph.

The trick is dependency injection (ARCHITECTURE.md §4.7 + §6): ``build_graph``
accepts the supervisor's llm and the worker nodes as parameters, so we pass

- ``FakeRouterLLM`` — its ``with_structured_output(Router)`` returns a fake
  runnable that replays a *scripted* sequence of ``Router`` decisions. The
  graph still runs the genuine supervisor node code (system prompt assembly,
  state update); only the model call is scripted.
- plain functions as workers — each appends ``AIMessage(name=<worker>)``
  exactly like a real wrapped ReAct agent would.

If a test here fails, the bug is in graph topology or routing logic, not in
any LLM behavior — that separation is what makes multi-agent systems
debuggable in production.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import Router, make_worker_node
from src.graph import ascii_graph, build_graph
from src.state import WORKERS


class _ScriptedStructuredRunnable:
    """Stand-in for ``llm.with_structured_output(Router)`` — replays a script."""

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self._calls = 0

    def invoke(self, _messages: Any, config: Any = None) -> Router:
        route = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return Router(next=route, reason=f"scripted decision #{self._calls}: {route}")


class FakeRouterLLM:
    """Fake chat model exposing only what the supervisor node actually uses."""

    def __init__(self, script: list[str]) -> None:
        self._script = script

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _ScriptedStructuredRunnable:
        assert schema is Router, "supervisor must request the Router schema"
        return _ScriptedStructuredRunnable(self._script)


def _fake_workers() -> dict[str, Callable[[dict], dict]]:
    """Plain-callable worker nodes, mimicking make_worker_node's output shape."""

    def make(name: str) -> Callable[[dict], dict]:
        def node(state: dict) -> dict:
            return {"messages": [AIMessage(content=f"{name} report: work done.", name=name)]}

        return node

    return {name: make(name) for name in WORKERS}


def _invoke(app: Any, question: str) -> dict:
    return app.invoke(
        {"messages": [HumanMessage(content=question)], "next_agent": ""},
        config={"recursion_limit": 25},
    )


def test_graph_compiles_offline() -> None:
    app = build_graph(llm=FakeRouterLLM(["FINISH"]), workers=_fake_workers())
    assert app is not None
    assert hasattr(app, "invoke") and hasattr(app, "stream")


def test_routes_through_sql_analyst_then_finishes() -> None:
    app = build_graph(
        llm=FakeRouterLLM(["sql_analyst", "FINISH"]), workers=_fake_workers()
    )
    result = _invoke(app, "How many orders did each carrier handle?")

    worker_messages = [
        m
        for m in result["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", None) == "sql_analyst"
    ]
    assert worker_messages, "expected an AIMessage tagged name='sql_analyst'"
    assert "sql_analyst report" in worker_messages[0].content
    assert result["next_agent"] == "FINISH"


def test_multi_hop_routing_visits_workers_in_order() -> None:
    script = ["contracts_analyst", "ml_engineer", "FINISH"]
    app = build_graph(llm=FakeRouterLLM(script), workers=_fake_workers())
    result = _invoke(app, "Penalty if order 104872 is late, and how likely is that?")

    visited = [
        m.name
        for m in result["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", None)
    ]
    assert visited == ["contracts_analyst", "ml_engineer"]
    assert result["next_agent"] == "FINISH"


def test_direct_finish_skips_all_workers() -> None:
    app = build_graph(llm=FakeRouterLLM(["FINISH"]), workers=_fake_workers())
    result = _invoke(app, "Thanks, that's all!")

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_messages == [], "no worker should have run on a direct FINISH"
    assert result["next_agent"] == "FINISH"


def test_make_worker_node_wraps_agent_output() -> None:
    class FakeReActAgent:
        def invoke(self, payload: dict) -> dict:
            return {"messages": list(payload["messages"]) + [AIMessage(content="42 rows.")]}

    node = make_worker_node("sql_analyst", FakeReActAgent())
    out = node({"messages": [HumanMessage(content="count orders")], "next_agent": ""})

    assert len(out["messages"]) == 1
    message = out["messages"][0]
    assert isinstance(message, AIMessage)
    assert message.name == "sql_analyst"
    assert message.content == "42 rows."


def test_ascii_graph_mentions_supervisor() -> None:
    art = ascii_graph()
    assert isinstance(art, str)
    assert art.strip(), "ascii_graph() must return a non-empty rendering"
    assert "supervisor" in art
