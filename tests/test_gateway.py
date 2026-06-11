"""WHY THIS EXISTS
---------------
Proof the HTTP front door speaks the contract, offline. The gateway takes a
``graph_factory`` (dependency injection, same idea as ``build_graph_v2``), so
these tests mount the REAL v2 graph — fake router, fake workers, scripted
validator, MemorySaver — behind the REAL FastAPI routes via TestClient. The
full ask -> needs_human -> resume cycle runs over HTTP with no key and no port.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.supervisor_v2 import RouterV2
from src.gateway import create_app
from src.graph_v2 import build_graph_v2
from src.state_v2 import WORKERS_V2


class _ScriptedStructuredRunnable:
    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self._calls = 0

    def invoke(self, _messages: Any, config: Any = None) -> RouterV2:
        route = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return RouterV2(next=route, reason=f"scripted decision #{self._calls}: {route}")


class FakeRouterV2LLM:
    def __init__(self, script: list[str]) -> None:
        self._script = script

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _ScriptedStructuredRunnable:
        assert schema is RouterV2
        return _ScriptedStructuredRunnable(self._script)


def _fake_workers() -> dict[str, Callable[[dict], dict]]:
    def make(name: str) -> Callable[[dict], dict]:
        def node(state: dict) -> dict:
            return {"messages": [AIMessage(content=f"{name} report: work done.", name=name)]}

        return node

    return {name: make(name) for name in WORKERS_V2}


def _client(script: list[str], verdict: str) -> TestClient:
    """Gateway over the real v2 graph with scripted routing and verdict."""

    def factory() -> Any:
        def validator(state: dict) -> dict:
            return {"verdict": verdict, "verdict_reason": f"scripted: {verdict}"}

        return build_graph_v2(
            llm=FakeRouterV2LLM(script),
            workers=_fake_workers(),
            validator=validator,
            checkpointer=MemorySaver(),
        )

    return TestClient(create_app(graph_factory=factory))


# ---- /health: cheap, never builds the graph -----------------------------------

def test_health_does_not_build_the_graph() -> None:
    def exploding_factory() -> Any:
        raise AssertionError("/health must not trigger a graph build")

    client = TestClient(create_app(graph_factory=exploding_factory))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] in ("live", "offline-demo")


# ---- /ask: auto-complete path ---------------------------------------------------

def test_ask_complete_returns_answer_and_hops() -> None:
    client = _client(["sql_analyst", "FINISH"], verdict="complete")
    body = client.post("/ask", json={"question": "How many orders per carrier?"}).json()

    assert body["status"] == "complete"
    assert body["hops"] == ["sql_analyst"]
    assert "sql_analyst report" in body["answer"]
    assert body["thread_id"]  # auto-generated


# ---- /ask -> needs_human -> /resume: the full HITL cycle over HTTP ---------------

def test_ask_needs_human_then_resume_approve() -> None:
    client = _client(["sql_analyst", "FINISH"], verdict="needs_human")
    asked = client.post("/ask", json={"question": "How many orders per carrier?"}).json()

    assert asked["status"] == "needs_human"
    assert "draft_answer" in asked["review"]
    thread_id = asked["thread_id"]

    state = client.get(f"/threads/{thread_id}/state").json()
    assert state["pending"] is True
    assert "human_review" in state["paused_at"]

    resumed = client.post("/resume", json={"thread_id": thread_id, "action": "approve"}).json()
    assert resumed["status"] == "complete"

    state_after = client.get(f"/threads/{thread_id}/state").json()
    assert state_after["pending"] is False


def test_resume_edit_replaces_answer() -> None:
    client = _client(["sql_analyst", "FINISH"], verdict="needs_human")
    thread_id = client.post("/ask", json={"question": "Count orders."}).json()["thread_id"]

    resumed = client.post(
        "/resume",
        json={"thread_id": thread_id, "action": "edit", "text": "Human-corrected: 2,841 orders."},
    ).json()
    assert resumed["status"] == "complete"
    assert resumed["answer"] == "Human-corrected: 2,841 orders."
    assert resumed["hops"][-1] == "human_reviewer"


# ---- Error contract ---------------------------------------------------------------

def test_resume_unknown_thread_404() -> None:
    client = _client(["FINISH"], verdict="complete")
    response = client.post("/resume", json={"thread_id": "nope", "action": "approve"})
    assert response.status_code == 404


def test_resume_thread_that_is_not_paused_409() -> None:
    client = _client(["sql_analyst", "FINISH"], verdict="complete")
    thread_id = client.post("/ask", json={"question": "Count orders."}).json()["thread_id"]
    response = client.post("/resume", json={"thread_id": thread_id, "action": "approve"})
    assert response.status_code == 409


def test_thread_state_unknown_thread_404() -> None:
    client = _client(["FINISH"], verdict="complete")
    assert client.get("/threads/ghost/state").status_code == 404


def test_ask_empty_question_422() -> None:
    client = _client(["FINISH"], verdict="complete")
    assert client.post("/ask", json={"question": ""}).status_code == 422
