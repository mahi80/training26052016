"""WHY THIS EXISTS
---------------
The **API Gateway** from the architecture diagram — the front door. Users
(or any frontend) speak plain HTTP+JSON; the gateway owns the LangGraph
runtime behind it: builds the graph once, manages ``thread_id``s, and
translates interrupts into a REST-shaped conversation:

    POST /ask     {"question": "..."}            -> complete | needs_human
    POST /resume  {"thread_id": "...", "action": "approve" | "edit"}
    GET  /threads/{thread_id}/state              -> where is my run?
    GET  /health                                 -> is the service up?

Run it:   python -m uvicorn src.gateway:app --port 8000
Try it:   open http://127.0.0.1:8000/docs (Swagger UI — the built-in frontend)

DESIGN NOTES (load-bearing, do not "fix" these):

- **Handlers are sync ``def``, not ``async def``.** FastAPI runs sync
  handlers in a threadpool. The MCP tools underneath call ``asyncio.run()``,
  which would crash inside an async handler's already-running event loop.
- **The graph builds lazily on the first request**, never at import time —
  so ``uvicorn src.gateway:app`` starts and ``/health`` answers even with no
  API key. Offline, the gateway serves ``build_demo_graph_v2`` (real tools,
  keyword routing) and tags every response ``"mode": "offline-demo"``.
- **MemorySaver means threads live in process memory**: restart the server,
  lose the threads. The production upgrade (SqliteSaver/PostgresSaver) is one
  line — see CHALLENGES_GUIDE.md §3.10.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import is_offline

_RECURSION_LIMIT = 25


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The question for the agent team.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread to continue; omit to start a new one.",
    )


class ResumeRequest(BaseModel):
    thread_id: str = Field(description="The thread that is paused at human review.")
    action: Literal["approve", "edit"] = "approve"
    text: str = Field(default="", description="Replacement answer when action='edit'.")


def _default_graph_factory() -> Any:
    """Live graph with a key, keyword-routed demo graph without one."""
    from langgraph.checkpoint.memory import MemorySaver

    if is_offline():
        from src.graph_v2 import build_demo_graph_v2

        return build_demo_graph_v2(checkpointer=MemorySaver())
    from src.graph_v2 import build_graph_v2

    return build_graph_v2(checkpointer=MemorySaver())


def _answer_and_hops(state_values: dict) -> tuple[str, list[str]]:
    """Extract the final answer text and the visited-worker audit trail."""
    from langchain_core.messages import AIMessage

    answers = [m for m in state_values.get("messages", []) if isinstance(m, AIMessage)]
    hops = [m.name for m in answers if getattr(m, "name", None)]
    return (str(answers[-1].content) if answers else ""), hops


def create_app(graph_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Build the gateway app. ``graph_factory`` is injectable for tests."""
    factory = graph_factory or _default_graph_factory
    mode = "offline-demo" if is_offline() else "live"
    cache: dict[str, Any] = {}
    build_lock = threading.Lock()

    def get_graph() -> Any:
        with build_lock:  # threadpool-safe single build
            if "graph" not in cache:
                cache["graph"] = factory()
            return cache["graph"]

    def thread_config(thread_id: str) -> dict:
        return {"recursion_limit": _RECURSION_LIMIT, "configurable": {"thread_id": thread_id}}

    def result_payload(thread_id: str, result: dict) -> dict:
        if "__interrupt__" in result:
            return {
                "status": "needs_human",
                "thread_id": thread_id,
                "mode": mode,
                "review": result["__interrupt__"][0].value,
                "how_to_resume": "POST /resume with this thread_id and action approve|edit",
            }
        answer, hops = _answer_and_hops(result)
        return {
            "status": "complete",
            "thread_id": thread_id,
            "mode": mode,
            "answer": answer,
            "hops": hops,
            "verdict_reason": result.get("verdict_reason", ""),
        }

    app = FastAPI(
        title="Supply Chain Ops Copilot — API Gateway",
        description="HTTP front door for the LangGraph v2 multi-agent system. "
        "Use /docs to try it interactively.",
    )

    @app.get("/health")
    def health() -> dict:
        # Deliberately does NOT build the graph: liveness must be cheap and
        # must work before any API key exists.
        return {"status": "ok", "offline": is_offline(), "mode": mode}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        from langchain_core.messages import HumanMessage

        thread_id = request.thread_id or uuid.uuid4().hex[:8]
        graph = get_graph()
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request.question)],
                "next_agent": "",
                "verdict": "",
                "verdict_reason": "",
            },
            config=thread_config(thread_id),
        )
        return result_payload(thread_id, result)

    @app.post("/resume")
    def resume(request: ResumeRequest) -> dict:
        from langgraph.types import Command

        graph = get_graph()
        config = thread_config(request.thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise HTTPException(404, f"Unknown thread_id {request.thread_id!r}.")
        if not snapshot.next:
            raise HTTPException(409, f"Thread {request.thread_id!r} is not paused — nothing to resume.")
        result = graph.invoke(
            Command(resume={"action": request.action, "text": request.text}),
            config=config,
        )
        return result_payload(request.thread_id, result)

    @app.get("/threads/{thread_id}/state")
    def thread_state(thread_id: str) -> dict:
        graph = get_graph()
        snapshot = graph.get_state(thread_config(thread_id))
        if not snapshot.values:
            raise HTTPException(404, f"Unknown thread_id {thread_id!r}.")
        answer, hops = _answer_and_hops(snapshot.values)
        return {
            "thread_id": thread_id,
            "mode": mode,
            "pending": bool(snapshot.next),  # True while paused at human_review
            "paused_at": list(snapshot.next),
            "message_count": len(snapshot.values.get("messages", [])),
            "last_answer": answer,
            "hops": hops,
            "next_agent": snapshot.values.get("next_agent", ""),
            "verdict": snapshot.values.get("verdict", ""),
        }

    return app


# `python -m uvicorn src.gateway:app --port 8000` imports this module-level app.
app = create_app()
