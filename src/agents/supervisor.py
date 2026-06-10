"""WHY THIS EXISTS
---------------
The supervisor pattern is THE core LangGraph lesson of this course: instead of
one giant agent, a lightweight *router* LLM decides which specialist worker
acts next — or that the conversation is done (FINISH).

Two ideas make it reliable:

1. **Structured output, not prose.** The supervisor must produce a routing
   decision a graph edge can act on. ``llm.with_structured_output(Router)``
   forces the model to return a validated Pydantic object whose ``next`` field
   is constrained to a ``Literal`` of legal routes — the model literally cannot
   answer "hmm, maybe the SQL person?".

2. **Factories, not globals.** ``make_supervisor_node(llm)`` and
   ``make_worker_node(name, agent)`` return plain ``state -> dict`` functions.
   The graph never knows about LLMs; tests inject a fake llm whose
   ``with_structured_output`` yields scripted ``Router`` objects, so the whole
   routing flow is testable offline (ARCHITECTURE.md §5-6).

Worker nodes wrap each ReAct agent's final answer as ``AIMessage(name=worker)``
— the ``name`` tag is how the supervisor (and the demo CLI) can tell which
specialist already answered what.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, get_args

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_llm
from src.state import ROUTE_OPTIONS, WORKER_DESCRIPTIONS, SupplyChainState

# Literal must be spelled with literal strings for pydantic/type-checkers; the
# guard below keeps it in lock-step with src.state.ROUTE_OPTIONS.
RouteName = Literal["data_analyst", "ml_engineer", "contracts_analyst", "sql_analyst", "FINISH"]

if set(get_args(RouteName)) != set(ROUTE_OPTIONS):  # pragma: no cover
    raise RuntimeError(
        "supervisor.RouteName is out of sync with src.state.ROUTE_OPTIONS — "
        "update the Literal to match."
    )


class Router(BaseModel):
    """The supervisor's routing decision — the structured-output schema."""

    next: RouteName = Field(
        description="The single worker to act next, or FINISH when the user's "
        "question is fully answered by the conversation so far."
    )
    reason: str = Field(description="One short sentence justifying the choice.")


def _supervisor_system_prompt() -> str:
    """Format WORKER_DESCRIPTIONS into the routing instructions."""
    workers_block = "\n".join(
        f"- {name}: {description}" for name, description in WORKER_DESCRIPTIONS.items()
    )
    return (
        "You are the supervisor of a supply-chain analytics team. Given the "
        "conversation so far, route to exactly ONE of these workers, or FINISH:\n"
        f"{workers_block}\n\n"
        "Rules:\n"
        "- Route to ONE worker per turn; complex questions may need several "
        "workers in sequence across turns.\n"
        "- Choose FINISH when the user's question is fully answered by prior AI "
        "messages in the conversation.\n"
        "- Prefer FINISH over repeating work a worker has already done — never "
        "send the same sub-question to the same worker twice.\n"
    )


def _coerce_text(content: Any) -> str:
    """Flatten message content (str or provider content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def make_supervisor_node(
    llm: BaseChatModel | None = None,
) -> Callable[[SupplyChainState], dict]:
    """Build the supervisor node function (dependency-injectable for tests).

    The returned node reads the full message history, asks the (structured-
    output) LLM for a ``Router`` decision, and writes it to ``next_agent`` —
    the conditional edge in ``src/graph.py`` does the actual branching.
    """
    if llm is None:
        llm = get_llm()
    router = llm.with_structured_output(Router)
    system_prompt = _supervisor_system_prompt()

    def supervisor_node(state: SupplyChainState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        decision: Router = router.invoke(messages)
        return {"next_agent": decision.next}

    return supervisor_node


# Lazily-built default supervisor (uses get_llm()); see supervisor_node below.
_default_supervisor: Callable[[SupplyChainState], dict] | None = None


def supervisor_node(state: SupplyChainState) -> dict:
    """ARCHITECTURE.md §4.7 entry point: route ``state`` with the default LLM.

    Thin wrapper over ``make_supervisor_node()`` — the factory is what
    ``src/graph.py`` and the tests use (dependency injection), but this
    module-level function is the contract's plug-and-play form: it builds the
    default ``get_llm()`` supervisor on *first call* (never at import time, so
    ``import src.agents.supervisor`` stays safe offline) and caches it.
    """
    global _default_supervisor
    if _default_supervisor is None:
        _default_supervisor = make_supervisor_node()
    return _default_supervisor(state)


def make_worker_node(name: str, agent: Any) -> Callable[[SupplyChainState], dict]:
    """Wrap a compiled ReAct agent as a graph node named ``name``.

    The agent runs its full tool-calling loop internally; only its *final*
    answer is appended to the shared state, tagged with the worker's name so
    the supervisor can see who has already answered. Routing back to the
    supervisor is a fixed edge in the graph, not this function's job.
    """

    def worker_node(state: SupplyChainState) -> dict:
        result = agent.invoke({"messages": list(state["messages"])})
        last = result["messages"][-1]
        content = _coerce_text(getattr(last, "content", last))
        return {"messages": [AIMessage(content=content, name=name)]}

    return worker_node
