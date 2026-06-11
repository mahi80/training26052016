"""WHY THIS EXISTS
---------------
The v2 supervisor: identical mechanics to ``src/agents/supervisor.py`` (read
that docstring first — structured output + factories), with exactly one
change: the routing ``Literal`` now includes ``logistics_coordinator``.

That this file is a near-copy is deliberate teaching material: adding a
worker to a supervisor system means (1) extend the route Literal, (2) add the
worker's description to the system prompt, (3) register the node in the
graph. Nothing else. The lock-step guard below fails loudly at import time
if the Literal and ``ROUTE_OPTIONS_V2`` ever drift apart.
"""

from __future__ import annotations

from typing import Callable, Literal, get_args

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.config import get_llm
from src.state_v2 import ROUTE_OPTIONS_V2, WORKER_DESCRIPTIONS_V2, SupplyChainStateV2

RouteNameV2 = Literal[
    "data_analyst",
    "ml_engineer",
    "contracts_analyst",
    "sql_analyst",
    "logistics_coordinator",
    "FINISH",
]

if set(get_args(RouteNameV2)) != set(ROUTE_OPTIONS_V2):  # pragma: no cover
    raise RuntimeError(
        "supervisor_v2.RouteNameV2 is out of sync with src.state_v2.ROUTE_OPTIONS_V2 — "
        "update the Literal to match."
    )


class RouterV2(BaseModel):
    """The v2 supervisor's routing decision — the structured-output schema."""

    next: RouteNameV2 = Field(
        description="The single worker to act next, or FINISH when the user's "
        "question is fully answered by the conversation so far."
    )
    reason: str = Field(description="One short sentence justifying the choice.")


def _supervisor_system_prompt() -> str:
    """Format WORKER_DESCRIPTIONS_V2 into the routing instructions."""
    workers_block = "\n".join(
        f"- {name}: {description}" for name, description in WORKER_DESCRIPTIONS_V2.items()
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
        "- Questions about LIVE operations (open POs, current incidents, "
        "today's carrier capacity) go to logistics_coordinator; historical "
        "analysis stays with the other specialists.\n"
    )


def make_supervisor_node_v2(
    llm: BaseChatModel | None = None,
) -> Callable[[SupplyChainStateV2], dict]:
    """Build the v2 supervisor node function (dependency-injectable for tests)."""
    if llm is None:
        llm = get_llm()
    router = llm.with_structured_output(RouterV2)
    system_prompt = _supervisor_system_prompt()

    def supervisor_node(state: SupplyChainStateV2) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        decision: RouterV2 = router.invoke(messages)
        return {"next_agent": decision.next}

    return supervisor_node
