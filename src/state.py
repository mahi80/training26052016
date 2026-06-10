"""Shared LangGraph state schema for the supervisor multi-agent system.

WHY THIS EXISTS
---------------
LangGraph graphs pass a single *state* object between nodes. Defining it in one place
(and importing it everywhere) is the contract that lets four independently-built
worker agents plug into the same supervisor. This is the first thing to read when
learning the codebase: the whole system is "a state machine over this dict".

- ``messages`` uses the ``add_messages`` reducer: each node *appends* messages instead
  of overwriting, giving every agent the full conversation context.
- ``next_agent`` carries the supervisor's routing decision to the conditional edge.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Worker agent names — node names in the graph AND routing labels for the supervisor.
WORKERS: tuple[str, ...] = (
    "data_analyst",
    "ml_engineer",
    "contracts_analyst",
    "sql_analyst",
)

# The supervisor chooses one of these each turn; FINISH ends the conversation turn.
ROUTE_OPTIONS: tuple[str, ...] = WORKERS + ("FINISH",)

WORKER_DESCRIPTIONS: dict[str, str] = {
    "data_analyst": (
        "Explores the historical orders dataset: late-delivery rates by shipping "
        "mode/region/category, monthly trends, class balance, summary statistics."
    ),
    "ml_engineer": (
        "Trains and evaluates late-delivery prediction models, tunes the decision "
        "threshold for recall, explains feature drivers, and scores individual "
        "orders for late-delivery risk."
    ),
    "contracts_analyst": (
        "Answers questions about carrier and supplier procurement contracts: "
        "late-delivery penalties, on-time SLAs, payment terms, termination clauses."
    ),
    "sql_analyst": (
        "Translates business questions into SQL against the supply-chain warehouse "
        "(orders, customers, products, carriers) using the metadata catalog."
    ),
}


class SupplyChainState(TypedDict):
    """The single state object every node in the graph reads and writes."""

    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
