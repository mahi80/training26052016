"""Shared state schema for the v2 (production-layers) graph.

WHY THIS EXISTS
---------------
The v1 stack (``src/state.py``, ``src/graph.py``) is the course's binding
contract and stays frozen. BUILD_MANUAL.md Part II *extends* it instead:

- a 5th worker, ``logistics_coordinator``, whose tools reach SAP/ServiceNow
  over MCP and the external carrier agent over A2A;
- two new state fields, ``verdict`` and ``verdict_reason``, written by the
  validator/judge node so the graph can branch to auto-complete or human
  review.

``SupplyChainStateV2`` *inherits* the v1 TypedDict — every v1 node keeps
working unchanged on the v2 state. Additive evolution is the lesson: extend
the state, never rewrite running agents.
"""

from __future__ import annotations

from src.state import WORKER_DESCRIPTIONS, WORKERS, SupplyChainState

# v1's four specialists + the new MCP/A2A-powered coordinator.
WORKERS_V2: tuple[str, ...] = WORKERS + ("logistics_coordinator",)

ROUTE_OPTIONS_V2: tuple[str, ...] = WORKERS_V2 + ("FINISH",)

WORKER_DESCRIPTIONS_V2: dict[str, str] = {
    **WORKER_DESCRIPTIONS,
    "logistics_coordinator": (
        "Checks LIVE operational systems: SAP purchase orders per vendor, "
        "open ServiceNow incident tickets per carrier, and the carrier's own "
        "partner agent for current fleet status, pickup capacity, and hub "
        "congestion. Use for 'right now' questions the historical data, "
        "contracts, and warehouse cannot answer."
    ),
}


class SupplyChainStateV2(SupplyChainState):
    """v1 state + the validator's verdict (drives the post-FINISH branch)."""

    verdict: str  # "" until the validator runs, then "complete" | "needs_human"
    verdict_reason: str
