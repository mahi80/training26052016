"""WHY THIS EXISTS
---------------
The v2 assembly point — the full architecture-diagram machine. v1's
hub-and-spoke (``src/graph.py``) plus the production layers:

    START -> supervisor -(next_agent)-> worker -> supervisor -> ...
             supervisor -(FINISH)-> validator -(complete)----> END
                                    validator -(needs_human)-> human_review -> END

Three additions over v1, each one diagram box:

- **5th worker** ``logistics_coordinator`` — reaches SAP/ServiceNow over MCP
  and the external carrier agent over A2A.
- **validator** — judges the draft answer after FINISH (``src/agents/validator``).
- **human_review** — ``interrupt()`` pauses the graph mid-run; the checkpointer
  saves everything; ``graph.invoke(Command(resume=...), config)`` with the same
  ``thread_id`` continues exactly where it stopped. A pause button, not a poll.

Same dependency injection as v1: tests pass fake llms / plain-callable nodes
and exercise the *exact same wiring* offline. ``build_demo_graph_v2`` goes one
further — a keyword router and tool-calling workers with **no LLM at all** —
so the whole topology (MCP calls included) runs end-to-end without a key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.supervisor import make_worker_node
from src.agents.supervisor_v2 import make_supervisor_node_v2
from src.agents.validator import make_validator_node
from src.state_v2 import WORKERS_V2, SupplyChainStateV2

if TYPE_CHECKING:  # import only for type hints — keeps runtime deps minimal
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

WorkerNode = Callable[[SupplyChainStateV2], dict]


def make_human_review_node() -> WorkerNode:
    """Build the human-review node: pause the graph, apply the human's decision.

    ``interrupt(payload)`` stops execution and surfaces ``payload`` to whoever
    drives the graph (CLI, gateway, lab). On resume, ``interrupt`` *returns*
    the value passed in ``Command(resume=...)``:

    - ``{"action": "approve"}``            -> draft answer ships as-is
    - ``{"action": "edit", "text": "..."}`` -> the human's text is appended as
      the final answer, tagged ``name='human_reviewer'`` for the audit trail
    """

    def human_review_node(state: SupplyChainStateV2) -> dict:
        messages = state.get("messages", [])
        answers = [m for m in messages if isinstance(m, AIMessage)]
        decision = interrupt(
            {
                "question": str(messages[0].content) if messages else "",
                "draft_answer": str(answers[-1].content) if answers else "",
                "validator_reason": state.get("verdict_reason", ""),
                "options": "resume with {'action': 'approve'} or {'action': 'edit', 'text': '...'}",
            }
        )
        action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"
        if action == "edit":
            return {
                "verdict": "complete",
                "verdict_reason": "Human reviewer replaced the draft answer.",
                "messages": [AIMessage(content=str(decision.get("text", "")), name="human_reviewer")],
            }
        return {"verdict": "complete", "verdict_reason": "Human reviewer approved the draft answer."}

    return human_review_node


def _assemble_v2(
    supervisor_node: WorkerNode,
    workers: dict[str, WorkerNode],
    validator_node: WorkerNode,
    checkpointer: Any = None,
) -> "CompiledStateGraph":
    """Wire supervisor + workers + validator + human_review into one graph."""
    builder = StateGraph(SupplyChainStateV2)
    builder.add_node("supervisor", supervisor_node)
    for name, node in workers.items():
        builder.add_node(name, node)
        builder.add_edge(name, "supervisor")  # every worker reports back
    builder.add_node("validator", validator_node)
    builder.add_node("human_review", make_human_review_node())
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        # v1 sent FINISH straight to END; v2 inserts the quality gate.
        {name: name for name in workers} | {"FINISH": "validator"},
    )
    builder.add_conditional_edges(
        "validator",
        lambda state: state["verdict"],
        {"complete": END, "needs_human": "human_review"},
    )
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)


def build_graph_v2(
    llm: "BaseChatModel | None" = None,
    workers: dict[str, WorkerNode] | None = None,
    validator: WorkerNode | None = None,
    checkpointer: Any = None,
) -> "CompiledStateGraph":
    """Build and compile the full v2 graph (supervisor + 5 workers + gate).

    Args mirror v1's ``build_graph``: every expensive part is injectable.
    ``None`` everywhere -> real ReAct agents + LLM judge on ``get_llm()``
    (requires an API key). Tests inject fakes and run the wiring offline.
    """
    if workers is None:
        # Imported lazily: building real agents pulls in every tool module.
        from src.agents import workers as worker_factories

        factories: dict[str, Callable[..., Any]] = {
            "data_analyst": worker_factories.make_data_analyst,
            "ml_engineer": worker_factories.make_ml_engineer,
            "contracts_analyst": worker_factories.make_contracts_analyst,
            "sql_analyst": worker_factories.make_sql_analyst,
            "logistics_coordinator": worker_factories.make_logistics_coordinator,
        }
        workers = {
            name: make_worker_node(name, factory(llm)) for name, factory in factories.items()
        }
    if validator is None:
        validator = make_validator_node(llm)
    return _assemble_v2(make_supervisor_node_v2(llm), workers, validator, checkpointer)


# --------------------------------------------------------------------------
# Offline demo graph: the same topology with zero LLM calls.
# --------------------------------------------------------------------------

_DEMO_ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("incident", "ticket", "servicenow", "outage"), "live_incidents"),
    (("purchase order", "po ", "sap", "vendor", "delayed order"), "live_purchase_orders"),
    (("pickup", "fleet", "hub", "capacity", "congestion", "partner agent"), "carrier_partner"),
    (("contract", "penalty", "sla", "clause", "termination"), "contracts_lookup"),
    (("order", "carrier", "warehouse", "count", "how many", "late"), "warehouse_overview"),
]

_DEMO_HELP = (
    "Offline demo: no API key, so questions route by keyword. Try asking about "
    "incidents/tickets, purchase orders/vendors, pickup capacity or hub "
    "congestion, contracts/penalties, or order counts per carrier."
)


def _demo_supervisor(state: SupplyChainStateV2) -> dict:
    """Keyword router: pick one demo worker for the question, then FINISH."""
    messages = state.get("messages", [])
    if any(isinstance(m, AIMessage) for m in messages):
        return {"next_agent": "FINISH"}  # a worker already answered
    question = str(messages[0].content).lower() if messages else ""
    for keywords, worker in _DEMO_ROUTES:
        if any(k in question for k in keywords):
            return {"next_agent": worker}
    return {"next_agent": "demo_help"}


def _make_demo_worker(name: str, run: Callable[[str], str]) -> WorkerNode:
    """Wrap a plain question->answer function as a worker node (no LLM)."""

    def node(state: SupplyChainStateV2) -> dict:
        question = str(state["messages"][0].content) if state.get("messages") else ""
        return {"messages": [AIMessage(content=run(question), name=name)]}

    return node


def _demo_workers() -> dict[str, WorkerNode]:
    """Tool-backed demo workers: real MCP/A2A/RAG/SQL calls, scripted prompts."""
    from src.agents.tools_a2a import ask_carrier_agent
    from src.agents.tools_mcp import sap_purchase_orders, servicenow_incidents
    from src.agents.tools_rag import search_contracts
    from src.agents.tools_sql import run_sql_query

    return {
        "live_incidents": _make_demo_worker(
            "live_incidents",
            lambda q: "Open tickets (ServiceNow via MCP):\n"
            + servicenow_incidents.invoke({"carrier": "SwiftShip" if "swiftship" in q.lower() else ""}),
        ),
        "live_purchase_orders": _make_demo_worker(
            "live_purchase_orders",
            lambda q: "Purchase orders (SAP via MCP):\n"
            + sap_purchase_orders.invoke({"vendor": "SwiftShip" if "swiftship" in q.lower() else ""}),
        ),
        "carrier_partner": _make_demo_worker(
            "carrier_partner",
            lambda q: "Carrier's own agent says (A2A):\n" + ask_carrier_agent.invoke({"question": q}),
        ),
        "contracts_lookup": _make_demo_worker(
            "contracts_lookup",
            lambda q: "Matching contract sections (PageIndex):\n" + search_contracts.invoke({"query": q}),
        ),
        "warehouse_overview": _make_demo_worker(
            "warehouse_overview",
            lambda q: "Warehouse overview (SQL):\n"
            + run_sql_query.invoke(
                {
                    "sql": "SELECT c.carrier_name, COUNT(*) AS n_orders, "
                    "ROUND(AVG(o.is_late) * 100, 1) AS late_rate_pct "
                    "FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id "
                    "GROUP BY c.carrier_name ORDER BY late_rate_pct DESC"
                }
            ),
        ),
        "demo_help": _make_demo_worker("demo_help", lambda q: _DEMO_HELP),
    }


def build_demo_graph_v2(checkpointer: Any = None) -> "CompiledStateGraph":
    """The v2 topology with zero LLM calls — for offline demos and the gateway.

    Keyword router instead of the supervisor LLM, tool-backed workers instead
    of ReAct agents, heuristic validator. Real MCP subprocesses, real A2A
    HTTP, real SQL guardrails, real interrupt/resume — only the *reasoning*
    is canned. ``python main.py --offline-check`` quality, graph edition.
    """
    return _assemble_v2(
        _demo_supervisor, _demo_workers(), make_validator_node(llm=None), checkpointer
    )


def _placeholder_node(state: SupplyChainStateV2) -> dict:  # pragma: no cover - drawing only
    return {}


def _placeholder_supervisor(state: SupplyChainStateV2) -> dict:  # pragma: no cover
    return {"next_agent": "FINISH"}


def _placeholder_validator(state: SupplyChainStateV2) -> dict:  # pragma: no cover
    return {"verdict": "complete", "verdict_reason": ""}


def ascii_graph_v2() -> str:
    """Text rendering of the v2 topology, for labs and BUILD_MANUAL.md.

    Placeholder nodes, zero LLM dependency — same trick as v1's ascii_graph.
    """
    app = _assemble_v2(
        _placeholder_supervisor,
        {name: _placeholder_node for name in WORKERS_V2},
        _placeholder_validator,
    )
    drawable = app.get_graph()
    try:
        return drawable.draw_ascii()
    except Exception:  # grandalf not installed, or layout failure
        return drawable.draw_mermaid()
