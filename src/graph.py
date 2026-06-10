"""WHY THIS EXISTS
---------------
This is the assembly point of the whole course: one StateGraph that wires the
supervisor and the four specialist workers into a hub-and-spoke machine over
``SupplyChainState``:

    START -> supervisor -(next_agent)-> worker -> supervisor -> ... -> END

The shape to internalize:

- The supervisor is the only node with *outgoing conditional* edges; it writes
  ``state["next_agent"]`` and the conditional edge maps that string to a node
  name (or END for "FINISH").
- Every worker has exactly one fixed edge back to the supervisor — workers
  never talk to each other directly. All coordination flows through state.

DEPENDENCY INJECTION (the testability lesson): ``build_graph(llm, workers)``
takes its expensive parts as parameters. Production calls ``build_graph()``
and gets real ReAct agents on ``get_llm()``; tests pass a fake structured-
output llm plus plain-callable worker nodes and exercise the *exact same
wiring* offline. Graph topology bugs get caught without an API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from langgraph.graph import END, START, StateGraph

from src.agents.supervisor import make_supervisor_node, make_worker_node
from src.state import WORKERS, SupplyChainState

if TYPE_CHECKING:  # import only for type hints — keeps runtime deps minimal
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

WorkerNode = Callable[[SupplyChainState], dict]


def _assemble(
    supervisor_node: WorkerNode,
    workers: dict[str, WorkerNode],
    checkpointer: Any = None,
) -> "CompiledStateGraph":
    """Wire supervisor + worker nodes into the compiled hub-and-spoke graph."""
    builder = StateGraph(SupplyChainState)
    builder.add_node("supervisor", supervisor_node)
    for name, node in workers.items():
        builder.add_node(name, node)
        builder.add_edge(name, "supervisor")  # every worker reports back
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {name: name for name in workers} | {"FINISH": END},
    )
    return builder.compile(checkpointer=checkpointer)


def build_graph(
    llm: "BaseChatModel | None" = None,
    workers: dict[str, WorkerNode] | None = None,
    checkpointer: Any = None,
) -> "CompiledStateGraph":
    """Build and compile the supervisor multi-agent graph.

    Args:
        llm: chat model for the supervisor (and, by default, the workers).
            ``None`` -> ``get_llm()`` from env config (requires an API key).
        workers: optional override mapping worker name -> *node callable*
            (an already-wrapped ``state -> dict`` function). ``None`` builds
            the real ReAct agents from ``src.agents.workers`` and wraps them
            with ``make_worker_node``. Tests inject plain functions here.
        checkpointer: optional LangGraph checkpointer (e.g. ``MemorySaver``)
            enabling multi-turn memory per ``thread_id``.
    """
    if workers is None:
        # Imported lazily: building real agents pulls in every tool module.
        from src.agents import workers as worker_factories

        factories: dict[str, Callable[..., Any]] = {
            "data_analyst": worker_factories.make_data_analyst,
            "ml_engineer": worker_factories.make_ml_engineer,
            "contracts_analyst": worker_factories.make_contracts_analyst,
            "sql_analyst": worker_factories.make_sql_analyst,
        }
        workers = {
            name: make_worker_node(name, factory(llm)) for name, factory in factories.items()
        }
    return _assemble(make_supervisor_node(llm), workers, checkpointer)


def _placeholder_node(state: SupplyChainState) -> dict:  # pragma: no cover - drawing only
    return {}


def _placeholder_supervisor(state: SupplyChainState) -> dict:  # pragma: no cover
    return {"next_agent": "FINISH"}


def ascii_graph() -> str:
    """Text rendering of the graph topology, for labs and the README.

    Builds the graph with no-op placeholder nodes (same wiring, zero LLM
    dependency) so it works offline. Prefers ASCII art; falls back to Mermaid
    text when the optional ``grandalf`` package is unavailable.
    """
    app = _assemble(
        _placeholder_supervisor,
        {name: _placeholder_node for name in WORKERS},
    )
    drawable = app.get_graph()
    try:
        return drawable.draw_ascii()
    except Exception:  # grandalf not installed, or layout failure
        return drawable.draw_mermaid()
