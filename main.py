"""WHY THIS EXISTS
---------------
The CLI front door to the whole training project — what a consultant runs to
*see* the multi-agent system work before reading any code:

- ``python main.py --ask "..."``      one question through the supervisor graph
- ``python main.py --demo``           5 canned scenarios (one per worker + the
                                      flagship 3-agent SwiftShip question)
- ``python main.py --offline-check``  verifies the no-API-key fallbacks
                                      (PageIndex lexical RAG, SQL guardrails,
                                      EDA) without building the graph at all

The streaming loop doubles as a LangGraph lesson: with
``stream_mode="values"`` the app yields the *full state after every node*, so
by watching whether the message list grew we can tell a supervisor decision
(``[supervisor -> sql_analyst]``) from a worker answer — exactly the
hub-and-spoke rhythm drawn in ``src/graph.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # allow `python path/to/main.py` from anywhere
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import is_offline  # noqa: E402

DEMO_QUESTIONS: list[tuple[str, str]] = [
    (
        "data_analyst",
        "Which shipping mode has the worst late-delivery rate, and how imbalanced "
        "is the dataset overall?",
    ),
    (
        "ml_engineer",
        "Train the hist_gb late-delivery model and tell me the top 5 drivers of "
        "late delivery.",
    ),
    (
        "contracts_analyst",
        "What late-delivery penalty does the Atlas Freight contract specify, and "
        "what is its on-time SLA?",
    ),
    (
        "sql_analyst",
        "How many orders did each carrier handle, and what share of each "
        "carrier's orders were late?",
    ),
    (
        "flagship (3 agents)",
        "Order 104872 ships Same Day via SwiftShip Express to Western Europe — "
        "how likely is it to be late, what penalty applies under the SwiftShip "
        "contract if it is, and what was SwiftShip's late rate last quarter?",
    ),
]

OFFLINE_POINTER = """\
This command needs an LLM, but no API key was detected (or OFFLINE=1 is set).

To enable the agents:
  1. Copy .env.example to .env
  2. Set LLM_PROVIDER (azure_openai | openai | anthropic) and the matching key
  3. Re-run this command

Works without a key right now:
  python main.py --offline-check     # verify all offline fallbacks
  python -m pytest tests/ -q         # full offline test suite
"""


def _ensure_data() -> None:
    """Generate any missing data artifact via the (idempotent, seeded) generators."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"
    contracts_dir = PROJECT_ROOT / "data" / "contracts"
    db_path = PROJECT_ROOT / "data" / "warehouse.db"
    catalog_path = PROJECT_ROOT / "data" / "metadata_catalog.json"

    if not csv_path.exists():
        from data import generate_supply_chain_data

        generate_supply_chain_data.main()
    if not contracts_dir.exists() or not any(contracts_dir.glob("*.md")):
        from data import generate_contracts

        generate_contracts.main()
    if not db_path.exists() or not catalog_path.exists():
        from data import build_database

        build_database.main()


# --------------------------------------------------------------------------
# --offline-check: exercise the no-LLM fallbacks, no graph build involved
# --------------------------------------------------------------------------

def _check_contracts_rag() -> tuple[bool, str]:
    from src.agents.tools_rag import search_contracts

    result = search_contracts.invoke({"query": "late delivery penalty SwiftShip Express"})
    ok = isinstance(result, str) and "###" in result and len(result) > 100
    return ok, f"search_contracts returned {len(result)} chars; first line: {result.splitlines()[0] if result else '(empty)'}"


def _check_sql_safe_query() -> tuple[bool, str]:
    from src.agents.tools_sql import run_sql_query

    result = run_sql_query.invoke({"sql": "SELECT carrier_name, on_time_sla_pct FROM carriers"})
    ok = isinstance(result, str) and "SQL_ERROR" not in result and len(result) > 20
    return ok, f"safe SELECT over carriers returned {len(result)} chars without SQL_ERROR" if ok else f"unexpected: {result[:200]}"


def _check_sql_guardrail() -> tuple[bool, str]:
    from src.agents.tools_sql import run_sql_query

    result = run_sql_query.invoke({"sql": "DROP TABLE orders"})
    ok = isinstance(result, str) and "SQL_ERROR" in result
    return ok, "DROP TABLE was rejected with SQL_ERROR" if ok else f"guardrail missed: {result[:200]}"


def _check_ml_class_balance() -> tuple[bool, str]:
    from src.ml_pipeline.data_loader import load_orders
    from src.ml_pipeline.eda import class_balance

    balance = class_balance(load_orders())
    ok = (
        isinstance(balance, dict)
        and {"on_time", "late", "late_rate"} <= set(balance)
        and 0.0 < float(balance["late_rate"]) < 1.0
    )
    return ok, f"class_balance -> {balance}"


def run_offline_check() -> int:
    """Run every offline-fallback check; print PASS/FAIL per check; 0 = all pass."""
    import os

    # This command verifies the *offline* code paths, so force them on — a
    # half-configured key in the environment must not flip us into LLM mode.
    os.environ["OFFLINE"] = "1"
    print("Offline checks (no API key required)")
    print("=" * 60)
    _ensure_data()
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("contracts RAG (PageIndex lexical fallback)", _check_contracts_rag),
        ("SQL tool: safe SELECT runs", _check_sql_safe_query),
        ("SQL tool: guardrail rejects DROP", _check_sql_guardrail),
        ("ML pipeline: class_balance", _check_ml_class_balance),
    ]
    failures = 0
    for label, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:  # missing sibling module, bad data, etc.
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {label}\n       {detail}")
    print("=" * 60)
    print("All offline checks passed." if failures == 0 else f"{failures} check(s) failed.")
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------
# --ask / --demo: stream the supervisor graph
# --------------------------------------------------------------------------

def _build_app() -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    from src.graph import build_graph

    return build_graph(checkpointer=MemorySaver())


def ask(app: Any, question: str, thread_id: str) -> None:
    """Stream one question through the graph, printing each supervisor hop."""
    from langchain_core.messages import AIMessage, HumanMessage

    config = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}
    print(f"\nQ: {question}")
    print("-" * 60)

    seen_messages = 0
    final_state: dict | None = None
    first_yield = True
    for state in app.stream(
        {"messages": [HumanMessage(content=question)], "next_agent": ""},
        config=config,
        stream_mode="values",
    ):
        messages = state.get("messages", [])
        if first_yield:  # the input state itself
            seen_messages = len(messages)
            first_yield = False
        elif len(messages) == seen_messages:  # no new message => supervisor decided
            print(f"  [supervisor -> {state.get('next_agent', '?')}]")
        else:  # a worker appended its answer
            for message in messages[seen_messages:]:
                name = getattr(message, "name", None) or "agent"
                print(f"  [{name}] answered ({len(str(message.content))} chars)")
            seen_messages = len(messages)
        final_state = state

    print("-" * 60)
    answers = [
        m for m in (final_state or {}).get("messages", []) if isinstance(m, AIMessage)
    ]
    if answers:
        print("Final answer:\n" + str(answers[-1].content))
    else:
        print("(The supervisor finished without dispatching a worker.)")


def _build_app_or_explain() -> Any | None:
    """Build the graph; on provider misconfiguration explain instead of crashing."""
    try:
        return _build_app()
    except Exception as exc:
        print(f"Could not initialize the LLM provider: {type(exc).__name__}: {exc}\n")
        print(OFFLINE_POINTER)
        return None


def run_ask(question: str) -> int:
    if is_offline():
        print(OFFLINE_POINTER)
        return 0
    _ensure_data()
    app = _build_app_or_explain()
    if app is None:
        return 1
    ask(app, question, thread_id="ask-1")
    return 0


def run_demo() -> int:
    if is_offline():
        print(OFFLINE_POINTER)
        return 0
    _ensure_data()
    app = _build_app_or_explain()
    if app is None:
        return 1
    print(f"Running {len(DEMO_QUESTIONS)} demo scenarios through the supervisor graph.")
    for i, (label, question) in enumerate(DEMO_QUESTIONS, start=1):
        print(f"\n=== Scenario {i}/{len(DEMO_QUESTIONS)} — {label} ===")
        ask(app, question, thread_id=f"demo-{i}")  # fresh thread per scenario
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LangGraph supply-chain multi-agent demo (Week 3 & 4 training).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ask", metavar="QUESTION", help="ask the agent team one question")
    group.add_argument("--demo", action="store_true", help="run the 5 canned demo scenarios")
    group.add_argument(
        "--offline-check",
        action="store_true",
        help="verify offline fallbacks (no API key, no graph build)",
    )
    args = parser.parse_args(argv)

    if args.offline_check:
        return run_offline_check()
    if args.demo:
        return run_demo()
    return run_ask(args.ask)


if __name__ == "__main__":
    raise SystemExit(main())
