"""WEEK 4 ASSESSMENT — "The Trial": RAG + NL2SQL + multi-agent integration (300 XP).

WHY THIS EXISTS
---------------
Week 4 turned standalone capabilities into a supervised multi-agent system. This
trial proves you can drive each layer of it yourself:

    TASK 1 (70 XP)  Build the PageIndex tree + retrieve contract terms (offline)
    TASK 2 (60 XP)  Answer a business question through the guarded SQL tool
    TASK 3 (50 XP)  Think like the supervisor: route 6 questions to workers
    TASK 4 (70 XP)  Author a brand-new @tool over the contracts index
    TASK 5 (50 XP)  STRETCH (ONLINE ONLY): run the flagship 3-agent scenario

Each task is a function stub whose docstring tells you exactly what to build,
followed by a ``_check_*`` function (DO NOT EDIT) whose asserts you must make pass.
Run the file after every task — the trial stops at the first failure with a hint:

    python exercises/week4_starter.py        # run from the week3&4 project root

TASKS 1–4 run fully OFFLINE (the PageIndex retriever degrades to lexical scoring,
SQL guardrails never involve an LLM). TASK 5 needs an API key in ``.env`` — it is
skipped gracefully when offline, so 250/300 XP is a complete offline run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The flagship demo question from ARCHITECTURE.md §1 — touches 3 of the 4 agents.
FLAGSHIP_QUESTION = (
    "Order 104872 ships Same Day via SwiftShip Express to Western Europe — how "
    "likely is it to be late, what penalty applies under the SwiftShip contract "
    "if it is, and what was SwiftShip's late rate last quarter?"
)


# ============================================================================
# TASK 1 (70 XP) — Build the PageIndex and retrieve Atlas payment terms
# ============================================================================

def retrieve_atlas_payment_terms() -> list[Any]:
    """Build the contracts tree index, then retrieve Atlas Freight's payment terms.

    Requirements (all from ``src.pageindex``):
    1. ``build_index(use_llm=False)`` — parse the 6 contracts in
       ``data/contracts/`` into tree indexes with heuristic (offline) summaries.
       Forcing ``use_llm=False`` keeps the build deterministic and key-free.
    2. ``PageIndexRetriever().search(query, top_k=3)`` — ask for Atlas Freight's
       invoicing / payment terms. Make the query SPECIFIC: name the carrier AND
       the topic ("payment" alone matches every contract — they all have a
       payment section; that ambiguity is the lesson).
    3. Return the list of RetrievedSection results.
    """
    raise NotImplementedError(
        "TASK 1: implement retrieve_atlas_payment_terms() — "
        "from src.pageindex import build_index, PageIndexRetriever; "
        "build_index(use_llm=False), then search for Atlas Freight's "
        "invoicing/payment terms with top_k=3 and return the hits."
    )


def _check_task1() -> None:
    hits = retrieve_atlas_payment_terms()
    assert isinstance(hits, list) and hits, "expected a non-empty list of sections"
    assert len(hits) <= 5, "keep top_k small — retrieval is about precision"
    for attr in ("node_id", "doc", "title", "path", "text", "score"):
        assert hasattr(hits[0], attr), f"results should be RetrievedSection ({attr}?)"
    assert any("atlas" in h.doc.lower() for h in hits), (
        f"no Atlas section retrieved — docs found: {[h.doc for h in hits]}. "
        "Did your query name the carrier, or just say 'payment terms'?"
    )


# ============================================================================
# TASK 2 (60 XP) — Late orders per market in 2025, through the guarded tool
# ============================================================================

def late_orders_per_market_2025() -> str:
    """Answer "how many late orders per market in 2025?" via ``run_sql_query``.

    Requirements:
    - Use the ``run_sql_query`` tool from ``src.agents.tools_sql`` — it is a
      LangChain tool, so call it with ``run_sql_query.invoke({"sql": ...})``.
    - Write ONE SELECT that filters to 2025 (``order_date`` is ISO text —
      ``LIKE '2025%'`` works), GROUPs BY market, and SUMs ``late_delivery``.
    - Respect the guardrails: single statement, no semicolons, read-only.
    - Return the tool's string output (a markdown table) unchanged.
    """
    raise NotImplementedError(
        "TASK 2: implement late_orders_per_market_2025() — "
        "from src.agents.tools_sql import run_sql_query; invoke it with a "
        "SELECT market, SUM(late_delivery) ... WHERE order_date LIKE '2025%' "
        "GROUP BY market query and return the markdown string."
    )


def _check_task2() -> None:
    out = late_orders_per_market_2025()
    assert isinstance(out, str), "return the tool's string output"
    assert not out.startswith("SQL_ERROR"), f"the tool rejected your SQL: {out}"
    assert "|" in out, "expected a markdown table"
    for market in ("Africa", "Europe", "LATAM", "Pacific Asia", "USCA"):
        assert market in out, f"market {market!r} missing — did you GROUP BY market?"
    # Guardrail spot-check (provided): writes must bounce off as SQL_ERROR text.
    from src.agents.tools_sql import run_sql_query

    blocked = run_sql_query.invoke({"sql": "DELETE FROM orders"})
    assert blocked.startswith("SQL_ERROR"), "guardrails should reject writes"


# ============================================================================
# TASK 3 (50 XP) — Think like the supervisor: route questions to workers
# ============================================================================

# The 6 questions your router must classify, with the expected worker.
# These mirror the supervisor's Router(next=Literal[...]) decision — done with
# keywords here so it is gradeable offline. DO NOT EDIT this table.
ROUTING_CASES: list[tuple[str, str]] = [
    ("What is the late delivery rate by shipping mode this year?", "data_analyst"),
    ("Show the monthly order volume and late rate trend.", "data_analyst"),
    ("Train a classifier to predict late deliveries and tune its threshold for recall.", "ml_engineer"),
    ("What penalty applies under the SwiftShip contract if a Same Day shipment is late?", "contracts_analyst"),
    ("Which carrier MSA has the strictest on-time SLA commitment?", "contracts_analyst"),
    ("How many late orders did each market have in 2025? Query the warehouse.", "sql_analyst"),
]


def route_question(question: str) -> str:
    """Return the correct worker name from ``src.state.WORKERS`` for a question.

    Requirements:
    - Pure keyword routing over the lowercased question — no LLM.
    - ORDER MATTERS: check contract vocabulary (penalty/contract/msa/sla/...)
      BEFORE ML vocabulary, or "the SLA in the MSA" style questions fall
      through to the wrong worker. Then SQL vocabulary (query/warehouse/
      how many/...), then ML (train/model/predict/threshold/...), and default
      to ``data_analyst`` — EDA is the safe fallback for vague questions.
    """
    raise NotImplementedError(
        "TASK 3: implement route_question(question) — keyword checks in "
        "priority order: contracts -> sql -> ml -> default data_analyst. "
        "Return names from src.state.WORKERS."
    )


def _check_task3() -> None:
    from src.state import WORKERS

    for question, expected in ROUTING_CASES:
        got = route_question(question)
        assert got in WORKERS, f"{got!r} is not a worker name {WORKERS}"
        assert got == expected, f"{question!r}\n  routed to {got!r}, want {expected!r}"


# ============================================================================
# TASK 4 (70 XP) — Author a NEW @tool over the contracts index
# ============================================================================

def make_contract_term_counter() -> Any:
    """Build and return a new ``@tool``: ``count_contracts_mentioning(term)``.

    Requirements:
    - Decorate a function named exactly ``count_contracts_mentioning`` taking
      one argument ``term: str`` (``from langchain_core.tools import tool``).
    - Inside: ``load_index()`` from ``src.pageindex`` (TASK 1 built the JSON),
      walk EVERY doc's node tree — title + text, RECURSING into ``children``
      (top-level nodes alone undercount!) — and count how many of the 6
      contracts mention the term, case-insensitively.
    - Return the string ``"<count> of <total> contracts mention '<term>'"``.
    - Write a docstring the contracts_analyst LLM could route on.
    - Return the decorated tool object.
    """
    raise NotImplementedError(
        "TASK 4: implement make_contract_term_counter() — define "
        "count_contracts_mentioning(term: str) -> str inside, decorate with "
        "@tool, recurse through index['docs'][i]['nodes'] and children, "
        "and return the tool object."
    )


def _check_task4() -> None:
    from langchain_core.tools import BaseTool

    counter = make_contract_term_counter()
    assert isinstance(counter, BaseTool), "return the decorated tool object"
    assert counter.name == "count_contracts_mentioning", f"name: {counter.name!r}"
    assert counter.description, "empty docstring — the LLM can't route to this"
    out_penalty = counter.invoke({"term": "penalty"})
    assert isinstance(out_penalty, str), "tools return strings"
    assert "6 of 6" in out_penalty, (
        f"got {out_penalty!r} — every contract has a penalty clause. "
        "Are you recursing into children nodes?"
    )
    out_swift = counter.invoke({"term": "SwiftShip"})
    assert "1 of 6" in out_swift, (
        f"got {out_swift!r} — only one contract names SwiftShip. "
        "Is your match case-insensitive on BOTH sides?"
    )


# ============================================================================
# TASK 5 (50 XP) — STRETCH, ONLINE ONLY: the flagship 3-agent scenario
# ============================================================================

def flagship_scenario() -> str | None:
    """Run FLAGSHIP_QUESTION through the full supervisor graph. ONLINE ONLY.

    Requirements:
    - If ``is_offline()`` (from ``src.config``) return ``None`` — the runner
      records a graceful SKIP, not a failure. Never let this task crash an
      offline classroom.
    - Online: ``from src.graph import build_graph``; invoke the compiled app
      with ``{"messages": [HumanMessage(content=FLAGSHIP_QUESTION)],
      "next_agent": ""}`` and ``config={"recursion_limit": 40}`` (three agent
      round-trips need headroom). Return the final message's content.
    """
    raise NotImplementedError(
        "TASK 5 (stretch): implement flagship_scenario() — return None when "
        "is_offline(), else build_graph() and invoke it on FLAGSHIP_QUESTION "
        "with recursion_limit=40, returning the last message's content."
    )


def _check_task5() -> str | None:
    answer = flagship_scenario()
    if answer is None:
        return "SKIP"  # offline classroom — stretch task deferred, not failed
    assert isinstance(answer, str) and len(answer) > 50, "expected a real synthesis"
    assert "swiftship" in answer.lower(), "the answer should discuss SwiftShip"
    print(f"Flagship answer:\n{answer[:600]}")
    return None


# ============================================================================
# The Trial runner — do not edit below this line
# ============================================================================

TASKS: list[tuple[str, int, Any]] = [
    ("TASK 1 — PageIndex build + retrieval", 70, _check_task1),
    ("TASK 2 — guarded NL2SQL query", 60, _check_task2),
    ("TASK 3 — supervisor-style routing", 50, _check_task3),
    ("TASK 4 — new tool over the index", 70, _check_task4),
    ("TASK 5 — flagship scenario (STRETCH, online)", 50, _check_task5),
]


def main() -> int:
    print("=" * 72)
    print("WEEK 4 TRIAL — RAG + NL2SQL + multi-agent integration (300 XP)")
    print("(offline classrooms: 250 XP is a complete run; TASK 5 needs a key)")
    print("=" * 72)
    earned = 0
    for label, xp, check in TASKS:
        print(f"\n--- {label} [{xp} XP] ---")
        try:
            outcome = check()
        except NotImplementedError as exc:
            print(f"NOT IMPLEMENTED: {exc}")
            print("\nThe trial stops here. Implement the TODO above and re-run.")
            print(f"XP so far: {earned} / 300")
            return 1
        except AssertionError as exc:
            print(f"ASSERT FAILED: {exc}")
            print("\nThe trial stops here. Fix the implementation and re-run.")
            print(f"XP so far: {earned} / 300")
            return 1
        if outcome == "SKIP":
            print("SKIPPED (offline — set an API key in .env to attempt the stretch)")
            continue
        earned += xp
        print(f"PASS (+{xp} XP)")
    print("\n" + "=" * 72)
    print(f"TRIAL COMPLETE — {earned} / 300 XP. Week 4 skills confirmed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
