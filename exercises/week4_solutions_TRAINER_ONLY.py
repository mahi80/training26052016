"""WEEK 4 TRIAL — TRAINER SOLUTIONS. Do not distribute to trainees.

WHY THIS EXISTS
---------------
Reference implementations for every task in ``week4_starter.py``, with TRAINER
NOTES on the failure modes trainees actually hit. Checks are identical to the
starter's. Offline this must print 250/300 (TASK 5 SKIPPED) and exit 0; with an
API key it attempts the flagship scenario for the full 300:

    python exercises/week4_solutions_TRAINER_ONLY.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FLAGSHIP_QUESTION = (
    "Order 104872 ships Same Day via SwiftShip Express to Western Europe — how "
    "likely is it to be late, what penalty applies under the SwiftShip contract "
    "if it is, and what was SwiftShip's late rate last quarter?"
)


# ============================================================================
# TASK 1 (70 XP) — Build the PageIndex and retrieve Atlas payment terms
# ============================================================================
# TRAINER NOTES:
# * Query "payment terms" with no carrier name: every contract has a payment
#   section, so the lexical fallback returns a grab-bag and the atlas assert
#   fires. THE lesson of reasoning-based retrieval: specific queries navigate,
#   vague ones wander. Ask trainees to compare both queries' hits live.
# * use_llm=None (the default) silently calls the API when a key is present —
#   fine in prod, but the trial demands use_llm=False for determinism.
# * Atlas section to find: "## 7. Invoicing & Payment" (net 45 days, 1.0%/month
#   late-payment interest) — have trainees read the .text of the top hit aloud.

def retrieve_atlas_payment_terms() -> list[Any]:
    """Build the tree index offline, then navigate to Atlas's payment terms."""
    from src.pageindex import PageIndexRetriever, build_index

    build_index(use_llm=False)  # deterministic heuristic summaries, no API key
    retriever = PageIndexRetriever()
    return retriever.search(
        "Atlas Freight invoicing and payment terms net 45 days", top_k=3
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
# TRAINER NOTES:
# * run_sql_query("...") positional — LangChain tools take a dict:
#   .invoke({"sql": "..."}). The TypeError confuses people; show it once.
# * strftime('%Y', order_date) = '2025' also works; LIKE '2025%' is simpler
#   because order_date is ISO TEXT in SQLite (no real date type).
# * Trailing ';' is stripped by the tool, but ';' INSIDE = rejected — a
#   trainee who pastes two statements gets the multi-statement SQL_ERROR.
# * SELECT * with no GROUP BY returns 50 raw rows (auto-LIMIT) and fails the
#   five-markets assert — aggregate, don't dump.

def late_orders_per_market_2025() -> str:
    """Markdown table of late orders per market in 2025 via the guarded tool."""
    from src.agents.tools_sql import run_sql_query

    sql = (
        "SELECT market, "
        "       SUM(late_delivery) AS late_orders, "
        "       COUNT(*) AS total_orders, "
        "       ROUND(AVG(late_delivery), 3) AS late_rate "
        "FROM orders "
        "WHERE order_date LIKE '2025%' "
        "GROUP BY market "
        "ORDER BY late_orders DESC"
    )
    return run_sql_query.invoke({"sql": sql})


def _check_task2() -> None:
    out = late_orders_per_market_2025()
    assert isinstance(out, str), "return the tool's string output"
    assert not out.startswith("SQL_ERROR"), f"the tool rejected your SQL: {out}"
    assert "|" in out, "expected a markdown table"
    for market in ("Africa", "Europe", "LATAM", "Pacific Asia", "USCA"):
        assert market in out, f"market {market!r} missing — did you GROUP BY market?"
    from src.agents.tools_sql import run_sql_query

    blocked = run_sql_query.invoke({"sql": "DELETE FROM orders"})
    assert blocked.startswith("SQL_ERROR"), "guardrails should reject writes"


# ============================================================================
# TASK 3 (50 XP) — Think like the supervisor: route questions to workers
# ============================================================================
# TRAINER NOTES:
# * Priority order is the whole exercise: "Which carrier MSA has the strictest
#   on-time SLA?" contains zero ML/SQL words but trainees who check SQL first
#   on "how many"-style phrasing still pass; checking ML before contracts
#   breaks nothing HERE but discuss why contracts-first is safer ("evaluate
#   the penalty clause..."). The real supervisor solves this with an LLM +
#   structured output; keywords make the same decision auditable.
# * Returning "ml" instead of "ml_engineer" — names must come from WORKERS;
#   that tuple is the shared contract between supervisor and graph nodes.

ROUTING_CASES: list[tuple[str, str]] = [
    ("What is the late delivery rate by shipping mode this year?", "data_analyst"),
    ("Show the monthly order volume and late rate trend.", "data_analyst"),
    ("Train a classifier to predict late deliveries and tune its threshold for recall.", "ml_engineer"),
    ("What penalty applies under the SwiftShip contract if a Same Day shipment is late?", "contracts_analyst"),
    ("Which carrier MSA has the strictest on-time SLA commitment?", "contracts_analyst"),
    ("How many late orders did each market have in 2025? Query the warehouse.", "sql_analyst"),
]

_CONTRACT_KEYWORDS = ("contract", "penalty", "msa", "sla", "clause", "termination", "payment terms")
_SQL_KEYWORDS = ("sql", "query", "warehouse", "how many", "count", "database")
_ML_KEYWORDS = ("train", "model", "predict", "classifier", "threshold", "recall", "score", "risk")


def route_question(question: str) -> str:
    """Keyword router mirroring the supervisor: contracts -> sql -> ml -> data."""
    q = question.lower()
    if any(k in q for k in _CONTRACT_KEYWORDS):
        return "contracts_analyst"
    if any(k in q for k in _SQL_KEYWORDS):
        return "sql_analyst"
    if any(k in q for k in _ML_KEYWORDS):
        return "ml_engineer"
    return "data_analyst"  # EDA is the safe default for vague questions


def _check_task3() -> None:
    from src.state import WORKERS

    for question, expected in ROUTING_CASES:
        got = route_question(question)
        assert got in WORKERS, f"{got!r} is not a worker name {WORKERS}"
        assert got == expected, f"{question!r}\n  routed to {got!r}, want {expected!r}"


# ============================================================================
# TASK 4 (70 XP) — Author a NEW @tool over the contracts index
# ============================================================================
# TRAINER NOTES:
# * Only scanning top-level nodes: penalty clauses live in ### subsections,
#   so the count comes up short and the "recursing into children?" assert
#   message points straight at it.
# * term.lower() on one side only — "SwiftShip" then fails (index text has
#   mixed case). Lowercase BOTH the needle and the haystack.
# * Returning the int count — tools return strings; the LLM needs prose it
#   can quote ("1 of 6 contracts mention 'SwiftShip'").

def make_contract_term_counter() -> Any:
    """A new contracts_analyst tool: how many contracts mention <term>?"""
    from langchain_core.tools import tool

    from src.pageindex import load_index

    @tool
    def count_contracts_mentioning(term: str) -> str:
        """Count how many of the six contracts mention a term anywhere.

        Use this for coverage questions like "do all carrier contracts have a
        fuel surcharge clause?" before drilling into specific sections with
        search_contracts. Matching is case-insensitive over section titles
        and full text.

        Args:
            term: word or phrase to look for, e.g. "penalty" or "force majeure".
        """
        index = load_index()
        needle = term.lower()

        def mentions(node: dict) -> bool:
            haystack = f"{node.get('title', '')} {node.get('text', '')}".lower()
            if needle in haystack:
                return True
            return any(mentions(child) for child in node.get("children", []))

        docs = index["docs"]
        count = sum(1 for d in docs if any(mentions(n) for n in d["nodes"]))
        return f"{count} of {len(docs)} contracts mention '{term}'"

    return count_contracts_mentioning


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
# TRAINER NOTES:
# * The recursion_limit matters: supervisor -> ml_engineer -> supervisor ->
#   contracts_analyst -> supervisor -> data_analyst/sql_analyst -> supervisor
#   -> FINISH, each worker running its own ReAct tool loop. The default limit
#   (25) can trip mid-scenario; 40 gives headroom.
# * Expect ~60-120s wall time and a dozen LLM calls — run it as the finale
#   demo, not in a loop. Watch the [supervisor -> worker] hops via main.py
#   --demo if you want the pretty-printed version instead.
# * Offline this MUST skip, not fail — graceful degradation is itself one of
#   the week's lessons (is_offline() checks before get_llm(), never after).

def flagship_scenario() -> str | None:
    """Run the flagship question through the full graph; None when offline."""
    from src.config import is_offline

    if is_offline():
        return None  # graceful classroom degradation — runner records a SKIP

    from langchain_core.messages import HumanMessage

    from src.graph import build_graph

    app = build_graph()
    result = app.invoke(
        {"messages": [HumanMessage(content=FLAGSHIP_QUESTION)], "next_agent": ""},
        config={"recursion_limit": 40},
    )
    return str(result["messages"][-1].content)


def _check_task5() -> str | None:
    answer = flagship_scenario()
    if answer is None:
        return "SKIP"
    assert isinstance(answer, str) and len(answer) > 50, "expected a real synthesis"
    assert "swiftship" in answer.lower(), "the answer should discuss SwiftShip"
    print(f"Flagship answer:\n{answer[:600]}")
    return None


# ============================================================================
# The Trial runner — identical to the starter's
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
    print("WEEK 4 TRIAL (TRAINER SOLUTIONS) — 250/300 offline, 300/300 online")
    print("=" * 72)
    earned = 0
    for label, xp, check in TASKS:
        print(f"\n--- {label} [{xp} XP] ---")
        try:
            outcome = check()
        except (NotImplementedError, AssertionError) as exc:
            print(f"FAILED: {exc}")
            print(f"XP so far: {earned} / 300")
            return 1
        if outcome == "SKIP":
            print("SKIPPED (offline — set an API key in .env to attempt the stretch)")
            continue
        earned += xp
        print(f"PASS (+{xp} XP)")
    print("\n" + "=" * 72)
    print(f"TRIAL COMPLETE — {earned} / 300 XP.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
