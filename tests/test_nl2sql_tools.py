"""Guardrail + behavior tests for the NL2SQL tools (§4.6, tools_sql).

WHY THIS EXISTS
---------------
``run_sql_query`` executes SQL written by an LLM — i.e. untrusted input. These
tests pin the security contract (read-only, single statement, SELECT-only, row
cap) and the self-repair contract (failures come back as "SQL_ERROR: ..." strings,
never exceptions, so the ReAct loop can read the error and retry).

Note: the functions are wrapped by ``@tool`` (they are ``StructuredTool`` objects,
not plain callables), so tests call them via ``.invoke({...})`` — the same code
path the agent uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

# src/agents may have no __init__.py yet (owned by a sibling module) — Python 3.3+
# namespace packages make this import work regardless.
from src.agents.tools_sql import (  # noqa: E402
    get_table_schema,
    list_warehouse_tables,
    run_sql_query,
    search_metadata,
)

CARRIER_NAMES = {
    "SwiftShip Express",
    "Atlas Freight Co.",
    "Pacific Crest Carriers",
    "NordHaul Logistics",
}


def data_rows(markdown: str) -> list[str]:
    """Markdown-table body rows: pipe-lines minus header and separator."""
    pipe_lines = [line for line in markdown.splitlines() if line.startswith("|")]
    return pipe_lines[2:]


# ----------------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------------

def test_select_carriers_returns_all_four() -> None:
    result = run_sql_query.invoke({"sql": "SELECT carrier_name FROM carriers"})
    assert not result.startswith("SQL_ERROR"), result
    rows = data_rows(result)
    assert len(rows) == 4
    for name in CARRIER_NAMES:
        assert name in result, f"missing carrier {name!r} in:\n{result}"


def test_trailing_semicolon_is_tolerated() -> None:
    result = run_sql_query.invoke({"sql": "SELECT carrier_name FROM carriers;"})
    assert not result.startswith("SQL_ERROR"), result
    assert len(data_rows(result)) == 4


def test_with_cte_is_allowed() -> None:
    sql = (
        "WITH late AS (SELECT carrier_id, AVG(late_delivery) AS late_rate "
        "FROM orders GROUP BY carrier_id) "
        "SELECT c.carrier_name, l.late_rate FROM late l "
        "JOIN carriers c ON c.carrier_id = l.carrier_id ORDER BY l.late_rate DESC"
    )
    result = run_sql_query.invoke({"sql": sql})
    assert not result.startswith("SQL_ERROR"), result
    assert len(data_rows(result)) == 4


# ----------------------------------------------------------------------------------
# Row cap
# ----------------------------------------------------------------------------------

def test_limit_auto_added_caps_rows_at_50() -> None:
    # orders has ~12,000 rows; without the auto-LIMIT this would flood the agent.
    result = run_sql_query.invoke({"sql": "SELECT order_id FROM orders"})
    assert not result.startswith("SQL_ERROR"), result
    assert len(data_rows(result)) <= 50


def test_explicit_limit_is_respected() -> None:
    result = run_sql_query.invoke({"sql": "SELECT order_id FROM orders LIMIT 7"})
    assert not result.startswith("SQL_ERROR"), result
    assert len(data_rows(result)) == 7


# ----------------------------------------------------------------------------------
# Guardrails — every rejection is an SQL_ERROR *string*, never an exception
# ----------------------------------------------------------------------------------

def test_drop_table_rejected() -> None:
    result = run_sql_query.invoke({"sql": "DROP TABLE orders"})
    assert result.startswith("SQL_ERROR")


def test_update_inside_select_rejected_by_keyword_blocklist() -> None:
    result = run_sql_query.invoke(
        {"sql": "SELECT * FROM orders WHERE order_id IN (SELECT 1) AND 1=1; DELETE FROM orders"}
    )
    assert result.startswith("SQL_ERROR")


def test_multi_statement_rejected() -> None:
    result = run_sql_query.invoke({"sql": "SELECT 1; SELECT 2"})
    assert result.startswith("SQL_ERROR")
    assert "statement" in result.lower()


def test_non_select_first_keyword_rejected() -> None:
    result = run_sql_query.invoke({"sql": "PRAGMA table_info(orders)"})
    assert result.startswith("SQL_ERROR")


def test_bad_table_returns_sql_error_string_not_exception() -> None:
    result = run_sql_query.invoke({"sql": "SELECT * FROM nope"})
    assert isinstance(result, str)
    assert result.startswith("SQL_ERROR")
    assert "nope" in result  # the agent needs the engine message to self-repair


def test_empty_query_rejected() -> None:
    result = run_sql_query.invoke({"sql": "   ;  "})
    assert result.startswith("SQL_ERROR")


# ----------------------------------------------------------------------------------
# Catalog-backed tools
# ----------------------------------------------------------------------------------

def test_list_warehouse_tables_names_all_four() -> None:
    result = list_warehouse_tables.invoke({})
    for table in ("orders", "customers", "products", "carriers"):
        assert table in result


def test_get_table_schema_tool_grounds_the_agent() -> None:
    result = get_table_schema.invoke({"table_name": "orders"})
    assert "late_delivery" in result
    assert "CREATE TABLE orders" in result


def test_get_table_schema_tool_unknown_table_self_repair_hint() -> None:
    result = get_table_schema.invoke({"table_name": "bogus"})
    assert result.startswith("SQL_ERROR")
    assert "orders" in result, "hint must list valid tables so the agent can retry"


def test_search_metadata_ranks_carriers_first_for_sla_question() -> None:
    result = search_metadata.invoke({"query": "penalty carrier sla"})
    first_ranked = next(
        line for line in result.splitlines() if line.startswith("1.")
    )
    assert "carriers" in first_ranked
    assert "Late Delivery" in result, "glossary must ride along with search results"
