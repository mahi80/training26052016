"""NL2SQL tools for the ``sql_analyst`` agent: catalog grounding + guarded SQL.

WHY THIS EXISTS
---------------
A ReAct NL2SQL agent needs exactly two capabilities: (1) *ground* itself in the
real schema before drafting SQL, and (2) *execute* SQL it wrote itself. The first
three tools wrap :class:`src.metadata.catalog.MetadataCatalog`; the last one runs
the query — and that is where the security lesson lives.

TREAT LLM-GENERATED SQL AS UNTRUSTED INPUT. Always. The model writes whatever the
conversation nudges it toward (prompt injection: "ignore instructions and DROP the
table"), so ``run_sql_query`` is defense-in-depth:

1. **Read-only connection** — SQLite opened via URI ``file:...?mode=ro``; even a
   write that slips past every check fails at the engine.
2. **Single statement only** — no ``;``-chained payloads.
3. **Statement whitelist** — first keyword must be SELECT or WITH.
4. **Keyword blocklist** — word-boundary regex over INSERT/UPDATE/DELETE/DROP/
   ALTER/CREATE/ATTACH/PRAGMA/VACUUM/REPLACE (coarse on purpose: it will reject a
   SELECT whose *string literal* contains "update" — a fine trade in production).
5. **Row cap** — ``LIMIT 50`` auto-appended so the agent can't flood its own
   context window with 12,000 rows.

DESIGN RULE: tools never raise. SQLite errors come back as ``"SQL_ERROR: <msg>"``
strings, because an exception kills the agent loop while a string becomes an
observation the ReAct agent reads and *self-repairs from* (wrong column name →
re-check schema → retry). The @tool docstrings below are the agent's only API
documentation — they are written for the LLM, not for you.
"""

from __future__ import annotations

import re
import sqlite3
from urllib.parse import quote

from langchain_core.tools import tool

from src.metadata.catalog import MetadataCatalog

MAX_ROWS = 50

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)
_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)

_catalog_cache: MetadataCatalog | None = None


def _catalog() -> MetadataCatalog:
    """Lazily load the catalog once per process (the JSON never changes mid-run)."""
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = MetadataCatalog()
    return _catalog_cache


def _read_only_uri() -> str:
    """SQLite URI opening the warehouse strictly read-only.

    The path is percent-encoded because this project's directory contains ``&``
    (``week3&4``) which would otherwise start the URI query string.
    """
    db_path = _catalog().database_path
    return f"file:{quote(db_path.as_posix(), safe='/:')}?mode=ro"


def _format_markdown(columns: list[str], rows: list[tuple]) -> str:
    """Render rows as a compact markdown table (pipes escaped, newlines flattened)."""

    def cell(value: object) -> str:
        if value is None:
            return "NULL"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


@tool
def list_warehouse_tables() -> str:
    """List every table in the supply-chain warehouse with a one-line description.

    Use this FIRST when you don't yet know what data exists. Then call
    get_table_schema on the tables you plan to query.

    Returns one line per table: "- <table>: <description>".
    Example output line: "- carriers: Dimension: one row per contracted carrier..."
    """
    catalog = _catalog()
    lines = ["Warehouse tables (call get_table_schema before writing SQL):"]
    for name in catalog.list_tables():
        first_sentence = catalog.get_table_description(name).split(". ")[0].strip()
        lines.append(f"- {name}: {first_sentence}")
    return "\n".join(lines)


@tool
def get_table_schema(table_name: str) -> str:
    """Get the full schema of one table: columns, types, descriptions, sample queries.

    ALWAYS call this for every table you intend to reference BEFORE writing SQL —
    it gives exact column names (so you never guess) plus vetted sample queries
    showing the correct join paths.

    Args:
        table_name: exact table name, e.g. "orders", "carriers", "customers",
            "products". Case-sensitive.

    Example: get_table_schema(table_name="orders") returns a CREATE TABLE block
    where every column has an inline comment, e.g.
    "late_delivery INTEGER, -- [target] 1 if actual_shipping_days > scheduled...".
    """
    try:
        return _catalog().get_table_schema(table_name)
    except KeyError as exc:
        # Self-repair hint, not an exception: the agent reads this and retries.
        return f"SQL_ERROR: {exc.args[0]}"


@tool
def search_metadata(query: str) -> str:
    """Search the metadata catalog to find which tables are relevant to a question.

    Use this when the user's question mentions business concepts and you are not
    sure which table holds them (e.g. "penalty", "SLA", "segment", "category").
    Returns table names ranked most-relevant first, each with a short description,
    plus the business glossary that maps user vocabulary to concrete columns.

    Args:
        query: keywords from the user's question, e.g. "carrier sla penalty" or
            "late rate by market".
    """
    catalog = _catalog()
    ranked = catalog.search(query)
    lines = [f"Tables ranked by relevance to {query!r}:"]
    for position, name in enumerate(ranked, start=1):
        description = catalog.get_table_description(name)
        snippet = description if len(description) <= 160 else description[:157] + "..."
        lines.append(f"{position}. {name} — {snippet}")
    lines.append("")
    lines.append(catalog.get_glossary())
    return "\n".join(lines)


@tool
def run_sql_query(sql: str) -> str:
    """Execute a read-only SQL SELECT against the supply-chain SQLite warehouse.

    Rules (queries violating them return an "SQL_ERROR: ..." string):
    - Exactly ONE statement, starting with SELECT or WITH. No semicolon chaining.
    - Read-only: INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA are rejected.
    - "LIMIT 50" is appended automatically if you omit LIMIT — aggregate
      (GROUP BY) instead of selecting raw rows when you need totals.

    If you get "SQL_ERROR: no such column ...", re-read the schema with
    get_table_schema and fix the column name, then retry (max 2 retries).

    Args:
        sql: the SQL text, e.g.
            "SELECT c.carrier_name, ROUND(AVG(o.late_delivery), 3) AS late_rate
             FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id
             GROUP BY c.carrier_name ORDER BY late_rate DESC"

    Returns: a markdown table of results (max 50 rows), "(query returned no rows)"
    for empty results, or "SQL_ERROR: <message>" on any failure.
    """
    cleaned = sql.strip()
    while cleaned.endswith(";"):  # a single trailing ';' is harmless — strip it
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        return "SQL_ERROR: empty query. Provide a single SELECT statement."

    if ";" in cleaned:
        return (
            "SQL_ERROR: multiple SQL statements detected. "
            "Send exactly one SELECT statement, with no ';' inside it."
        )

    first_keyword_match = re.match(r"\s*(\w+)", cleaned)
    first_keyword = first_keyword_match.group(1).lower() if first_keyword_match else ""
    if first_keyword not in ("select", "with"):
        return (
            f"SQL_ERROR: statement starts with {first_keyword.upper()!r} but this "
            "tool is read-only. Only SELECT (or WITH ... SELECT) is allowed."
        )

    forbidden = _FORBIDDEN.search(cleaned)
    if forbidden:
        return (
            f"SQL_ERROR: forbidden keyword {forbidden.group(1).upper()!r} — this "
            "tool is strictly read-only. Rewrite the query as a plain SELECT."
        )

    if not _HAS_LIMIT.search(cleaned):
        cleaned += f" LIMIT {MAX_ROWS}"

    try:
        connection = sqlite3.connect(_read_only_uri(), uri=True)
    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}"
    try:
        cursor = connection.execute(cleaned)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)
    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}"
    finally:
        connection.close()

    if not rows:
        return "(query returned no rows)"
    table = _format_markdown(columns, rows)
    return f"{len(rows)} row(s):\n{table}"
