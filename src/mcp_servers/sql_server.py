"""WHY THIS EXISTS
---------------
The teaching punchline of the MCP chapter: this "SQL MCP server" contains
**zero new query logic**. It is a thin protocol shim over the *exact same*
guard-railed functions the sql_analyst agent already uses in-process
(``src/agents/tools_sql.py``) — SELECT-only, single statement, keyword
blocklist, LIMIT 50, errors as ``SQL_ERROR:`` strings. Same function, now
reachable over a protocol; the guardrails come along for free.

That is the whole point of MCP: you don't rewrite capabilities per consumer,
you put a standard wire format in front of the ones you trust.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def query_warehouse(sql: str) -> str:
    """Run one read-only SELECT against the supply-chain SQLite warehouse.

    Same rules as the in-process tool: exactly one statement starting with
    SELECT or WITH; writes are rejected; LIMIT 50 is appended if missing.
    Returns a markdown table or an 'SQL_ERROR: ...' string.
    """
    from src.agents.tools_sql import run_sql_query

    # .invoke because run_sql_query is a LangChain @tool, not a bare function.
    return run_sql_query.invoke({"sql": sql})


def list_tables() -> str:
    """List the warehouse tables (orders, customers, products, carriers)."""
    from src.agents.tools_sql import list_warehouse_tables

    return list_warehouse_tables.invoke({})


mcp = FastMCP("supply-chain-sql")
mcp.tool()(query_warehouse)
mcp.tool()(list_tables)


if __name__ == "__main__":
    mcp.run()
