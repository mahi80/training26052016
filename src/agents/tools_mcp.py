"""WHY THIS EXISTS
---------------
``@tool`` wrappers that let a worker agent reach company systems **over MCP**
instead of via Python imports. Each tool is a one-liner around
``src.mcp_servers.client.call_mcp_tool`` — the agent neither knows nor cares
that a subprocess and a JSON-RPC conversation happen underneath, exactly as
it never knew PageIndex internals behind ``search_contracts``.

Failures arrive as ``MCP_ERROR: ...`` strings (the client guarantees it), so
the ReAct loop can read the message, adjust, or report honestly.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def sap_purchase_orders(vendor: str = "", status: str = "") -> str:
    """Look up purchase orders in SAP, optionally filtered by vendor and status.

    vendor matches any part of the vendor name (e.g. 'SwiftShip'); status is
    one of: Open, In Transit, Delivered, Delayed. Leave both empty to list
    all POs. Returns a markdown table or 'MCP_ERROR: ...' on failure.
    """
    from src.mcp_servers.client import call_mcp_tool

    arguments: dict = {}
    if vendor:
        arguments["vendor"] = vendor
    if status:
        arguments["status"] = status
    return call_mcp_tool("sap", "get_purchase_orders", arguments)


@tool
def servicenow_incidents(carrier: str = "", state: str = "") -> str:
    """Look up incident tickets in ServiceNow, optionally filtered.

    carrier matches any part of the carrier name (e.g. 'SwiftShip'); state is
    one of: New, In Progress, Resolved. Leave both empty to list all open and
    closed tickets. Returns a markdown table or 'MCP_ERROR: ...' on failure.
    """
    from src.mcp_servers.client import call_mcp_tool

    arguments: dict = {}
    if carrier:
        arguments["carrier"] = carrier
    if state:
        arguments["state"] = state
    return call_mcp_tool("servicenow", "get_incidents", arguments)


@tool
def warehouse_query_via_mcp(sql: str) -> str:
    """Run one read-only SELECT against the warehouse — over MCP.

    Same rules as the in-process SQL tool: single SELECT/WITH statement,
    writes rejected, LIMIT 50 appended. Returns a markdown table,
    'SQL_ERROR: ...' for bad SQL, or 'MCP_ERROR: ...' for protocol failures.
    """
    from src.mcp_servers.client import call_mcp_tool

    return call_mcp_tool("sql", "query_warehouse", {"sql": sql})
