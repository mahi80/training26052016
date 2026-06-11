"""WHY THIS EXISTS
---------------
Proof the MCP layer works offline. The fast tests call the servers' plain
functions directly — no subprocess, no protocol — because the tools were
deliberately defined as module-level functions *before* being registered on
FastMCP. One slow test does a genuine stdio round-trip (spawn server, speak
MCP, get an answer) to prove the protocol path on this machine; deselect it
with ``-m "not slow"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.mcp_servers.client import call_mcp_tool, list_mcp_tools
from src.mcp_servers.sap_server import get_po, get_purchase_orders, list_vendors
from src.mcp_servers.servicenow_server import get_incident, get_incidents
from src.mcp_servers.sql_server import list_tables, query_warehouse


# ---- SAP mock: direct function calls (no protocol) -------------------------

def test_sap_purchase_orders_returns_markdown_table() -> None:
    out = get_purchase_orders()
    assert "| po_number |" in out
    assert "PO-2025-0041" in out


def test_sap_vendor_filter_is_case_insensitive_substring() -> None:
    out = get_purchase_orders(vendor="swiftship")
    assert "SwiftShip Express" in out
    assert "Atlas" not in out


def test_sap_status_filter_combines_with_vendor() -> None:
    out = get_purchase_orders(vendor="SwiftShip", status="Delayed")
    assert out.startswith("3 purchase order(s):")


def test_sap_no_match_lists_known_vendors() -> None:
    out = get_purchase_orders(vendor="DHL")
    assert "No purchase orders matched" in out
    assert "SwiftShip Express" in out  # the helpful hint


def test_sap_get_po_found_and_missing() -> None:
    assert "Same Day freight lane" in get_po("PO-2025-0041")
    assert "No purchase order found" in get_po("PO-9999-9999")


def test_sap_list_vendors_has_all_six_contract_parties() -> None:
    out = list_vendors()
    assert out.count("- ") == 6


# ---- ServiceNow mock: direct function calls ---------------------------------

def test_servicenow_incidents_filter_by_carrier_and_state() -> None:
    out = get_incidents(carrier="SwiftShip", state="In Progress")
    assert "2 incident(s):" in out
    assert "INC0010001" in out


def test_servicenow_get_incident_found_and_missing() -> None:
    assert "Missed Same Day pickup" in get_incident("INC0010001")
    assert "No incident found" in get_incident("INC0099999")


# ---- SQL shim: guardrails are inherited, not re-implemented -----------------

def test_sql_server_select_works() -> None:
    out = query_warehouse("SELECT COUNT(*) AS n FROM orders")
    assert "SQL_ERROR" not in out
    assert "n" in out


def test_sql_server_rejects_writes_via_inherited_guardrails() -> None:
    out = query_warehouse("DROP TABLE orders")
    assert out.startswith("SQL_ERROR:")


def test_sql_server_list_tables_mentions_orders() -> None:
    assert "orders" in list_tables()


# ---- Client error contract: strings, never exceptions -----------------------

def test_client_unknown_server_returns_mcp_error_string() -> None:
    out = call_mcp_tool("jira", "anything")
    assert out.startswith("MCP_ERROR:")
    assert "sap" in out  # names the known servers


def test_client_list_unknown_server_returns_mcp_error_string() -> None:
    assert list_mcp_tools("jira").startswith("MCP_ERROR:")


# ---- One genuine protocol round-trip (subprocess spawn ~1-2 s) ---------------

@pytest.mark.slow
def test_stdio_round_trip_to_sap_server() -> None:
    out = call_mcp_tool("sap", "list_vendors")
    assert "SwiftShip Express" in out, f"expected vendor list, got: {out!r}"
