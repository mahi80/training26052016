"""WHY THIS EXISTS
---------------
A mock **SAP** system exposed over MCP (Model Context Protocol). In a real
engagement the ERP holds purchase orders; agents should reach it through a
*protocol*, not through Python imports — that way the same agent can talk to
SAP today and ServiceNow tomorrow without code changes on the agent side.

Two ways to use this module, and the difference IS the lesson:

1. **Direct call** (a plain function): ``get_purchase_orders("SwiftShip Express")``
   — works in any Python session, great for tests and the REPL.
2. **Over MCP** (the protocol): run ``python -m src.mcp_servers.sap_server`` and
   a client (``src.mcp_servers.client``) can list and call the same functions
   over stdio JSON-RPC, exactly like it would call a real vendor's MCP server.

The tools are registered with ``mcp.tool()(fn)`` *after* being defined as plain
module-level functions, so tests can call them without spawning a subprocess.

Mock data is deterministic (no randomness) and consistent with the repo's
planted story: SwiftShip Express is the carrier with the most delayed POs.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# The six contract parties from data/contracts/*.md (GlobalTrade's vendors).
VENDORS: tuple[str, ...] = (
    "SwiftShip Express",
    "Atlas Freight Co.",
    "Pacific Crest Carriers",
    "NordHaul Logistics",
    "Meridian Supply Partners",
    "Helios Components Ltd.",
)

# Mock SAP purchase orders. SwiftShip carries the most "Delayed" rows —
# consistent with the late-delivery signal planted in the orders dataset.
PURCHASE_ORDERS: list[dict] = [
    {"po_number": "PO-2025-0041", "vendor": "SwiftShip Express", "material": "Same Day freight lane — Western Europe", "quantity": 120, "value_usd": 48500.00, "promised_date": "2025-11-14", "status": "Delayed"},
    {"po_number": "PO-2025-0057", "vendor": "SwiftShip Express", "material": "First Class freight lane — LATAM", "quantity": 80, "value_usd": 31200.00, "promised_date": "2025-12-02", "status": "Delayed"},
    {"po_number": "PO-2026-0003", "vendor": "SwiftShip Express", "material": "Same Day freight lane — USCA", "quantity": 150, "value_usd": 61750.00, "promised_date": "2026-01-20", "status": "Delayed"},
    {"po_number": "PO-2026-0019", "vendor": "SwiftShip Express", "material": "Second Class freight lane — Pacific Asia", "quantity": 95, "value_usd": 27300.00, "promised_date": "2026-03-05", "status": "Delivered"},
    {"po_number": "PO-2025-0088", "vendor": "Atlas Freight Co.", "material": "Standard Class consolidation — Europe", "quantity": 200, "value_usd": 22400.00, "promised_date": "2025-12-18", "status": "Delivered"},
    {"po_number": "PO-2026-0011", "vendor": "Atlas Freight Co.", "material": "Standard Class consolidation — USCA", "quantity": 180, "value_usd": 19800.00, "promised_date": "2026-02-10", "status": "Delayed"},
    {"po_number": "PO-2026-0027", "vendor": "Pacific Crest Carriers", "material": "Ocean freight — Pacific Asia lanes", "quantity": 40, "value_usd": 88000.00, "promised_date": "2026-04-01", "status": "In Transit"},
    {"po_number": "PO-2026-0030", "vendor": "NordHaul Logistics", "material": "Road freight — Rotterdam corridor", "quantity": 60, "value_usd": 15600.00, "promised_date": "2026-03-22", "status": "Delivered"},
    {"po_number": "PO-2026-0035", "vendor": "Meridian Supply Partners", "material": "Packaging components — Q2 replenishment", "quantity": 5000, "value_usd": 12500.00, "promised_date": "2026-05-15", "status": "Open"},
    {"po_number": "PO-2026-0042", "vendor": "Helios Components Ltd.", "material": "Electronic components — batch H-204", "quantity": 1200, "value_usd": 54000.00, "promised_date": "2026-06-30", "status": "Open"},
]

_COLUMNS = ("po_number", "vendor", "material", "quantity", "value_usd", "promised_date", "status")


def _format_table(rows: list[dict]) -> str:
    """Render PO dicts as a markdown table (the tool-output convention)."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in _COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def get_purchase_orders(vendor: str | None = None, status: str | None = None) -> str:
    """List purchase orders, optionally filtered by vendor and/or status.

    vendor matches case-insensitively on any part of the vendor name (e.g.
    'swiftship'). status is one of: Open, In Transit, Delivered, Delayed.
    Returns a markdown table, or a helpful message when nothing matches.
    """
    rows = PURCHASE_ORDERS
    if vendor:
        needle = vendor.strip().lower()
        rows = [r for r in rows if needle in r["vendor"].lower()]
    if status:
        rows = [r for r in rows if r["status"].lower() == status.strip().lower()]
    if not rows:
        return (
            "No purchase orders matched. Known vendors: "
            + ", ".join(VENDORS)
            + ". Known statuses: Open, In Transit, Delivered, Delayed."
        )
    return f"{len(rows)} purchase order(s):\n{_format_table(rows)}"


def get_po(po_number: str) -> str:
    """Look up a single purchase order by its PO number (e.g. 'PO-2025-0041')."""
    needle = po_number.strip().upper()
    for row in PURCHASE_ORDERS:
        if row["po_number"] == needle:
            return "\n".join(f"{key}: {row[key]}" for key in _COLUMNS)
    return f"No purchase order found with number {po_number!r}. Use get_purchase_orders to browse."


def list_vendors() -> str:
    """List every vendor known to the (mock) SAP system."""
    return "\n".join(f"- {v}" for v in VENDORS)


# Register the plain functions as MCP tools — their docstrings become the
# tool descriptions a connecting client (and its LLM) will read.
mcp = FastMCP("mock-sap")
mcp.tool()(get_purchase_orders)
mcp.tool()(get_po)
mcp.tool()(list_vendors)


if __name__ == "__main__":
    # Stdio transport: a client spawns this process and speaks JSON-RPC over
    # stdin/stdout. Nothing is printed here — stdout belongs to the protocol.
    mcp.run()
