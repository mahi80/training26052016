"""WHY THIS EXISTS
---------------
A mock **ServiceNow** ITSM exposed over MCP. Operations teams file incident
tickets when carriers miss pickups or hubs congest; an agent that can read
those tickets answers "is something wrong with SwiftShip *right now*?" — a
question no historical dataset or contract can answer.

Same dual-use design as ``sap_server``: plain functions for tests and the
REPL, registered on a ``FastMCP`` server for protocol access. See the
sap_server docstring for the longer story.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# Mock incident tickets. SwiftShip again has the worst run — consistent with
# the planted signal in the orders data and the SAP mock.
INCIDENTS: list[dict] = [
    {"number": "INC0010001", "carrier": "SwiftShip Express", "short_description": "Missed Same Day pickup window — Western Europe hub", "priority": "P1", "state": "In Progress", "opened_at": "2026-06-08"},
    {"number": "INC0010002", "carrier": "SwiftShip Express", "short_description": "Tracking feed outage: no scan events for 6 hours", "priority": "P2", "state": "In Progress", "opened_at": "2026-06-09"},
    {"number": "INC0010003", "carrier": "SwiftShip Express", "short_description": "Late tender acceptance on USCA Same Day lane", "priority": "P3", "state": "New", "opened_at": "2026-06-10"},
    {"number": "INC0010004", "carrier": "Atlas Freight Co.", "short_description": "Consolidation center cutoff missed (>48h delay risk)", "priority": "P2", "state": "Resolved", "opened_at": "2026-05-28"},
    {"number": "INC0010005", "carrier": "Pacific Crest Carriers", "short_description": "Port congestion advisory — Pacific Asia lanes", "priority": "P3", "state": "In Progress", "opened_at": "2026-06-05"},
    {"number": "INC0010006", "carrier": "NordHaul Logistics", "short_description": "Control Tower reporting delay (daily file late)", "priority": "P4", "state": "Resolved", "opened_at": "2026-05-20"},
    {"number": "INC0010007", "carrier": "SwiftShip Express", "short_description": "Damaged shipment claim — pending carrier response", "priority": "P3", "state": "Resolved", "opened_at": "2026-05-15"},
    {"number": "INC0010008", "carrier": "Atlas Freight Co.", "short_description": "EDI invoice mismatch for May consolidations", "priority": "P4", "state": "New", "opened_at": "2026-06-07"},
]

_COLUMNS = ("number", "carrier", "short_description", "priority", "state", "opened_at")


def _format_table(rows: list[dict]) -> str:
    """Render incident dicts as a markdown table (the tool-output convention)."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in _COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def get_incidents(carrier: str | None = None, state: str | None = None) -> str:
    """List incident tickets, optionally filtered by carrier and/or state.

    carrier matches case-insensitively on any part of the carrier name (e.g.
    'swiftship'). state is one of: New, In Progress, Resolved.
    Returns a markdown table, or a helpful message when nothing matches.
    """
    rows = INCIDENTS
    if carrier:
        needle = carrier.strip().lower()
        rows = [r for r in rows if needle in r["carrier"].lower()]
    if state:
        rows = [r for r in rows if r["state"].lower() == state.strip().lower()]
    if not rows:
        return (
            "No incidents matched. Known states: New, In Progress, Resolved. "
            "Try filtering by carrier name only, or omit filters to see all."
        )
    return f"{len(rows)} incident(s):\n{_format_table(rows)}"


def get_incident(number: str) -> str:
    """Look up a single incident ticket by number (e.g. 'INC0010001')."""
    needle = number.strip().upper()
    for row in INCIDENTS:
        if row["number"] == needle:
            return "\n".join(f"{key}: {row[key]}" for key in _COLUMNS)
    return f"No incident found with number {number!r}. Use get_incidents to browse."


mcp = FastMCP("mock-servicenow")
mcp.tool()(get_incidents)
mcp.tool()(get_incident)


if __name__ == "__main__":
    mcp.run()
