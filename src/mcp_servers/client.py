"""WHY THIS EXISTS
---------------
The **MCP client** — the dialer. Agent tools call ``call_mcp_tool("sap",
"get_purchase_orders", {...})`` and this module does the protocol work:

1. spawn the server as a subprocess (``python -m src.mcp_servers.sap_server``),
2. speak MCP JSON-RPC to it over stdin/stdout (initialize → call_tool),
3. return the tool's text result, then let the subprocess exit.

DESIGN CHOICES (read these — they are the lesson):

- **Sync wrapper over an async SDK.** The official ``mcp`` SDK is async; this
  whole repo is sync (``.invoke()`` everywhere). ``asyncio.run()`` per call
  keeps every caller simple. In production you'd keep sessions open and use
  ``langchain-mcp-adapters`` (async, pooled); one-subprocess-per-call costs
  ~1-2 s but needs zero session management — the right trade for teaching.
- **Errors are strings, never exceptions** (``MCP_ERROR: ...``), exactly like
  ``SQL_ERROR:`` in tools_sql.py — a ReAct agent reads the message and adapts;
  an exception would kill the whole graph run.
- **Registry, not URLs.** ``SERVERS`` maps a short name to the module to spawn.
  Swapping a mock for a real vendor server means editing one line here.
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.config import PROJECT_ROOT

# Short name -> module spawned with `python -m <module>` (stdio transport).
SERVERS: dict[str, str] = {
    "sap": "src.mcp_servers.sap_server",
    "servicenow": "src.mcp_servers.servicenow_server",
    "sql": "src.mcp_servers.sql_server",
}


def _server_params(module: str):
    """Build stdio spawn parameters for one server module.

    cwd + PYTHONPATH must point at the project root or the child process
    cannot ``import src`` — the classic Windows subprocess gotcha.
    """
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )


async def _call(module: str, tool: str, arguments: dict) -> str:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_server_params(module)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            texts = [c.text for c in result.content if getattr(c, "text", None)]
            if result.isError:
                return "MCP_ERROR: " + ("\n".join(texts) or f"tool {tool!r} failed")
            return "\n".join(texts) or "(tool returned no text content)"


async def _list(module: str) -> str:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_server_params(module)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            return "\n".join(
                f"- {t.name}: {(t.description or '').strip().splitlines()[0]}"
                for t in listing.tools
            )


def call_mcp_tool(server: str, tool: str, arguments: dict | None = None, timeout: float = 30.0) -> str:
    """Call one tool on one MCP server and return its text output.

    server is a key of ``SERVERS`` ('sap', 'servicenow', 'sql'). Never raises:
    every failure (unknown server, spawn failure, timeout, tool error) comes
    back as an ``MCP_ERROR: ...`` string the calling agent can read and react to.
    """
    module = SERVERS.get(server)
    if module is None:
        return f"MCP_ERROR: unknown server {server!r}. Known servers: {', '.join(SERVERS)}."
    try:
        return asyncio.run(asyncio.wait_for(_call(module, tool, arguments or {}), timeout))
    except Exception as exc:  # noqa: BLE001 — by contract, errors become strings
        return f"MCP_ERROR: {type(exc).__name__}: {exc}"


def list_mcp_tools(server: str, timeout: float = 30.0) -> str:
    """List the tools an MCP server offers (name + first docstring line)."""
    module = SERVERS.get(server)
    if module is None:
        return f"MCP_ERROR: unknown server {server!r}. Known servers: {', '.join(SERVERS)}."
    try:
        return asyncio.run(asyncio.wait_for(_list(module), timeout))
    except Exception as exc:  # noqa: BLE001
        return f"MCP_ERROR: {type(exc).__name__}: {exc}"
