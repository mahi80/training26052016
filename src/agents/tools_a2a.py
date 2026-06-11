"""WHY THIS EXISTS
---------------
Our side of the A2A conversation: a ``@tool`` that lets an internal worker
ask the *external* SwiftShip partner agent a question over the network.

From the worker's point of view this is just another tool — same shape as
``run_sql_query`` or ``search_contracts``. That symmetry is deliberate: "ask
another company's agent" should be no more exotic to the LLM than "query the
warehouse". The protocol plumbing (JSON-RPC message/send, agent-card
discovery) lives here, invisible to the agent.

Network failures follow the repo's errors-as-strings convention
(``A2A_ERROR: ...``) — the agent reads the message, tells the user honestly,
and moves on; an exception would kill the whole graph run.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

DEFAULT_A2A_BASE_URL = "http://127.0.0.1:8001"
_TIMEOUT_SECONDS = 5.0


def _base_url() -> str:
    return os.getenv("A2A_BASE_URL", DEFAULT_A2A_BASE_URL).rstrip("/")


@tool
def ask_carrier_agent(question: str) -> str:
    """Ask SwiftShip's own partner agent about LIVE carrier operations.

    Use for current fleet status, today's pickup capacity, or hub congestion
    — information only the carrier itself has. Do NOT use for historical
    data, contracts, or our own warehouse; other tools cover those.
    Returns the partner agent's text answer, or 'A2A_ERROR: ...' when the
    partner service is unreachable.
    """
    import httpx

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": {"role": "user", "parts": [{"text": question}]}},
    }
    url = _base_url()
    try:
        response = httpx.post(f"{url}/a2a", json=payload, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:  # noqa: BLE001 — by contract, errors become strings
        return (
            f"A2A_ERROR: carrier agent not reachable at {url} "
            f"({type(exc).__name__}). Start it with: python -m src.a2a.external_agent"
        )
    if "error" in body:
        return f"A2A_ERROR: partner agent rejected the request: {body['error'].get('message', body['error'])}"
    parts = body.get("result", {}).get("message", {}).get("parts", [])
    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    return text or "A2A_ERROR: partner agent returned an empty answer."


def fetch_agent_card(base_url: str | None = None) -> str:
    """Fetch a partner agent's card (A2A discovery) — who is it, what can it do.

    Not an LLM tool; used by lab10 and the manual to show discovery working.
    """
    import httpx

    url = (base_url or _base_url()).rstrip("/")
    try:
        response = httpx.get(f"{url}/.well-known/agent.json", timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        card = response.json()
    except Exception as exc:  # noqa: BLE001
        return (
            f"A2A_ERROR: could not fetch agent card from {url} "
            f"({type(exc).__name__}). Start it with: python -m src.a2a.external_agent"
        )
    skills = ", ".join(s.get("name", "?") for s in card.get("skills", []))
    return f"{card.get('name', '?')} v{card.get('version', '?')} — skills: {skills}"
