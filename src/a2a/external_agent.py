"""WHY THIS EXISTS
---------------
The **external agent** from the architecture diagram: SwiftShip's own carrier
agent — another company's system. We do not import it, share state with it,
or see its internals; we send it a question over the network and get an
answer back. That boundary is the whole lesson of A2A (agent-to-agent):

- **MCP** connects an agent to *systems* (databases, ERPs, ticket queues).
- **A2A** connects an agent to *other agents* — each side keeps its own
  brain, tools, and data, and exposes only a conversational surface.

This service follows the A2A protocol's shape without pulling in a heavy SDK
(labeled honestly: "A2A-style"):

- ``GET /.well-known/agent.json`` — the *agent card*: who am I, what skills
  do I offer. Real A2A clients discover partners by fetching exactly this.
- ``POST /a2a`` — JSON-RPC 2.0 ``message/send``: one user message in, one
  agent message out.

The brain is **rule-based** (keyword matching over mock operations data), so
it runs with zero API keys — the point here is the wire, not the wit.

Run it:  python -m src.a2a.external_agent   (listens on 127.0.0.1:8001)
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

AGENT_CARD: dict = {
    "name": "SwiftShip Carrier Partner Agent",
    "description": (
        "SwiftShip Express's customer-facing operations agent. Answers "
        "questions about live fleet status, pickup capacity, and hub "
        "congestion for contracted partners."
    ),
    "version": "1.0.0",
    "url": "http://127.0.0.1:8001/a2a",
    "skills": [
        {"id": "fleet_status", "name": "Fleet status", "description": "Current fleet availability and known disruptions."},
        {"id": "pickup_capacity", "name": "Pickup capacity", "description": "Remaining same-day / next-day pickup slots by region."},
        {"id": "hub_congestion", "name": "Hub congestion", "description": "Congestion level at SwiftShip sorting hubs."},
    ],
}

# Mock operations data — what SwiftShip's agent knows and we don't.
CARRIER_OPS: dict[str, str] = {
    "fleet": (
        "Fleet status: 87% of line-haul capacity operating normally. Known "
        "disruption: Western Europe Same Day fleet running ~4h behind due to "
        "a hub outage in Rotterdam (since 2026-06-08)."
    ),
    "pickup": (
        "Pickup capacity today: USCA 14 same-day slots free, Western Europe "
        "0 same-day slots (book next-day), LATAM 6 same-day slots, Pacific "
        "Asia 9 same-day slots."
    ),
    "congestion": (
        "Hub congestion: Rotterdam RED (outage recovery), Memphis YELLOW "
        "(volume peak), Singapore GREEN, Sao Paulo GREEN."
    ),
}

_KEYWORD_ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("fleet", "disruption", "status", "delay", "late", "behind"), "fleet"),
    (("pickup", "capacity", "slot", "book", "collect"), "pickup"),
    (("hub", "congestion", "rotterdam", "memphis", "backlog"), "congestion"),
]


def answer_carrier_question(text: str) -> str:
    """Rule-based 'brain': match keywords, return the relevant ops report(s)."""
    lowered = text.lower()
    hits = [CARRIER_OPS[topic] for keywords, topic in _KEYWORD_ROUTES if any(k in lowered for k in keywords)]
    if not hits:
        return (
            "SwiftShip Partner Agent: I can answer questions about fleet "
            "status, pickup capacity, and hub congestion. Please rephrase "
            "your question around one of those topics."
        )
    # dict.fromkeys dedupes while preserving order (a question may hit 2 topics)
    return "SwiftShip Partner Agent: " + " ".join(dict.fromkeys(hits))


class JsonRpcRequest(BaseModel):
    """The subset of JSON-RPC 2.0 that ``message/send`` needs."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = {}


app = FastAPI(title="SwiftShip Carrier Partner Agent (external, A2A-style)")


@app.get("/.well-known/agent.json")
def agent_card() -> dict:
    """A2A discovery: the agent card tells callers who we are and what we do."""
    return AGENT_CARD


@app.post("/a2a")
def a2a_endpoint(request: JsonRpcRequest) -> dict:
    """JSON-RPC message/send: extract the text parts, answer, wrap the reply."""
    if request.method != "message/send":
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32601, "message": f"Unknown method {request.method!r}. Use 'message/send'."},
        }
    parts = request.params.get("message", {}).get("parts", [])
    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32602, "message": "message.parts must contain at least one {'text': ...} part."},
        }
    return {
        "jsonrpc": "2.0",
        "id": request.id,
        "result": {"message": {"role": "agent", "parts": [{"text": answer_carrier_question(text)}]}},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
