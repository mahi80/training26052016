"""WHY THIS EXISTS
---------------
Proof the A2A boundary works offline. The external agent is exercised
in-process via FastAPI's TestClient — same routes, same JSON shapes as the
live ``python -m src.a2a.external_agent`` service, no port needed. The
client tool's failure path is tested against a dead port: it must come back
as an ``A2A_ERROR:`` *string* (the errors-as-strings convention), never an
exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.a2a.external_agent import answer_carrier_question, app
from src.agents.tools_a2a import ask_carrier_agent, fetch_agent_card

client = TestClient(app)


def _send(text: str) -> dict:
    return client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": {"role": "user", "parts": [{"text": text}]}},
        },
    ).json()


# ---- Agent card (discovery) --------------------------------------------------

def test_agent_card_has_name_and_skills() -> None:
    card = client.get("/.well-known/agent.json").json()
    assert card["name"] == "SwiftShip Carrier Partner Agent"
    assert {s["id"] for s in card["skills"]} == {"fleet_status", "pickup_capacity", "hub_congestion"}


# ---- message/send happy paths ------------------------------------------------

def test_message_send_about_pickup_capacity() -> None:
    body = _send("How many same-day pickup slots are free in Western Europe?")
    text = body["result"]["message"]["parts"][0]["text"]
    assert "pickup" in text.lower()
    assert "Western Europe" in text


def test_message_send_about_fleet_delay_mentions_rotterdam_outage() -> None:
    body = _send("Is the SwiftShip fleet running late anywhere?")
    text = body["result"]["message"]["parts"][0]["text"]
    assert "Rotterdam" in text


def test_unmatched_question_gets_polite_topic_list() -> None:
    assert "rephrase" in answer_carrier_question("What is your CEO's favorite color?")


# ---- JSON-RPC error paths ----------------------------------------------------

def test_unknown_method_returns_jsonrpc_error() -> None:
    body = client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {}}).json()
    assert body["error"]["code"] == -32601


def test_empty_message_parts_returns_jsonrpc_error() -> None:
    body = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 3, "method": "message/send", "params": {"message": {"parts": []}}},
    ).json()
    assert body["error"]["code"] == -32602


# ---- Client tool failure contract: strings, never exceptions ------------------

def test_ask_carrier_agent_dead_port_returns_a2a_error_string(monkeypatch) -> None:
    monkeypatch.setenv("A2A_BASE_URL", "http://127.0.0.1:59999")
    out = ask_carrier_agent.invoke({"question": "fleet status?"})
    assert out.startswith("A2A_ERROR:")
    assert "python -m src.a2a.external_agent" in out  # tells the user how to fix it


def test_fetch_agent_card_dead_port_returns_a2a_error_string(monkeypatch) -> None:
    monkeypatch.setenv("A2A_BASE_URL", "http://127.0.0.1:59999")
    assert fetch_agent_card().startswith("A2A_ERROR:")
