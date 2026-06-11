"""WHY THIS EXISTS
=================
LAB 10 — Production Layers: MCP, A2A, the Validator Gate, and Human Review

Labs 04-09 built a working multi-agent system — but a *closed* one: every tool
was a Python import, every answer shipped unreviewed. Production systems live
behind boundaries: company systems are reached over protocols (MCP), other
companies' agents over the network (A2A), and answers pass a quality gate that
can pause the graph for a human (interrupt/resume). This lab runs each new
layer in isolation; BUILD_MANUAL.md chapters 5-13 assemble them end to end.

LEARNING OBJECTIVES
-------------------
- Call the same mock-SAP function directly AND through a real MCP stdio
  round-trip — and explain what the protocol boundary buys
- Speak A2A-style JSON-RPC to the external carrier agent and read its agent card
- Predict the validator's verdict for good, short, and error-leaking answers
- Pause a real compiled graph at human_review with ``interrupt()``, inspect the
  payload, and resume it twice: approve, then edit
- Read the v2 topology and name the edge v1 did not have

DURATION: ~120 minutes
PREREQUISITES: Lab 09 (the v1 graph); ``pip install -r requirements.txt`` rerun
  after the v2 update (adds fastapi, httpx, mcp).
API KEY: needed ONLY for section 3b (LLM-as-judge). Everything else is offline.
"""

# %% ── Banner: paths, offline status ───────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles often default to cp1252 — force UTF-8 so emoji/box chars print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive cells: run from the project root
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import is_offline  # noqa: E402

OFFLINE = is_offline()
print("=" * 72)
print("LAB 10 — Production layers: MCP, A2A, validator, human review")
print(f"OFFLINE mode : {OFFLINE}")
print("API key needed: only for section 3b (LLM-as-judge).")
print("Sections 1-2, 3a, 4-5 (MCP, A2A, heuristic gate, HITL) run offline.")
print("=" * 72)

# %% ── 1a. MCP, take one: the tool as a plain function ──────────────────────────
# The mock SAP server's tools are ordinary module-level functions. Calling one
# directly is just Python — no protocol, no subprocess. Remember this output;
# you are about to get the identical answer the hard (= production) way.
from src.mcp_servers.sap_server import get_purchase_orders  # noqa: E402

print(get_purchase_orders(vendor="SwiftShip Express", status="Delayed"))

# %% ── 1b. MCP, take two: the same tool over the protocol ───────────────────────
# call_mcp_tool spawns `python -m src.mcp_servers.sap_server` as a subprocess,
# performs the MCP handshake over stdin/stdout, calls the tool, and tears down.
# Same answer, ~1-2 s slower — and that slowness buys decoupling: the server
# could be rewritten in another language or hosted by the vendor tomorrow, and
# this calling code would not change ONE character.
from src.mcp_servers.client import call_mcp_tool, list_mcp_tools  # noqa: E402

print("Tools the SAP server advertises:")
print(list_mcp_tools("sap"))
print()
print(call_mcp_tool("sap", "get_purchase_orders", {"vendor": "SwiftShip Express", "status": "Delayed"}))

# %% ── 1c. MCP error contract: failures are strings, not exceptions ─────────────
# A bogus server name comes back as "MCP_ERROR: ..." — readable by a ReAct
# agent mid-conversation. Compare tools_sql.py's SQL_ERROR convention: same
# idea, one layer further out.
print(call_mcp_tool("jira", "anything"))

# %% ── 2. A2A: talk to ANOTHER COMPANY'S agent ──────────────────────────────────
# The external carrier agent is a separate FastAPI service (run it live with:
#   python -m src.a2a.external_agent
# ). For the lab we mount it in-process with TestClient — the same routes and
# JSON shapes, no port needed. Step one of any A2A handshake: fetch the agent
# card to learn who you are talking to and what it can do.
from fastapi.testclient import TestClient  # noqa: E402

from src.a2a.external_agent import app as carrier_app  # noqa: E402

carrier = TestClient(carrier_app)
print("Agent card:", carrier.get("/.well-known/agent.json").json()["name"])

response = carrier.post(
    "/a2a",
    json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": {"role": "user", "parts": [{"text": "Any same-day pickup slots left in Western Europe?"}]}},
    },
)
print("Partner agent replies:", response.json()["result"]["message"]["parts"][0]["text"])

# %% ── 3a. The validator gate: heuristic judge (offline) ────────────────────────
# Before an answer ships, the validator rules complete vs needs_human. The
# offline heuristic is deliberately dumb-but-explainable: empty, too-short, or
# tool-error-leaking answers go to a human. Predict each verdict before running.
from src.agents.validator import heuristic_verdict  # noqa: E402

cases = [
    ("good", "SwiftShip Express has the highest late rate at 34.2% across 2,841 orders."),
    ("short", "SwiftShip."),
    ("leaky", "I checked but got MCP_ERROR: TimeoutError under the hood."),
]
for label, answer in cases:
    ruling = heuristic_verdict("Which carrier is worst?", answer)
    print(f"{label:>6}: {ruling.verdict:<11} — {ruling.reason}")

# %% ── 3b. (online) The validator gate: LLM-as-judge ────────────────────────────
# With a key, make_validator_node() returns a structured-output judge instead.
# Same node contract, smarter rulings — and a new failure mode (the judge can
# be wrong), which is why production stacks run the heuristic FIRST.
if OFFLINE:
    print("OFFLINE — skipping LLM-as-judge. Set a key in .env to try this cell.")
else:
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

    from src.agents.validator import make_validator_node  # noqa: E402

    judge = make_validator_node()
    out = judge(
        {
            "messages": [
                HumanMessage(content="Which carrier is worst?"),
                AIMessage(content="Probably one of them is bad sometimes.", name="sql_analyst"),
            ],
            "next_agent": "FINISH",
            "verdict": "",
            "verdict_reason": "",
        }
    )
    print("LLM judge says:", out)

# %% ── 4. Human review: pause a REAL graph, then resume it twice ────────────────
# The full v2 graph with scripted parts (the same dependency-injection trick as
# tests/test_graph_v2_hitl.py): a router that always picks sql_analyst then
# FINISH, a worker that answers instantly, a validator that always says
# needs_human. What is NOT scripted: the graph itself, the checkpointer, and
# the interrupt/resume machinery — that is all real.
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.agents.supervisor_v2 import RouterV2  # noqa: E402
from src.graph_v2 import build_graph_v2  # noqa: E402
from src.state_v2 import WORKERS_V2  # noqa: E402


class ScriptedRouterLLM:
    """Replays a fixed routing script (see tests/test_graph_v2_hitl.py)."""

    def __init__(self, script: list[str]) -> None:
        self._script, self._calls = list(script), 0

    def with_structured_output(self, schema: object, **kwargs: object) -> "ScriptedRouterLLM":
        return self

    def invoke(self, _messages: object, config: object = None) -> RouterV2:
        route = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return RouterV2(next=route, reason="scripted")


def scripted_worker(name: str):
    def node(state: dict) -> dict:
        return {"messages": [AIMessage(content=f"{name}: draft answer (unreviewed).", name=name)]}

    return node


app = build_graph_v2(
    llm=ScriptedRouterLLM(["sql_analyst", "FINISH"]),
    workers={name: scripted_worker(name) for name in WORKERS_V2},
    validator=lambda state: {"verdict": "needs_human", "verdict_reason": "lab: always review"},
    checkpointer=MemorySaver(),  # interrupts REQUIRE a checkpointer
)
config = {"recursion_limit": 25, "configurable": {"thread_id": "lab10-hitl"}}

result = app.invoke(
    {"messages": [HumanMessage(content="Count orders per carrier.")], "next_agent": "", "verdict": "", "verdict_reason": ""},
    config=config,
)
print("Graph paused?      ", "__interrupt__" in result)
print("Paused at node:    ", app.get_state(config).next)
print("Payload for human: ", result["__interrupt__"][0].value["draft_answer"])

# %% ── 4b. Resume #1: approve as-is ─────────────────────────────────────────────
# Command(resume=X) makes the interrupt() call RETURN X inside human_review —
# the graph continues from exactly where it stopped, same thread_id.
resumed = app.invoke(Command(resume={"action": "approve"}), config=config)
print("Verdict after resume:", resumed["verdict"], "—", resumed["verdict_reason"])

# %% ── 4c. Resume #2: a fresh thread, this time the human EDITS the answer ───────
config2 = {"recursion_limit": 25, "configurable": {"thread_id": "lab10-hitl-edit"}}
app.invoke(
    {"messages": [HumanMessage(content="Count orders per carrier.")], "next_agent": "", "verdict": "", "verdict_reason": ""},
    config=config2,
)
edited = app.invoke(
    Command(resume={"action": "edit", "text": "Reviewed answer: 2,841 orders across 4 carriers."}),
    config=config2,
)
final = edited["messages"][-1]
print(f"Final answer by {final.name!r}: {final.content}")

# %% ── 5. Read the v2 topology ───────────────────────────────────────────────────
# One question to answer out loud before moving on: where does FINISH point in
# v2, and where did it point in v1's ascii_graph()? That one edge is the entire
# difference between "the answer ships" and "the answer is *allowed* to ship".
from src.graph_v2 import ascii_graph_v2  # noqa: E402

print(ascii_graph_v2())
print("Done. Next: BUILD_MANUAL.md chapter 11 puts an HTTP gateway in front of this.")
