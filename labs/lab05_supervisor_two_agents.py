"""WHY THIS EXISTS
=================
LAB 05 — The Supervisor Pattern: Two Workers, One Router

One ReAct agent with 20 tools becomes confused and expensive. The production
answer is the SUPERVISOR PATTERN: small specialist agents (few tools each, sharp
system prompts) plus a router that picks who acts next and when to stop. Here
you assemble a two-worker version BY HAND — data_analyst + ml_engineer — the
exact wiring of ``src/graph.py``, minus two workers. Build it small, understand
it forever.

LEARNING OBJECTIVES
-------------------
- Explain supervisor routing in terms of Lab 04 primitives (conditional edges)
- Hand-wire supervisor → workers → supervisor → FINISH on SupplyChainState
- Run the loop OFFLINE with a scripted supervisor + stub workers (no LLM)
- (online) Swap in the real LLM supervisor and ReAct workers from src/agents
- Read multi-agent transcripts: who was routed, what they returned, why it ended

DURATION: ~90 minutes
PREREQUISITES: Lab 04.
API KEY: needed ONLY for section 4 (real agents + 2 scenarios).
Sections 1-3 (pattern, hand-wiring, scripted run, topology) run offline.
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

from src.config import get_llm, is_offline  # noqa: E402

OFFLINE = is_offline()
print("=" * 72)
print("LAB 05 — Supervisor pattern with two workers")
print(f"OFFLINE mode : {OFFLINE}")
print("API key needed: only for section 4 (real LLM supervisor + ReAct workers).")
print("Sections 1-3 (wiring, scripted offline run, topology) run offline.")
print("=" * 72)

# %% ── 1. The shared state — the contract that lets agents plug together ────────
# src/state.py defines ONE state schema for the whole system. Any worker that
# reads/writes this dict can join the graph. That's the integration contract.
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from src.state import WORKER_DESCRIPTIONS, WORKERS, SupplyChainState  # noqa: E402

LAB_WORKERS = ("data_analyst", "ml_engineer")  # this lab: 2 of the 4
print(f"Full system workers : {WORKERS}")
print(f"This lab's workers  : {LAB_WORKERS}")
for w in LAB_WORKERS:
    print(f"  {w:<13} — {WORKER_DESCRIPTIONS[w][:70]}…")

# The supervisor loop, in one diagram:
#
#        START → supervisor ──(next_agent)──> data_analyst ─┐
#                  ↑   │                                    │
#                  │   ├────────────────────> ml_engineer ──┤
#                  │   └──(FINISH)──> END                   │
#                  └────────────────────────────────────────┘
#
# Every worker routes BACK to the supervisor, which decides: another worker,
# or FINISH. Compare with Lab 04's toy: same conditional edge, now in a loop.

# %% ── 2. Hand-wire it OFFLINE: scripted supervisor + stub workers ───────────────
# To study the WIRING without burning tokens, we use stand-ins (exactly what the
# project's tests do): a supervisor that follows a fixed script, and workers
# that echo. Same graph shape as production — swap the nodes, keep the wiring.
from itertools import count  # noqa: E402

ROUTE_SCRIPT = ["data_analyst", "ml_engineer", "FINISH"]
_turn = count()


def scripted_supervisor(state: SupplyChainState) -> dict:
    """Stand-in for the LLM router: replays a fixed routing script."""
    step = next(_turn)
    nxt = ROUTE_SCRIPT[step] if step < len(ROUTE_SCRIPT) else "FINISH"
    print(f"  [supervisor] turn {step}: routing → {nxt}")
    return {"next_agent": nxt}


def make_stub_worker(name: str):
    """Stand-in for a ReAct agent: returns a canned AIMessage tagged with name."""

    def node(state: SupplyChainState) -> dict:
        question = state["messages"][0].content
        reply = AIMessage(
            content=f"[{name}] (stub) I would analyze: {question!r}", name=name
        )
        return {"messages": [reply]}

    return node


def wire_two_worker_graph(supervisor_fn, worker_nodes: dict):
    """The supervisor wiring — IDENTICAL for stubs and real agents.

    This function is the whole lesson: src/graph.py does exactly this,
    just with all four workers.
    """
    builder = StateGraph(SupplyChainState)
    builder.add_node("supervisor", supervisor_fn)
    for name, node in worker_nodes.items():
        builder.add_node(name, node)
        builder.add_edge(name, "supervisor")  # workers ALWAYS report back
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s["next_agent"],  # reads the routing decision from state
        {**{name: name for name in worker_nodes}, "FINISH": END},
    )
    return builder.compile()


stub_app = wire_two_worker_graph(
    scripted_supervisor, {w: make_stub_worker(w) for w in LAB_WORKERS}
)

# %% ── 3. Topology + a scripted end-to-end run (fully offline) ──────────────────
print("Graph topology:")
try:
    print(stub_app.get_graph().draw_ascii())  # needs optional `grandalf`
except Exception:
    print(stub_app.get_graph().draw_mermaid())  # paste into mermaid.live

question = "Which shipping mode is riskiest, and can we predict late orders?"
print(f"\nQ: {question}")
out = stub_app.invoke(
    {"messages": [HumanMessage(content=question)], "next_agent": ""}
)
print("\nTranscript (note BOTH workers contributed, then FINISH):")
for msg in out["messages"]:
    who = getattr(msg, "name", None) or msg.type
    print(f"  {who:<13} | {msg.content[:90]}")

# 💡 CONSULTANT'S NOTE: this stub graph is also your DEMO INSURANCE. Client
# wifi dies, API quota dies — the scripted topology still runs and the
# architecture conversation still happens. Always have an offline path.

# %% ── 4. The real thing: LLM supervisor + ReAct workers  [NEEDS API KEY] ───────
# Same wiring function; only the nodes change:
#   scripted_supervisor  → make_supervisor_node() (LLM routes via structured output)
#   make_stub_worker(w)  → make_worker_node(w, make_<worker>()) (real ReAct agents)
if OFFLINE:
    print("⏭  SKIPPED — section 4 needs an API key (OFFLINE mode is active).")
    print("   Set LLM_PROVIDER + key in .env (see .env.example), then re-run.")
else:
    from src.agents.supervisor import make_supervisor_node, make_worker_node
    from src.agents.workers import make_data_analyst, make_ml_engineer

    real_app = wire_two_worker_graph(
        make_supervisor_node(get_llm()),
        {
            "data_analyst": make_worker_node("data_analyst", make_data_analyst()),
            "ml_engineer": make_worker_node("ml_engineer", make_ml_engineer()),
        },
    )

    scenarios = [
        "Which shipping mode and market have the highest late-delivery rates?",
        "Evaluate the hist_gb late-delivery model and explain its top 3 drivers.",
    ]
    for q in scenarios:
        print(f"\n{'=' * 72}\nQ: {q}")
        state = {"messages": [HumanMessage(content=q)], "next_agent": ""}
        # stream_mode="updates" yields {node_name: update} per step → a hop trace.
        final_messages = None
        for chunk in real_app.stream(state, config={"recursion_limit": 25},
                                     stream_mode="updates"):
            for node_name, update in chunk.items():
                print(f"  [hop] {node_name}")
                if update and update.get("messages"):
                    final_messages = update["messages"]
        if final_messages:
            print(f"\nFinal answer:\n{final_messages[-1].content}")
    # Watch the routing: scenario 1 should go to data_analyst, scenario 2 to
    # ml_engineer. The supervisor decided that from WORKER_DESCRIPTIONS alone —
    # write those descriptions as carefully as you write tool docstrings.

# %% ── 5. 🎯 EXERCISE — predict the transcript ──────────────────────────────────
# TODO (10 min), offline-friendly:
#   a) Change ROUTE_SCRIPT to ["ml_engineer", "ml_engineer", "FINISH"].
#   b) BEFORE running: how many messages will the final state hold? Who's named
#      on each? (Remember: 1 human + 1 AIMessage per worker visit.)
#   c) Rebuild the stub graph (reset `_turn = count()`!) and verify.
#
# region 📁 ANSWER (unfold)
# _turn2 = count()
# script2 = ["ml_engineer", "ml_engineer", "FINISH"]
# def sup2(state):
#     step = next(_turn2)
#     return {"next_agent": script2[step] if step < len(script2) else "FINISH"}
# app2 = wire_two_worker_graph(sup2, {w: make_stub_worker(w) for w in LAB_WORKERS})
# out2 = app2.invoke({"messages": [HumanMessage(content="hi")], "next_agent": ""})
# print(len(out2["messages"]))   # 3: human + ml_engineer + ml_engineer
# # Both AI messages carry name="ml_engineer" — the supervisor is ALLOWED to
# # revisit a worker (e.g. "train, then ALSO tune the threshold").
# endregion

# %% ── 6. Wrap-up & Week 4 preview ──────────────────────────────────────────────
print(
    """
LAB 05 TAKEAWAYS
----------------
1. Supervisor pattern = Lab 04's conditional edge, looped: route → work → report.
2. Workers ALWAYS report back; only the supervisor may FINISH.
3. The wiring is identical for stubs and real agents → testable offline.
4. Routing quality lives in WORKER_DESCRIPTIONS — prose is architecture.

WEEK 4 PREVIEW — completing the picture
---------------------------------------
Two more workers join this exact wiring (src/graph.py already has the slots):
  * contracts_analyst — reasoning-based RAG over carrier contracts (PageIndex
    tree navigation, no vector DB)            → Labs 06-07
  * sql_analyst       — NL→SQL over the warehouse, schema-grounded by an
    OpenMetadata-style catalog, with guardrails → Lab 08
Then Lab 09 runs the flagship question that needs THREE agents in one turn:
  "Order 104872 ships Same Day via SwiftShip Express — how likely is it late,
   what penalty applies under the SwiftShip contract, and what was SwiftShip's
   late rate last quarter?"
Same state, same wiring, more specialists. That's the whole trick.
"""
)
