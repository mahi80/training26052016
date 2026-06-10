"""LAB 09 — Capstone: the Full 4-Worker Supervisor System, End to End.

WHY THIS EXISTS
---------------
Labs 01-08 built the parts: EDA tools, an ML pipeline, a vectorless contracts
retriever, a guard-railed NL2SQL stack, and one worker agent for each. This
lab assembles them under a SUPERVISOR — a routing LLM that reads the
conversation, picks the next worker (or FINISH), and synthesizes the final
answer. The flagship scenario is the one from ARCHITECTURE.md §1: a single
operations question that no single agent can answer, because the facts live
in three different systems (a model, a contract, a warehouse). Orchestration
is the product; the workers are inventory.

OBJECTIVES
----------
1. Compile the full graph with ``build_graph()`` — and OFFLINE, compile the
   SAME graph via dependency injection (fake router LLM + scripted workers),
   exactly like tests/test_graph_routing.py.
2. Render the topology and read it as a state machine over SupplyChainState.
3. Stream the flagship 3-agent scenario hop by hop, with commentary on every
   supervisor decision.
4. Run the inline offline-check (the ``python main.py --offline-check``
   equivalent) and walk through the routing test conceptually.
5. Leave with a production-hardening checklist mapped to LangGraph features.

DURATION: ~120 minutes
PREREQS : Labs 04-08. Offline: everything below runs via DI mocks. Online:
          the real ReAct workers + supervisor run live. The free-form ops
          copilot loop additionally needs LABS_INTERACTIVE=1.
"""

# %% Setup — sys.path banner, offline status
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive `# %%` cell execution
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import is_offline  # noqa: E402

OFFLINE = is_offline()
print("=" * 72)
print("LAB 09 — full system integration: supervisor + 4 workers")
print(f"OFFLINE mode: {OFFLINE}")
print("  - Graph compilation, topology, scripted flagship run, offline-check:")
print("    work FULLY offline via dependency injection (fake LLM + workers).")
print("  - Live ReAct workers + real routing + ops-copilot loop: need a key.")
print("=" * 72)

# %% Build the graph — real when online, dependency-injected when offline
# build_graph(llm=None, workers=None) has two injection points (ARCHITECTURE
# §4.7): `llm` drives the supervisor's structured-output routing; `workers`
# replaces the four ReAct agents with plain callables. That design is not a
# testing trick — it is what makes a multi-agent system OWNABLE: you can run
# the whole topology in CI for $0 and swap any worker without touching wiring.
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from src.agents.supervisor import Router  # noqa: E402
from src.graph import ascii_graph, build_graph  # noqa: E402
from src.state import WORKERS  # noqa: E402

FLAGSHIP = (
    "Order 104872 ships Same Day via SwiftShip Express to Western Europe — "
    "how likely is it to be late, what penalty applies under the SwiftShip "
    "contract if it is, and what was SwiftShip's late rate last quarter?"
)

# The route plan the supervisor SHOULD produce for the flagship question.
ROUTE_PLAN: list[tuple[str, str]] = [
    ("ml_engineer", "Risk question first: score P(late) for this order."),
    ("contracts_analyst", "Penalty terms live in the SwiftShip MSA, not the data."),
    ("sql_analyst", "Historical late rate = aggregate query on the warehouse."),
    ("FINISH", "All three facts collected — synthesize the answer."),
]


class ScriptedRouterLLM:
    """Stands in for get_llm() in the supervisor: `.with_structured_output(Router)`
    returns an object whose `.invoke()` yields the next scripted Router decision.
    This is the same trick tests/test_graph_routing.py uses."""

    def __init__(self, plan: list[tuple[str, str]]) -> None:
        self._plan = list(plan)

    def with_structured_output(self, schema: type) -> "ScriptedRouterLLM":
        self._schema = schema
        return self

    def invoke(self, _input: object) -> object:
        nxt, reason = self._plan.pop(0)
        return self._schema(next=nxt, reason=reason)


# Scripted worker replies — canned stand-ins that MIRROR what the real ReAct
# workers return online (same facts you verified in labs 06-08).
SCRIPTED_REPLIES: dict[str, str] = {
    "data_analyst": "(not used in this scenario)",
    "ml_engineer": (
        "predict_order_risk(Same Day, SwiftShip Express, Western Europe): "
        "P(late) = 0.62 — HIGH risk band. Drivers: 1-day schedule, carrier."
    ),
    "contracts_analyst": (
        "Per swiftship_express_msa.md §6.1-6.2: penalty = 2% of Shipment Value "
        "per Late Business Day, no grace period, capped at 15% of Shipment Value."
    ),
    "sql_analyst": (
        "Warehouse, Q4-2025: SwiftShip Express had the worst late rate of the "
        "four carriers (orders JOIN carriers, AVG(late_delivery) by carrier)."
    ),
}


def make_scripted_worker(name: str):
    def node(_state: dict) -> dict:
        return {"messages": [AIMessage(content=SCRIPTED_REPLIES[name], name=name)]}

    return node


if OFFLINE:
    app = build_graph(
        llm=ScriptedRouterLLM(ROUTE_PLAN),
        workers={w: make_scripted_worker(w) for w in WORKERS},
    )
    print("Compiled graph with INJECTED fake router + scripted workers (offline).")
else:
    app = build_graph()  # real supervisor + 4 real ReAct workers
    print("Compiled the REAL graph: live supervisor + 4 ReAct workers.")

# %% Render the topology — read it as a state machine
# One supervisor node, four worker nodes, conditional edges out of the
# supervisor on state["next_agent"], every worker wired BACK to the
# supervisor. The hub-and-spoke shape is the whole pattern.
try:
    print(ascii_graph())
except Exception as exc:  # offline builds may not support the default path
    print(f"(ascii_graph() unavailable here: {exc} — rendering compiled app)")
    g = app.get_graph()
    try:
        print(g.draw_ascii())
    except Exception:
        print(g.draw_mermaid())

# %% The flagship 3-agent scenario — streamed hop by hop
# 💡 Read the stream like a consultant narrating a demo: each update is one
# node finishing. Watch the supervisor ALTERNATE with workers — route, work,
# return, re-route — until it has every fact and emits FINISH.
SUPERVISOR_COMMENTARY = [
    "the question opens with 'how likely is it to be late' → that is a\n"
    "    predictive-scoring task → route to ml_engineer (its tools wrap the\n"
    "    Week-3 sklearn pipeline).",
    "risk is scored; 'what penalty applies under the SwiftShip contract' is\n"
    "    a document question → contracts_analyst (PageIndex retrieval, lab07).",
    "penalty terms in hand; 'late rate last quarter' is a historical\n"
    "    aggregate → sql_analyst (schema-grounded SQL, lab08).",
    "all three facts are in the message history → FINISH: stop routing and\n"
    "    synthesize one answer for the operations analyst.",
]

print("\n" + "=" * 72)
print(f"USER: {FLAGSHIP}")
print("=" * 72)

initial_state = {"messages": [HumanMessage(content=FLAGSHIP)], "next_agent": ""}
visited: list[str] = []
hop = 0
for update in app.stream(initial_state, stream_mode="updates"):
    for node_name, delta in update.items():
        visited.append(node_name)
        print(f"\n--- node finished: [{node_name}] " + "-" * (40 - len(node_name)))
        if node_name == "supervisor":
            nxt = (delta or {}).get("next_agent", "?")
            print(f"    routing decision → {nxt}")
            if hop < len(SUPERVISOR_COMMENTARY):
                print(f"    why: {SUPERVISOR_COMMENTARY[hop]}")
            hop += 1
        for msg in (delta or {}).get("messages", []):
            content = " ".join(str(getattr(msg, "content", msg)).split())
            print(f"    {getattr(msg, 'name', None) or node_name}: {content[:220]}")

worker_visits = [n for n in visited if n in WORKERS]
print(f"\nHop sequence: {' -> '.join(visited)}")
if OFFLINE:
    assert worker_visits == ["ml_engineer", "contracts_analyst", "sql_analyst"], (
        f"unexpected worker order: {worker_visits}"
    )
    print("Scripted route verified: ml_engineer -> contracts_analyst -> sql_analyst.")
# 💡 CONSULTANT'S NOTE — no worker knows the others exist. The supervisor owns
# decomposition and sequencing; workers own depth. That separation is why you
# can sell 'add a fifth agent' as a one-node change, not a rewrite.

# %% Offline-check, inline — the `python main.py --offline-check` equivalent
# Before any classroom or client demo, prove the no-key fallbacks work.
from src.agents.tools_sql import run_sql_query  # noqa: E402
from src.metadata.catalog import MetadataCatalog  # noqa: E402
from src.pageindex.retriever import PageIndexRetriever  # noqa: E402

checks: list[tuple[str, bool]] = []

_hits = PageIndexRetriever().search("SwiftShip late delivery penalty", top_k=3)
checks.append(("pageindex lexical retrieval returns sections", len(_hits) >= 1))

_tables = MetadataCatalog().list_tables()
checks.append(("metadata catalog lists 4 tables incl. orders",
               len(_tables) == 4 and "orders" in _tables))

checks.append(("sql guardrail rejects DROP",
               "SQL_ERROR" in run_sql_query.invoke({"sql": "DROP TABLE orders"})))
checks.append(("sql happy path returns rows",
               "SQL_ERROR" not in run_sql_query.invoke(
                   {"sql": "SELECT COUNT(*) AS n FROM orders LIMIT 1"})))
checks.append(("graph compiles + streams via dependency injection",
               len(visited) >= 4))

print("\nOFFLINE-CHECK")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
assert all(ok for _, ok in checks), "offline-check failed — see FAIL lines above"

# %% The routing test, conceptually — tests/test_graph_routing.py
# What you just ran IS the routing test, live. The pytest version:
#   1. ARRANGE — build a fake structured-output LLM (like ScriptedRouterLLM)
#      with a scripted Router sequence, plus plain-callable workers; pass both
#      into build_graph(llm=..., workers=...). No API key, no network.
#   2. ACT     — invoke the compiled graph on a user message.
#   3. ASSERT  — supervisor → worker → supervisor → FINISH: the visited-node
#      order matches the script, every worker reply landed in state["messages"]
#      as an AIMessage(name=worker), and the final next_agent == "FINISH".
# WHY THIS MATTERS: it pins the ORCHESTRATION contract (wiring, conditional
# edges, message accumulation) independently of any model's mood. Prompt
# quality you evaluate with eval suites; topology you protect with unit tests.
print("\n[Concept] tests/test_graph_routing.py = exactly this DI pattern, "
      "run by pytest with assertions on the hop sequence.")

# %% 🎯 EXERCISE 9.1 — script a 6th scenario for an unused worker pair
# `python main.py --demo` ships 5 canned scenarios, and the flagship above
# used ml_engineer -> contracts_analyst -> sql_analyst. No canned scenario
# routes to data_analyst AND contracts_analyst together. Your task (~15 min,
# fully offline — this is the TRAINING_PLAN W4-D4 stretch goal):
#   a) Write a question that genuinely needs BOTH an EDA fact from the orders
#      CSV and a contract clause — but no model score and no SQL.
#   b) Build a SECOND graph with ScriptedRouterLLM + make_scripted_worker
#      (a fresh route plan — the first one was consumed by .pop(0)).
#   c) Stream it and ASSERT the worker hop sequence is exactly
#      ["data_analyst", "contracts_analyst"] — no other worker visited.
# Online stretch: re-run the same question through the REAL build_graph()
# and check whether the live supervisor discovers the same pair.

# %% ✅ ANSWER 9.1 (fold this cell)
# region ANSWER ----------------------------------------------------------
EX_QUESTION = (
    "Which market has our worst late-delivery rate, and if we shift that "
    "volume to NordHaul, what on-time SLA does their contract commit to?"
)
EX_PLAN: list[tuple[str, str]] = [
    ("data_analyst", "'worst late rate by market' = EDA aggregate; no model."),
    ("contracts_analyst", "The SLA number lives in the NordHaul MSA §5.2."),
    ("FINISH", "Both facts collected — synthesize."),
]
SCRIPTED_REPLIES.update(  # make_scripted_worker closures read this at call time
    data_analyst="late_rate_by(market): Africa is worst — lab01's planted signal.",
    contracts_analyst=("Per nordhaul_logistics_msa.md §5.2: On-Time Delivery Rate "
                       ">= 97.5% per lane per month — the portfolio's highest."),
)
ex_app = build_graph(
    llm=ScriptedRouterLLM(EX_PLAN),
    workers={w: make_scripted_worker(w) for w in WORKERS},
)
ex_hops: list[str] = []
for update in ex_app.stream(
        {"messages": [HumanMessage(content=EX_QUESTION)], "next_agent": ""},
        stream_mode="updates"):
    ex_hops.extend(update)  # each update is {node_name: delta}
ex_workers = [n for n in ex_hops if n in WORKERS]
assert ex_workers == ["data_analyst", "contracts_analyst"], ex_workers
print(f"EXERCISE 9.1 hop sequence verified: {' -> '.join(ex_hops)}")
# 💡 The lesson: a 6th scenario costs ONE question + ONE route expectation —
# zero new wiring. Offline, the DI harness pins the topology; online, whether
# the live supervisor finds this route depends on the WORKER_DESCRIPTIONS
# prose (lab05's lesson) — if it misroutes, fix the descriptions, not the graph.
# endregion ---------------------------------------------------------------

# %% Free-form ops copilot — gated behind LABS_INTERACTIVE=1 (+ API key)
# The same compiled app, as a console copilot for an operations analyst.
INTERACTIVE = os.getenv("LABS_INTERACTIVE", "0") == "1"
if INTERACTIVE and not OFFLINE:
    print("\nOps copilot — ask about orders, risk, contracts, or the warehouse.")
    print("Type 'quit' to exit.")
    while True:
        question = input("\nyou> ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            break
        out = app.invoke({"messages": [HumanMessage(content=question)],
                          "next_agent": ""})
        print(f"\ncopilot> {out['messages'][-1].content}")
else:
    reason = "OFFLINE (no API key)" if OFFLINE else "LABS_INTERACTIVE != 1"
    print(f"\n[SKIP] ops copilot loop skipped: {reason}.")
    print("       Enable with: set LABS_INTERACTIVE=1 + an API key in .env.")

# %% Production hardening checklist — each item mapped to a LangGraph feature
print("\nPRODUCTION HARDENING CHECKLIST")
print("""
1. DURABILITY & HUMAN-IN-THE-LOOP
   - compile(checkpointer=SqliteSaver/PostgresSaver): every hop persisted;
     a thread_id resumes mid-conversation after a crash or a deploy.
   - interrupt_before=["sql_analyst"]: pause the graph for human approval
     before any warehouse query runs; resume with app.invoke(None, config).
     Demo this to risk-averse clients — it usually closes the deal.
2. OBSERVABILITY
   - LangSmith: set LANGSMITH_TRACING=1 and every supervisor decision, tool
     call, token count and latency becomes a queryable trace tree.
   - OpenTelemetry: LangSmith exports OTel spans, so agent hops land in the
     client's EXISTING Grafana/Datadog stack — meet ops teams where they are.
3. EVALUATION HARNESSES
   - Routing/topology: the DI pattern above as pytest (cheap, deterministic).
   - Answer quality: a golden set of (question, expected-facts) pairs run
     nightly; LLM-as-judge for groundedness, exact-match for cited numbers.
   - Regression gate: no prompt or model change ships without the eval run.
4. COST CONTROL
   - Model tiering via get_llm(): a small/fast model for the supervisor's
     routing (a 10-token structured output), a stronger one for synthesis.
   - recursion_limit + per-thread token budgets stop runaway agent loops.
   - Cache stable tool outputs (contract outline, table schemas) — they
     change weekly, not per message.
""")

# %% Where this fits the curriculum + wrap-up
# 💡 CONSULTANT'S NOTE — the road ahead:
#   * WEEK 6 (MCP): every @tool you built is a candidate MCP server. Lift
#     tools_sql/tools_rag behind the Model Context Protocol and ANY client —
#     Claude Desktop, an IDE, another team's agent — can call your supply-
#     chain capabilities without importing your code base.
#   * WEEK 7 (security): retrieved contract text and SQL results are
#     UNTRUSTED INPUT — prompt-injection can ride in on a poisoned document.
#     The guardrails here (SELECT-only, allow-listed tools, grounding rules)
#     are layer one; Week 7 adds threat modeling, sandboxing, least-privilege
#     DB roles, and output filtering.
#   * What you can now say to a client, truthfully: "We built a multi-agent
#     system where every answer is grounded in a model, a contract section,
#     or a SQL result — and we can show you the trace for any of them."
print("=" * 72)
print("Lab 09 complete — the Week 3&4 system is assembled end to end.")
print("Run `python main.py --demo` (with a key) or `--offline-check` next.")
print("=" * 72)
