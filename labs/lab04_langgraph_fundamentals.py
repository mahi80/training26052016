"""WHY THIS EXISTS
=================
LAB 04 — LangGraph From First Principles: State, Nodes, Edges, Tools, Agents

Multi-agent systems look magical until you see the machinery: a LangGraph app is
just a STATE MACHINE — a typed dict passed between plain Python functions, with
edges deciding who runs next. This lab builds a toy 3-node graph with zero LLM
involvement (it runs fully offline), then inspects real @tool objects from the
project, and finally — when an API key is available — assembles a single ReAct
agent that picks tools by itself.

LEARNING OBJECTIVES
-------------------
- Define a TypedDict state with an ``add_messages`` reducer and explain reducers
- Wire nodes + a conditional edge into a StateGraph; compile, draw, invoke it
- Read a @tool's name / description / args — the agent's *only* view of your code
- Invoke a tool directly to demystify what an agent does under the hood
- (online) Build a ReAct agent with ``create_agent`` and watch the tool loop

DURATION: ~90 minutes
PREREQUISITES: Labs 01-03; Python typing basics.
API KEY: needed ONLY for section 6 (ReAct agent). Sections 1-5 run offline.
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
print("LAB 04 — LangGraph fundamentals")
print(f"OFFLINE mode : {OFFLINE}")
print("API key needed: only for section 6 (single ReAct agent).")
print("Sections 1-5 (toy graph, tool anatomy, direct tool calls) run offline.")
print("=" * 72)

# %% ── 1. The state: a TypedDict with a reducer ────────────────────────────────
# A LangGraph state is a dict schema. Each node returns a PARTIAL update; how
# updates merge is decided per-key by a "reducer". For `messages` we use
# add_messages: updates are APPENDED, so every node sees the full conversation.
# Keys without a reducer (like `category`) are simply overwritten.
from typing import Annotated, Literal  # noqa: E402

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402


class TriageState(TypedDict):
    """Toy state for a shipping-helpdesk triage graph."""

    messages: Annotated[list[AnyMessage], add_messages]  # append, don't overwrite
    category: str  # plain key → last writer wins


# %% ── 2. Nodes are plain functions: state in, partial update out ───────────────
# No classes, no framework voodoo. A node = Callable[[State], dict].
def classify(state: TriageState) -> dict:
    """A trivial keyword classifier — later, an LLM supervisor plays this role."""
    text = state["messages"][-1].content.lower()
    shippy = ("late", "ship", "delay", "carrier", "deliver")
    category = "shipping" if any(w in text for w in shippy) else "general"
    return {"category": category}  # partial update: only this key changes


def shipping_desk(state: TriageState) -> dict:
    reply = AIMessage(
        content="[shipping_desk] Routed to logistics — checking carrier status.",
        name="shipping_desk",
    )
    return {"messages": [reply]}  # add_messages APPENDS this


def general_desk(state: TriageState) -> dict:
    reply = AIMessage(
        content="[general_desk] Routed to general support.", name="general_desk"
    )
    return {"messages": [reply]}


# %% ── 3. Wire the graph: 3 nodes + one conditional edge ────────────────────────
def route_after_classify(state: TriageState) -> Literal["shipping", "general"]:
    """Conditional-edge function: reads state, returns the NAME of a branch."""
    return "shipping" if state["category"] == "shipping" else "general"


builder = StateGraph(TriageState)
builder.add_node("classify", classify)
builder.add_node("shipping_desk", shipping_desk)
builder.add_node("general_desk", general_desk)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {"shipping": "shipping_desk", "general": "general_desk"},
)
builder.add_edge("shipping_desk", END)
builder.add_edge("general_desk", END)
toy_app = builder.compile()  # compile() validates wiring and returns a runnable

# Draw it. (ascii needs the optional `grandalf` package; mermaid always works —
# paste mermaid output into https://mermaid.live to render.)
try:
    print(toy_app.get_graph().draw_ascii())
except Exception:
    print(toy_app.get_graph().draw_mermaid())

# %% ── 4. Invoke it and watch the reducer at work ───────────────────────────────
for question in ("My order is 3 days late, where is it?", "How do I reset my password?"):
    out = toy_app.invoke({"messages": [HumanMessage(content=question)], "category": ""})
    print(f"\nQ: {question}")
    print(f"   category   → {out['category']}")
    print(f"   n_messages → {len(out['messages'])}  (input + appended reply)")
    print(f"   reply      → {out['messages'][-1].content}")

# 💡 CONSULTANT'S NOTE: this 30-line toy IS the supervisor pattern's skeleton —
# classify ≈ supervisor, the desks ≈ worker agents, the conditional edge ≈
# routing. When you pitch "multi-agent AI" to a client, you are pitching THIS,
# plus LLMs inside the nodes. Demystifying it builds trust in the design review.

# %% ── 5. Tools: the agent's only window into your code ─────────────────────────
# A @tool wraps a function with a NAME, a DESCRIPTION (its docstring) and an ARGS
# SCHEMA. The LLM never sees your code — it sees exactly these three things and
# decides, from them alone, when and how to call the tool. Bad docstring = tool
# never gets called (or gets called wrong). Documentation IS the interface.
from src.agents.tools_ml import analyze_late_rate, get_data_overview  # noqa: E402

for t in (get_data_overview, analyze_late_rate):
    print(f"\nname        : {t.name}")
    print(f"description : {t.description}")
    print(f"args schema : {t.args}")

# An agent "calling a tool" is nothing more than .invoke with a dict of args —
# do it yourself, no LLM required:
print("\n--- get_data_overview() called directly ---")
print(get_data_overview.invoke({}))
print("\n--- analyze_late_rate(dimension='shipping_mode') called directly ---")
print(analyze_late_rate.invoke({"dimension": "shipping_mode"}))

# Note tools return STRINGS (markdown tables), never DataFrames: the result is
# pasted into the LLM's context window, so it must be plain readable text.

# %% ── 6. ReAct agent: an LLM choosing tools in a loop  [NEEDS API KEY] ─────────
# create_agent() returns a prebuilt LangGraph: the model loops
#   think → call a tool → read result → think …  until it can answer.
if OFFLINE:
    print("⏭  SKIPPED — section 6 needs an API key (OFFLINE mode is active).")
    print("   Set LLM_PROVIDER + key in .env (see .env.example), then re-run.")
else:
    from langchain.agents import create_agent

    analyst = create_agent(
        model=get_llm(),
        tools=[get_data_overview, analyze_late_rate],
        system_prompt=(
            "You are a supply-chain data analyst. Use your tools to ground every "
            "claim in the orders dataset; never invent numbers."
        ),
    )
    result = analyst.invoke(
        {"messages": [HumanMessage(content="Which shipping mode is riskiest?")]}
    )
    for msg in result["messages"]:
        msg.pretty_print()
    # Read the transcript bottom-up: final answer ← tool result ← tool call ←
    # the model DECIDING to call analyze_late_rate('shipping_mode'). That
    # decision was driven entirely by the tool descriptions you printed above.

# %% ── 7. 🎯 EXERCISE — add a third desk ────────────────────────────────────────
# TODO (10 min): extend the TOY graph with a "billing_desk":
#   a) classify() should return "billing" for words like invoice/refund/charge.
#   b) Add the node, extend the conditional-edge mapping, edge to END.
#   c) Invoke with "I was charged twice on my invoice" and verify the route.
#
# region 📁 ANSWER (unfold)
# def classify_v2(state: TriageState) -> dict:
#     text = state["messages"][-1].content.lower()
#     if any(w in text for w in ("invoice", "refund", "charge")):
#         return {"category": "billing"}
#     if any(w in text for w in ("late", "ship", "delay", "carrier", "deliver")):
#         return {"category": "shipping"}
#     return {"category": "general"}
#
# def billing_desk(state: TriageState) -> dict:
#     return {"messages": [AIMessage(content="[billing_desk] Reviewing charges.",
#                                    name="billing_desk")]}
#
# b2 = StateGraph(TriageState)
# b2.add_node("classify", classify_v2)
# for n, f in (("shipping_desk", shipping_desk), ("general_desk", general_desk),
#              ("billing_desk", billing_desk)):
#     b2.add_node(n, f); b2.add_edge(n, END)
# b2.add_edge(START, "classify")
# b2.add_conditional_edges("classify", lambda s: s["category"],
#     {"shipping": "shipping_desk", "general": "general_desk",
#      "billing": "billing_desk"})
# out = b2.compile().invoke({"messages": [HumanMessage(
#     content="I was charged twice on my invoice")], "category": ""})
# print(out["category"], "→", out["messages"][-1].content)
# endregion

# %% ── 8. Wrap-up ───────────────────────────────────────────────────────────────
print(
    """
LAB 04 TAKEAWAYS
----------------
1. A LangGraph app = TypedDict state + plain-function nodes + edges. No magic.
2. Reducers (add_messages) define HOW updates merge — append vs overwrite.
3. Conditional edges are the routing primitive behind every supervisor.
4. A @tool's name/description/args are the LLM's ONLY view of your code.
5. ReAct = an LLM choosing tools in a loop; create_agent gives it prebuilt.

Next: Lab 05 — scale from one agent to the SUPERVISOR PATTERN with two workers.
"""
)
