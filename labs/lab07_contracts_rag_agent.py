"""LAB 07 — The Contracts Analyst: a Grounded, Citing RAG Agent.

WHY THIS EXISTS
---------------
Lab 06 built retrieval; this lab builds the AGENT that uses it. The jump
matters commercially: clients do not buy "top-3 sections", they buy answers —
and the moment an LLM phrases an answer, hallucination risk appears. The
defense is a *grounding-and-citation contract* baked into the agent's system
prompt: every claim must come from a retrieved section, every number must
carry its document + section path, and "the contracts do not cover this" is a
first-class answer. This lab inspects the tools, builds the agent, and —
crucially — tests the case where the right answer is "I don't know".

OBJECTIVES
----------
1. Inspect ``search_contracts`` / ``get_contract_outline`` as LangChain @tool
   objects: name, description, args — the ONLY interface the LLM ever sees.
2. Build the worker with ``make_contracts_analyst()`` (LangChain v1
   ``create_agent`` ReAct loop).
3. Online: ask 4 questions, including one NO contract answers (verify the
   agent says so instead of hallucinating) and one comparison spanning 2
   contracts (verify it searches more than once).
4. Offline: run the retrieval-only flow and read exactly what the agent
   WOULD see — the tool messages that ground its answer.
5. Internalize grounding-and-citation as the consultant's anti-hallucination
   contract with the client.

DURATION: ~75 minutes
PREREQS : Lab 06 (the contracts index). Tool cells run OFFLINE; the live
          agent cells need an API key (.env → LLM_PROVIDER + key).
"""

# %% Setup — sys.path banner, offline status
from __future__ import annotations

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
print("LAB 07 — contracts_analyst: grounded RAG agent over the contracts")
print(f"OFFLINE mode: {OFFLINE}")
print("  - Tool inspection + retrieval-only flow: works FULLY offline.")
print("  - Live ReAct agent (make_contracts_analyst + 4 questions): needs a key.")
print("=" * 72)

# %% Tool inspection — what does the LLM actually see?
# @tool wraps a Python function into a schema the model receives on EVERY
# call: name + description + JSON args. The docstring IS the UX of your tool.
# If the description is vague, the agent calls it wrong — no amount of model
# quality fixes a bad tool card.
from src.agents.tools_rag import get_contract_outline, search_contracts  # noqa: E402

for t in (search_contracts, get_contract_outline):
    print(f"\nTOOL  : {t.name}")
    print(f"DESC  : {' '.join(t.description.split())[:240]}")
    print(f"ARGS  : {t.args}")

# Tools return STRINGS (never DataFrames/objects) — the model consumes text.
outline_text = get_contract_outline.invoke({})
print(f"\nget_contract_outline() → {len(outline_text):,} chars. First 600:\n")
print(outline_text[:600])

sample = search_contracts.invoke({"query": "SwiftShip late delivery penalty"})
print(f"\nsearch_contracts('SwiftShip late delivery penalty') → first 600 chars:\n")
print(sample[:600])

# %% The four test questions — including the trap
# Q4 is the most important test you will ever run on a RAG agent: there is NO
# DHL contract in the corpus. A grounded agent must say so. An ungrounded one
# will cheerfully invent fuel-surcharge terms — that behavior difference is
# written into the contracts_analyst system prompt (see src/agents/workers.py).
QUESTIONS: list[str] = [
    "What late-delivery penalty applies under the SwiftShip Express contract, "
    "and is there a cap?",
    "What are the payment terms in the Atlas Freight agreement?",
    "Compare the on-time delivery SLAs and late-delivery penalty structures of "
    "SwiftShip Express and NordHaul Logistics.",  # spans 2 contracts
    "What does our contract with DHL say about fuel surcharges?",  # NO such contract
]
for i, q in enumerate(QUESTIONS, 1):
    print(f"Q{i}. {q}")

# %% Live agent — ONLINE ONLY: build the worker and run all 4 questions
if not OFFLINE:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from src.agents.workers import make_contracts_analyst

    agent = make_contracts_analyst()  # create_agent(model, tools, system_prompt=...)

    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "-" * 72)
        print(f"Q{i}: {q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        # Show the ReAct trace: which tools were called with which args.
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"  [tool call] {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                print(f"  [tool result] {str(msg.content)[:120]}...")
        print(f"\nANSWER:\n{result['messages'][-1].content}")
    # 💡 What to verify by eye:
    #   Q3 → MULTIPLE search_contracts calls (one per carrier) before answering.
    #   Q4 → the agent states the corpus has no DHL contract. If it invents
    #        terms, tighten the system prompt — that is prompt-as-spec work.
else:
    print("\n[SKIP] Live agent cells skipped: OFFLINE mode (no API key).")
    print("       Set LLM_PROVIDER + key in .env to run the 4 questions live.")

# %% Offline path — the retrieval-only flow: what the agent WOULD see
# A ReAct agent is just: (system prompt + question) → tool call → TOOL OUTPUT
# → answer. Offline we can run everything except the final phrasing step.
# Reading raw tool output is the best hallucination training there is: if a
# number is not in this text, the agent has no business saying it.
if OFFLINE:
    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "-" * 72)
        print(f"Q{i}: {q}")
        tool_output = search_contracts.invoke({"query": q})
        print("AGENT WOULD SEE (search_contracts output, first 700 chars):")
        print(tool_output[:700])
    print("\n" + "-" * 72)
    # 💡 CONSULTANT'S NOTE on Q4 (the DHL trap): look at what retrieval just
    # returned — sections about fuel surcharges or carriers, but NOTHING from
    # any DHL document, because none exists. The retrieved text is the agent's
    # ENTIRE evidence base. Grounding rule: answer only from tool output ⇒ the
    # only honest answer is "no DHL contract in the corpus". Hallucination is
    # not a model bug you accept — it is a system-prompt contract you enforce.

# %% 🎯 EXERCISE 7.1 — catch cross-contract bleed before it catches you
# Run the single query "late delivery penalty cap" (no carrier named) and
# group the retrieved sections by the `doc` field of each result. How many
# DIFFERENT contracts appear in the top 6? What does that tell you about
# quoting "the penalty cap" without naming the document? Try it first.

# %% ✅ ANSWER 7.1 (fold this cell)
# region ANSWER ----------------------------------------------------------
from src.pageindex.retriever import PageIndexRetriever  # noqa: E402

_retr = PageIndexRetriever()
_hits = _retr.search("late delivery penalty cap", top_k=6)
by_doc: dict[str, list[str]] = {}
for h in _hits:
    by_doc.setdefault(h.doc, []).append(h.title)
print(f"\n'late delivery penalty cap' → sections from {len(by_doc)} different docs:")
for doc, titles in by_doc.items():
    print(f"  {doc}: {titles}")
print("""
LESSON: every carrier has a 'penalty cap' clause, and they all differ —
SwiftShip caps at 15% of shipment value, Atlas at $75,000/quarter, Pacific
Crest at 10% of freight charges, NordHaul at 12% of monthly lane fees.
A retrieval hit is only HALF a citation. The agent must always bind the
number to its `doc` + `path` — that is why RetrievedSection carries both,
and why the system prompt demands "document + section" with every figure.
""")
# endregion ---------------------------------------------------------------

# %% Wrap-up — grounding-and-citation as your contract with the client
# 💡 CONSULTANT'S NOTE — the closing argument for stakeholders:
#   * An LLM's fluency is constant whether it is right or wrong. The ONLY
#     scalable defense is architectural: the agent may state nothing that is
#     not in retrieved text, and must cite doc + section for every figure.
#   * "The contracts do not cover this" is a SUCCESS case, not a failure —
#     design for it, test for it (Q4), and demo it to the client on day one.
#     It is the single fastest way to earn trust in the system.
#   * Citations make the human reviewer cheap: checking §6.2 of one named
#     document takes 30 seconds; auditing an uncited paragraph takes an hour.
#   * This contract survives model swaps. Prompts and tools are your IP;
#     the model is a commodity behind get_llm().
print("\n" + "=" * 72)
print("Lab 07 complete. Next: lab08 applies the same discipline to SQL —")
print("schema-grounded NL2SQL with guardrails and self-repair.")
print("=" * 72)
