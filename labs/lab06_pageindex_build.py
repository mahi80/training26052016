"""LAB 06 — Vectorless RAG: Building a PageIndex over the Contracts Corpus.

WHY THIS EXISTS
---------------
Week 4 opens with the question every client asks about RAG: "Why did the bot
quote the wrong contract?" Classic chunk+embed+cosine pipelines lose the one
thing a contract has in abundance — STRUCTURE. This lab builds the alternative:
a PageIndex-style hierarchical tree index (a machine-readable table of contents)
that an LLM *navigates by reasoning*, the way a human expert flips to
"Section 6 — Late Delivery Penalties" instead of skimming 40 random paragraphs.
No vector database, no embeddings — and an audit trail of WHY each section was
retrieved, which is exactly what legal/compliance stakeholders demand.

OBJECTIVES
----------
1. Articulate the failure modes of chunk+embed+cosine RAG on structured docs.
2. Build the tree index over data/contracts with ``build_index()`` (offline mode).
3. Read a document tree and the corpus-wide ``outline()`` like a consultant.
4. Run lexical (offline) retrieval and inspect every ``RetrievedSection`` field.
5. Find the boundary: a query where lexical scoring fails but LLM tree
   navigation would succeed — and explain why to a client.

DURATION: ~90 minutes
PREREQS : Labs 01-05 (Week 3); contracts generated in data/contracts/
          (this lab regenerates them if missing). No API key needed —
          everything here runs fully OFFLINE.
"""

# %% Setup — sys.path banner, offline status
from __future__ import annotations

import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive `# %%` cell execution: assume cwd = project root
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles can fall back to a legacy codepage when output is piped;
# force UTF-8 so contract text (em-dashes, typographic quotes) prints cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import is_offline  # noqa: E402

OFFLINE = is_offline()
print("=" * 72)
print("LAB 06 — PageIndex: vectorless, reasoning-based RAG over contracts")
print(f"OFFLINE mode: {OFFLINE}")
print("  - Tree building + lexical retrieval: works FULLY offline (this lab).")
print("  - LLM-generated node summaries + LLM tree navigation: need an API key")
print("    (set LLM_PROVIDER + key in .env; see ARCHITECTURE.md §5).")
print("=" * 72)

# Ensure the contracts corpus exists (generators are idempotent, seed 42).
_contracts_dir = PROJECT_ROOT / "data" / "contracts"
if not _contracts_dir.exists() or not any(_contracts_dir.glob("*.md")):
    from data import generate_contracts

    generate_contracts.main()
    print("(regenerated data/contracts/*.md)")

# %% WHY vectorless? — chunk+embed+cosine vs reasoning-over-a-tree
# 💡 CONSULTANT'S NOTE — this cell is the slide you will draw on a whiteboard.
#
# CLASSIC RAG (chunk + embed + cosine top-k):
#   split docs into ~500-token chunks → embed each → at query time embed the
#   query → return the k nearest chunks by cosine similarity.
#   Failure modes on contracts:
#   1. LOSES DOCUMENT STRUCTURE. A chunk from SwiftShip §6.1 ("2% per late
#      day") and a chunk from Atlas §6.1 ("$250 flat") look almost identical
#      in embedding space. Cross-contract bleed = quoting the wrong party's
#      penalty. In a penalty dispute that is a career-limiting bug.
#   2. k-NN MYOPIA. Cosine finds text that LOOKS LIKE the query, not the
#      section that ANSWERS it. "What is the penalty cap?" embeds close to
#      every paragraph containing "penalty" — the per-day rate, the worked
#      example, the set-off clause — and the actual cap clause may rank 4th,
#      outside your top-3 context window.
#   3. EMBEDDING DRIFT & OPACITY. Swap the embedding model and every score
#      changes; nobody can explain WHY chunk 17 was retrieved. There is no
#      audit trail to show a client's legal team.
#
# PAGEINDEX (reasoning over a tree — https://pageindex.ai style):
#   parse each document's headings into a hierarchical tree (a ToC with
#   per-node summaries) → at query time an LLM READS THE OUTLINE and reasons:
#   "late-penalty question → SwiftShip MSA → §6 Late Delivery Penalties →
#   §6.2 Penalty Cap" — exactly how a human expert uses a table of contents.
#   Retrieval is a *navigation decision you can log and explain*, not a
#   geometric accident.
#
# WHERE IT SHINES: long, deeply structured, high-stakes documents — contracts,
# regulations, SEC filings, clinical protocols — where structure carries
# meaning and a wrong-section answer has real cost.
# WHERE VECTORS STILL WIN: huge heterogeneous corpora of short, unstructured
# text (support tickets, chat logs) where there is no ToC to navigate.
# A consultant recommends per-corpus, not per-fashion.
print("\n[Concept] chunk+embed+cosine vs PageIndex — see comments in this cell.")

# %% Build the index over data/contracts
# Offline: summaries = heading + first 240 chars (deterministic, free).
# Online : an LLM writes a 1-2 sentence summary per node (better navigation).
import time  # noqa: E402

from src.pageindex.tree_builder import build_index, load_index  # noqa: E402

t0 = time.perf_counter()
index = build_index()  # defaults: data/contracts → data/contracts_index.json
elapsed = time.perf_counter() - t0

n_docs = len(index["docs"])


def _count_nodes(nodes: list[dict]) -> int:
    return sum(1 + _count_nodes(n.get("children", [])) for n in nodes)


n_nodes = sum(_count_nodes(d["nodes"]) for d in index["docs"])
print(f"Indexed {n_docs} documents into {n_nodes} tree nodes "
      f"in {elapsed:.2f}s (persisted to data/contracts_index.json).")
assert n_docs == 6, "expected the 6 contracts from ARCHITECTURE.md §3.3"

# Round-trip check: the persisted JSON reloads to the same shape.
reloaded = load_index()
assert len(reloaded["docs"]) == n_docs

# %% Pretty-print one document's tree — the SwiftShip MSA
# This IS the index: titles + summaries, nested like a ToC. An LLM navigating
# this tree sees exactly what you see below — no embeddings anywhere.
swift = next(d for d in index["docs"] if "swiftship" in d["doc"].lower())
print(f"\nDocument: {swift['doc']}  —  {swift['title']}\n")


def print_tree(nodes: list[dict], indent: int = 0) -> None:
    for node in nodes:
        pad = "    " * indent
        print(f"{pad}[{node['node_id']}] {node['title']}")
        summary = " ".join((node.get("summary") or "").split())
        if summary:
            print(f"{pad}    > {summary[:100]}")
        print_tree(node.get("children", []), indent + 1)


print_tree(swift["nodes"])

# %% The corpus-wide outline — what the navigating LLM gets as its "map"
# retriever.outline() renders every document's tree as prompt-ready text.
# In lab07 this exact string is handed to the contracts_analyst agent.
from src.pageindex.retriever import PageIndexRetriever, RetrievedSection  # noqa: E402

retriever = PageIndexRetriever()
outline = retriever.outline()
print(f"\noutline() → {len(outline):,} chars covering {n_docs} docs. First 1200:\n")
print(outline[:1200])
print("... [truncated] ...")

# %% Offline lexical retrieval — three business questions
# Offline, search() scores token overlap over title+summary+text. Inspect the
# RetrievedSection dataclass: node_id, doc, title, path, text, score —
# `doc` + `path` are your CITATION; `score` is your honesty about confidence.
QUERIES: list[str] = [
    "What penalty applies when SwiftShip delivers a shipment late?",
    "What are the payment terms in the Atlas Freight agreement?",
    "Compare the force majeure provisions across the carrier contracts",
]


def show_hits(query: str, hits: list[RetrievedSection]) -> None:
    print(f"\nQUERY: {query!r}")
    for i, h in enumerate(hits, 1):
        flat = " ".join(h.text.split())
        print(f"  {i}. score={h.score:.3f}  doc={h.doc}")
        print(f"     path : {h.path}")
        print(f"     node : [{h.node_id}] {h.title}")
        print(f"     text : {flat[:160]}...")


for q in QUERIES:
    show_hits(q, retriever.search(q, top_k=3))

# 💡 CONSULTANT'S NOTE — read the scores, not just the hits:
#   Q1 lands cleanly in swiftship_express_msa.md §6 (2% per late business
#      day, 15% cap) — strong token overlap, lexical search at its best.
#   Q2 exposes k-NN myopia live: "payment terms" scores §10 'Termination',
#      §4 'Term' and §7 'Invoicing & Payment' in a 3-way TIE, because the
#      token "terms" matches "Term(ination)". The right section (§7, net 45
#      days, 1.0%/month interest) wins only by luck of ordering.
#   Q3 is a COMPARISON: one ranked list cannot "compare" six documents —
#      the agent layer (lab07) must issue per-document searches and
#      synthesize. Retrieval ≠ reasoning.

# %% 🎯 EXERCISE 6.1 — break lexical search
# Write ONE query about these contracts that (a) a human expert answers easily
# from the ToC, but (b) shares almost no vocabulary with the relevant section,
# so offline token-overlap retrieval misses or misranks it. Run it through
# retriever.search() and explain WHY it fails lexically and why LLM tree
# navigation would succeed. Try yours before unfolding the answer below.

# %% ✅ ANSWER 6.1 (fold this cell — VS Code: click the `# region` arrow)
# region ANSWER ----------------------------------------------------------
tricky = "Can GlobalTrade walk away from the SwiftShip deal early without giving a reason?"
hits = retriever.search(tricky, top_k=3)
show_hits(tricky, hits)
top_titles = " | ".join(h.title for h in hits)
print(f"\nTop-3 titles: {top_titles}")
print("""
WHY LEXICAL FAILS: the answer lives in SwiftShip MSA §10 'Termination' —
"GlobalTrade may terminate for convenience on ninety (90) days' written
notice". The query says "walk away", "deal", "early", "without giving a
reason" — near-zero overlap with "termination", "convenience", "written
notice". Look at the scores: ~0.30, 0.30, 0.28 — a near-tie, less than
half of Q1's 0.67. If §10 ranks first it is by accident of generic tokens
("GlobalTrade", "SwiftShip", "Party") that appear in EVERY section; the
ranker cannot distinguish the right answer from noise, and one synonym
swap ("exit the partnership"?) reshuffles the list. A retrieval system you
cannot trust on rephrasings is a retrieval system you cannot ship.

WHY TREE NAVIGATION SUCCEEDS: an LLM reading the outline performs a
*semantic mapping*, not a string match: "walk away early without reason"
→ that is 'termination for convenience' → open §10 of the SwiftShip MSA,
deliberately and reproducibly. The knowledge that resolves the query lives
in the NAVIGATOR (the LLM's world knowledge + the document's structure),
not in the index. That is the core PageIndex bet: keep the index dumb and
auditable, make the retrieval step intelligent.
""")
# endregion ---------------------------------------------------------------

# %% Wrap-up
# 💡 CONSULTANT'S NOTE — what you tell the client:
#   * The index is a ToC, not a vector blob: inspectable, diffable, versioned
#     alongside the contract file itself. Re-index = re-parse, deterministic.
#   * Every retrieval is a logged navigation path (doc → section → subsection)
#     you can replay in front of legal. Try doing that with cosine scores.
#   * Offline lexical search is the graceful-degradation floor; the SAME
#     PageIndexRetriever.search() interface upgrades to LLM navigation the
#     moment a key is present — labs 07 and 09 build on exactly that.
print("\n" + "=" * 72)
print("Lab 06 complete. Next: lab07 wraps this retriever in @tool functions")
print("and gives it to a ReAct agent with a grounding-and-citation contract.")
print("=" * 72)
