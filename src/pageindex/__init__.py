"""PageIndex-style vectorless RAG over the procurement contracts.

WHY THIS EXISTS
---------------
Most RAG tutorials teach one recipe: split documents into chunks, embed every chunk
into a vector, store the vectors in a vector DB, and at query time embed the question
and fetch the nearest chunks by cosine similarity. That recipe works, but it has real
costs consultants should know about: an embedding model and a vector store to operate,
chunk boundaries that slice clauses in half, and "semantic similarity" that often is
NOT the same thing as relevance ("penalty cap" and "liability cap" embed close together
but answer different questions).

PageIndex (by vectify-ai) is the counter-design this package teaches: **reasoning-based
retrieval with no vectors at all**. Two ingredients:

1. A **hierarchical tree index** per document — essentially a machine-readable table
   of contents, built from the document's own structure (markdown headings), where
   every node carries a short summary of its section (``tree_builder``).
2. At query time, an LLM **reads the tree and reasons its way to the right node** —
   exactly like a human expert who answers "what's SwiftShip's late penalty?" by
   flipping to *SwiftShip MSA → 6. Late Delivery Penalties → 6.1*, not by scanning
   every paragraph of every contract (``retriever``).

No vector DB, no embeddings, no chunking heuristics — and the retrieval decision is
*explainable*: the model can tell you which sections it chose and why. Offline (no API
key) the retriever degrades to transparent lexical scoring over the same tree, so the
whole package works in a classroom without credentials.

Public API (see ARCHITECTURE.md §4.4):
    build_index(...) / load_index(...)  — build & persist data/contracts_index.json
    PageIndexRetriever().search(query)  — returns list[RetrievedSection]
    PageIndexRetriever().outline()      — human-readable ToC of all contracts
"""

from __future__ import annotations

from src.pageindex.retriever import PageIndexRetriever, RetrievedSection
from src.pageindex.tree_builder import (
    CONTRACTS_DIR,
    INDEX_PATH,
    Node,
    build_index,
    load_index,
)

__all__ = [
    "CONTRACTS_DIR",
    "INDEX_PATH",
    "Node",
    "PageIndexRetriever",
    "RetrievedSection",
    "build_index",
    "load_index",
]
