"""WHY THIS EXISTS
---------------
The contracts analyst agent needs to *read contracts*, but an LLM cannot open
files — it can only call tools. These two tools expose the PageIndex-style
reasoning RAG (``src/pageindex``) to the agent:

- ``search_contracts``: navigate the hierarchical tree index (a machine-built
  table of contents) to the most relevant sections and return their full text.
  Online, an LLM reasons over the tree to pick sections; offline, a lexical
  scorer does — the tool's contract is identical either way, which is the
  point: agents should never know or care which retrieval mode is active.
- ``get_contract_outline``: the whole corpus's table of contents, so the agent
  can answer "which contracts do we have?" without a search.

DESIGN NOTE — the lazy singleton: building the tree index parses six markdown
contracts (and may call an LLM for summaries online). We do that at most once
per process, on *first use*, never at import time. Import must stay free of
side effects so ``import src.agents.tools_rag`` works in tests, offline, and
before the index file exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from src.config import PROJECT_ROOT

if TYPE_CHECKING:  # only for type hints — no import-time dependency on the sibling
    from src.pageindex.retriever import PageIndexRetriever

INDEX_PATH = PROJECT_ROOT / "data" / "contracts_index.json"

_retriever: "PageIndexRetriever | None" = None


def _get_retriever() -> "PageIndexRetriever":
    """Return the process-wide retriever, building the index on first use."""
    global _retriever
    if _retriever is None:
        if not INDEX_PATH.exists():
            from src.pageindex.tree_builder import build_index

            # Parses data/contracts/*.md into a tree index and persists it to
            # data/contracts_index.json. Offline-safe: summaries fall back to
            # heading + leading text when no LLM is available. LLM summaries
            # are a nice-to-have — if the LLM path fails for any reason
            # (misconfigured key, network), rebuild without it rather than
            # blocking retrieval entirely.
            try:
                build_index()
            except Exception:
                build_index(use_llm=False)
        from src.pageindex.retriever import PageIndexRetriever

        _retriever = PageIndexRetriever()
    return _retriever


@tool
def search_contracts(query: str) -> str:
    """Search the carrier and supplier contracts for relevant sections.

    Use for questions about late-delivery penalties, on-time SLAs, payment
    terms, liability, or termination clauses. Returns the top matching
    sections with their document name, section path, and full text — quote
    section numbers and exact figures from this text in your answer.
    """
    sections = _get_retriever().search(query, top_k=3)
    if not sections:
        return (
            "No contract sections matched that query. Try different wording, or "
            "call get_contract_outline to see which contracts and sections exist."
        )
    return "\n\n".join(f"### {s.doc} — {s.path}\n{s.text}" for s in sections)


@tool
def get_contract_outline() -> str:
    """Get the table of contents of every contract on file.

    Use to discover which carrier/supplier agreements exist and how they are
    organized (section numbers and titles) before searching for details.
    """
    return _get_retriever().outline()
