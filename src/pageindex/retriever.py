"""Reasoning-based retrieval over the contracts tree index — step 2 of PageIndex RAG.

WHY THIS EXISTS
---------------
With the tree index built (``tree_builder``), retrieval becomes *navigation*, not
similarity search. Ask "what penalty applies if SwiftShip delivers late?" and watch
how a human expert answers: open the SwiftShip MSA, scan the table of contents, jump
to "6. Late Delivery Penalties", read it. This module makes an LLM do exactly that:

ONLINE (API key set) — two-step LLM navigation:
    1. The LLM sees a compact outline of every contract (node ids + titles +
       summaries) and, via **structured output**, selects up to ``top_k`` node ids
       with a confidence each. Structured output matters: we get parseable ids back,
       not free text we'd have to regex.
    2. For any selected node that has children, the LLM is shown that subtree and may
       **drill one level deeper** to the most specific subsection (e.g. from
       "6. Late Delivery Penalties" down to "6.1 Per-Shipment Penalty").
    The full text of the chosen sections is then fetched from the tree and returned.

OFFLINE — transparent lexical fallback (classrooms without keys still work):
    token overlap between the query and each node's title/summary/text, weighted
    ``title*3 + summary*2 + text*1`` and normalized to [0, 1]. The document's own
    identity (title + filename) counts as context for every node — a query naming
    "SwiftShip" must prefer SwiftShip's contract, just as the LLM navigator sees
    which document each node belongs to.

Either way the result is a list of :class:`RetrievedSection` carrying the node id,
a human-readable path (a citation!), the full section text, and a score. Note what
is absent: no embedding model, no vector store, no chunk-size tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import get_llm, is_offline
from src.pageindex.tree_builder import INDEX_PATH, build_index, load_index

# Lexical-fallback weights per ARCHITECTURE.md §4.4: title*3 + summary*2 + text*1.
_W_TITLE, _W_SUMMARY, _W_TEXT = 3.0, 2.0, 1.0
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS: frozenset[str] = frozenset(
    "a an the of for to in on at by and or is are was were be been being it its as "
    "with that this these those what which who whom how when where why if then than "
    "do does did not no any all under per each from shall will would may might can "
    "could should has have had also into about other such".split()
)


def _stem(token: str) -> str:
    """Tiny plural-stripper so 'penalties' matches 'penalty' (no NLTK needed)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(
        _stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    )


def _token_match(query_tok: str, field_tokens: frozenset[str]) -> bool:
    """Exact match, or prefix match (≥4 chars) so 'deliver' ~ 'delivery'."""
    if query_tok in field_tokens:
        return True
    if len(query_tok) >= 4:
        return any(
            len(t) >= 4 and (t.startswith(query_tok) or query_tok.startswith(t))
            for t in field_tokens
        )
    return False


def _overlap(query_tokens: frozenset[str], field_tokens: frozenset[str]) -> float:
    """Fraction of query tokens found in the field — in [0, 1]."""
    if not query_tokens:
        return 0.0
    hits = sum(1 for q in query_tokens if _token_match(q, field_tokens))
    return hits / len(query_tokens)


@dataclass
class RetrievedSection:
    """One retrieved contract section — id, citation path, full text, score."""

    node_id: str
    doc: str
    title: str
    path: str
    text: str
    score: float


@dataclass
class _FlatNode:
    """Internal flattened view of one tree node, with precomputed token sets."""

    node_id: str
    doc: str
    title: str
    level: int
    path: str
    summary: str
    full_text: str  # own prose + all descendant sections (with their headings)
    title_tokens: frozenset[str]
    summary_tokens: frozenset[str]
    text_tokens: frozenset[str]  # own prose only — keeps leaf scoring precise
    child_ids: list[str] = field(default_factory=list)


class _NodeChoice(BaseModel):
    """One node the navigator picked, with how sure it is."""

    node_id: str = Field(description="A node_id copied verbatim from the outline.")
    confidence: float = Field(ge=0.0, le=1.0, description="Relevance, 0..1.")


class _Navigation(BaseModel):
    """Structured output schema for the LLM tree-navigation step."""

    reasoning: str = Field(description="One sentence on why these sections.")
    selections: list[_NodeChoice] = Field(description="Most relevant nodes first.")


class PageIndexRetriever:
    """Navigate the contracts tree to answer 'where is this written?' queries."""

    def __init__(self, index_path: Path | None = None) -> None:
        path = Path(index_path) if index_path is not None else INDEX_PATH
        self.index: dict = load_index(path) if path.exists() else build_index(out_path=path)
        self._nodes: dict[str, _FlatNode] = {}
        for entry in self.index["docs"]:
            doc_tokens = _tokenize(f"{entry['title']} {entry['doc'].replace('_', ' ')}")
            for node in entry["nodes"]:
                self._register(node, [], doc_tokens)

    # ------------------------------------------------------------------ indexing
    def _register(self, node: dict, crumbs: list[str], doc_tokens: frozenset[str]) -> str:
        """Flatten the tree depth-first; return the node's full markdown block."""
        my_crumbs = crumbs + [node["title"]]
        child_blocks = [self._register(c, my_crumbs, doc_tokens) for c in node["children"]]
        own = (node["text"] or "").strip()
        full_text = "\n\n".join(p for p in [own, *child_blocks] if p).strip()
        self._nodes[node["node_id"]] = _FlatNode(
            node_id=node["node_id"],
            doc=node["doc"],
            title=node["title"],
            level=node["level"],
            path=" > ".join(my_crumbs),
            summary=node["summary"],
            full_text=full_text,
            title_tokens=doc_tokens | _tokenize(node["title"]),
            summary_tokens=doc_tokens | _tokenize(node["summary"]),
            text_tokens=doc_tokens | _tokenize(own),
            child_ids=[c["node_id"] for c in node["children"]],
        )
        heading = "#" * node["level"] + " " + node["title"]
        return f"{heading}\n\n{full_text}" if full_text else heading

    def _to_section(self, flat: _FlatNode, score: float) -> RetrievedSection:
        return RetrievedSection(
            node_id=flat.node_id,
            doc=flat.doc,
            title=flat.title,
            path=flat.path,
            text=flat.full_text,
            score=round(score, 4),
        )

    # ------------------------------------------------------------------- public
    def search(self, query: str, top_k: int = 3) -> list[RetrievedSection]:
        """Return up to ``top_k`` relevant sections — LLM navigation when online."""
        if is_offline():
            return self._lexical_search(query, top_k)
        return self._llm_search(query, top_k)

    def outline(self) -> str:
        """Human-readable table of contents of every contract (for prompts/labs)."""
        lines: list[str] = []
        for entry in self.index["docs"]:
            lines.append(f"{entry['title']}  [{entry['doc']}]")
            for node in entry["nodes"]:
                self._outline_node(node, lines, with_summaries=False)
            lines.append("")
        return "\n".join(lines).strip()

    # --------------------------------------------------------------- offline path
    def _lexical_search(self, query: str, top_k: int) -> list[RetrievedSection]:
        """Weighted token-overlap ranking — the no-LLM stand-in for navigation."""
        q = _tokenize(query)
        scored: list[tuple[float, _FlatNode]] = []
        for flat in self._nodes.values():
            score = (
                _W_TITLE * _overlap(q, flat.title_tokens)
                + _W_SUMMARY * _overlap(q, flat.summary_tokens)
                + _W_TEXT * _overlap(q, flat.text_tokens)
            ) / (_W_TITLE + _W_SUMMARY + _W_TEXT)
            if score > 0.0:
                scored.append((score, flat))
        # Ties break toward the more specific (deeper) section, then stable by id.
        scored.sort(key=lambda pair: (-pair[0], -pair[1].level, pair[1].node_id))
        return [self._to_section(flat, score) for score, flat in scored[:top_k]]

    # ---------------------------------------------------------------- online path
    def _outline_node(self, node: dict, lines: list[str], with_summaries: bool) -> None:
        indent = "  " * (node["level"] - 1)
        if with_summaries:
            lines.append(f"{indent}[{node['node_id']}] {node['title']} :: {node['summary'][:160]}")
        else:
            lines.append(f"{indent}[{node['node_id']}] {node['title']}")
        for child in node["children"]:
            self._outline_node(child, lines, with_summaries)

    def _llm_outline(self, node_ids: list[str] | None = None) -> str:
        """Outline with ids + summaries; optionally only the given subtrees."""
        lines: list[str] = []
        if node_ids is None:
            for entry in self.index["docs"]:
                lines.append(f"DOCUMENT: {entry['title']}  [{entry['doc']}]")
                for node in entry["nodes"]:
                    self._outline_node(node, lines, with_summaries=True)
        else:
            for node_id in node_ids:
                flat = self._nodes[node_id]
                lines.append(f"[{flat.node_id}] {flat.title} :: {flat.summary[:160]}")
                for child_id in flat.child_ids:
                    child = self._nodes[child_id]
                    lines.append(f"  [{child.node_id}] {child.title} :: {child.summary[:160]}")
        return "\n".join(lines)

    def _llm_search(self, query: str, top_k: int) -> list[RetrievedSection]:
        """Two-step navigation: pick sections from the ToC, then drill one level."""
        navigator = get_llm(temperature=0.0).with_structured_output(_Navigation)
        step1 = navigator.invoke(
            "You navigate a table of contents of procurement contracts, like a human "
            "expert flipping to the right section. Select the node_ids of up to "
            f"{top_k} sections most likely to contain the answer.\n\n"
            f"QUESTION: {query}\n\nTABLE OF CONTENTS:\n{self._llm_outline()}"
        )
        chosen = {c.node_id: c.confidence for c in step1.selections if c.node_id in self._nodes}
        # Step 2 — drill down: re-decide among the chosen nodes AND their children.
        expandable = [nid for nid in chosen if self._nodes[nid].child_ids]
        if expandable:
            step2 = navigator.invoke(
                "Refine your selection: from the candidate sections and their "
                f"subsections below, select the up-to-{top_k} most specific node_ids "
                f"answering the question.\n\nQUESTION: {query}\n\n"
                f"CANDIDATES:\n{self._llm_outline(list(chosen))}"
            )
            refined = {c.node_id: c.confidence for c in step2.selections if c.node_id in self._nodes}
            if refined:
                chosen = refined
        if not chosen:  # navigator returned nothing usable — degrade gracefully
            return self._lexical_search(query, top_k)
        ranked = sorted(chosen.items(), key=lambda kv: -kv[1])[:top_k]
        return [self._to_section(self._nodes[nid], conf) for nid, conf in ranked]
