"""Build the hierarchical tree index ("table of contents") over the contracts.

WHY THIS EXISTS
---------------
This is step 1 of PageIndex-style RAG: turn each contract into a **tree** that mirrors
its own structure, instead of a flat bag of embedded chunks.

Contracts are *born structured* — numbered sections and subsections written precisely
so a human can navigate them. Chunk-and-embed RAG throws that structure away and then
spends an embedding model trying to recover relevance statistically. Here we keep it:

- every markdown heading (``#``, ``##``, ``###``) becomes a :class:`Node`;
- nesting follows heading levels, so ``### 6.1 Per-Shipment Penalty`` is a child of
  ``## 6. Late Delivery Penalties`` which is a child of the document root;
- each node stores its **own prose** (the text between its heading and the next
  heading) plus a short **summary** the retriever shows to the navigating LLM.

Summaries are the "scent" the LLM follows down the tree. Online (API key available)
they are LLM-written (≤40 words). Offline they degrade to *heading + the first 240
characters of the section* — cruder, but enough for the lexical fallback retriever.

The whole index persists as one small JSON file (``data/contracts_index.json``):
``{"docs": [{"doc": "<filename>", "title": ..., "nodes": [<Node dict>...]}]}`` with
nodes nested via ``children``. Compare that operational footprint with running a
vector database.

Node ids are stable and human-readable — ``swiftship_express_msa/6/6.1`` — so a
retrieval trace reads like a citation, not an opaque chunk hash.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_llm, is_offline

CONTRACTS_DIR: Path = PROJECT_ROOT / "data" / "contracts"
INDEX_PATH: Path = PROJECT_ROOT / "data" / "contracts_index.json"

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
_SUMMARY_CHARS = 240
_LLM_SUMMARY_PROMPT = (
    "You are indexing a procurement contract for retrieval. Summarize the section "
    "below in at most 40 words, keeping every concrete number (percentages, fees, "
    "day counts, caps) verbatim. Reply with the summary only.\n\n"
    "Section title: {title}\n\nSection text:\n{text}"
)


@dataclass
class Node:
    """One section of one contract — a node in the document's ToC tree."""

    node_id: str
    title: str
    level: int
    doc: str
    summary: str
    text: str
    children: list["Node"] = field(default_factory=list)


def _parse_sections(markdown: str) -> list[tuple[int, str, str]]:
    """Split markdown into a flat list of ``(level, title, own_prose)`` tuples.

    Only ``#``/``##``/``###`` are structural; deeper headings (rare) stay in the
    prose. ``own_prose`` is exactly the text between a heading and the next heading —
    parents do NOT absorb their children's text (the tree keeps them separate).
    """
    sections: list[tuple[int, str, list[str]]] = []
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            title = match.group(2).replace("**", "").strip()
            sections.append((len(match.group(1)), title, []))
        elif sections:
            sections[-1][2].append(line)
    return [(lvl, title, "\n".join(body).strip()) for lvl, title, body in sections]


def _build_tree(doc_name: str, doc_stem: str, sections: list[tuple[int, str, str]]) -> Node:
    """Nest the flat section list into a tree using heading levels.

    Node ids chain the numbered headings under the document stem, e.g.
    ``swiftship_express_msa/6/6.1``; unnumbered headings get a positional ``sN``.
    """
    root = Node(node_id=doc_stem, title=doc_stem, level=1, doc=doc_name, summary="", text="")
    stack: list[Node] = [root]
    saw_root = False
    for level, title, text in sections:
        if level == 1 and not saw_root:
            root.title, root.text = title, text
            saw_root = True
            continue
        level = max(level, 2)  # a stray second H1 is treated as a top-level section
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        num_match = _SECTION_NUM_RE.match(title)
        suffix = num_match.group(1) if num_match else f"s{len(parent.children) + 1}"
        node = Node(
            node_id=f"{parent.node_id}/{suffix}",
            title=title,
            level=level,
            doc=doc_name,
            summary="",
            text=text,
        )
        parent.children.append(node)
        stack.append(node)
    return root


def _heuristic_summary(node: Node) -> str:
    """Offline summary: heading + first 240 chars of the section's own prose.

    Parents whose prose lives entirely in subsections fall back to listing the
    subsection titles, so the outline still gives the retriever scent to follow.
    """
    body = node.text.strip() or ("Subsections: " + "; ".join(c.title for c in node.children))
    body = re.sub(r"\s+", " ", body.replace("**", "")).strip()
    return f"{node.title} — {body[:_SUMMARY_CHARS]}".rstrip(" —")


def _fill_summaries(node: Node, llm: Any | None) -> None:
    """Write a summary onto every node (recursively). LLM when available."""
    if llm is not None and node.text.strip():
        try:
            prompt = _LLM_SUMMARY_PROMPT.format(title=node.title, text=node.text[:2000])
            node.summary = str(llm.invoke(prompt).content).strip() or _heuristic_summary(node)
        except Exception:  # never let a flaky API call break index builds
            node.summary = _heuristic_summary(node)
    else:
        node.summary = _heuristic_summary(node)
    for child in node.children:
        _fill_summaries(child, llm)


def build_index(
    contracts_dir: Path | None = None,
    out_path: Path | None = None,
    use_llm: bool | None = None,
) -> dict:
    """Parse every contract into a tree, summarize nodes, persist + return the index.

    Args:
        contracts_dir: folder of ``*.md`` contracts (default ``data/contracts``).
        out_path: where to write the JSON index (default ``data/contracts_index.json``).
        use_llm: force LLM summaries on/off; ``None`` auto-detects via ``is_offline()``.

    Returns:
        The index dict — ``{"docs": [{"doc", "title", "nodes": [...]}, ...]}``.
    """
    contracts_dir = Path(contracts_dir) if contracts_dir is not None else CONTRACTS_DIR
    out_path = Path(out_path) if out_path is not None else INDEX_PATH
    if use_llm is None:
        use_llm = not is_offline()

    files = sorted(contracts_dir.glob("*.md")) if contracts_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No contracts found in {contracts_dir}. Generate them first by running "
            '`python "data/generate_contracts.py"` from the project root.'
        )

    llm = get_llm(temperature=0.0) if use_llm else None
    docs: list[dict] = []
    for path in files:
        root = _build_tree(path.name, path.stem, _parse_sections(path.read_text(encoding="utf-8")))
        _fill_summaries(root, llm)
        docs.append({"doc": path.name, "title": root.title, "nodes": [asdict(root)]})

    index = {"docs": docs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def load_index(path: Path | None = None) -> dict:
    """Load a previously built index JSON (default ``data/contracts_index.json``)."""
    path = Path(path) if path is not None else INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found at {path}. Build it first: "
            "`from src.pageindex import build_index; build_index()`."
        )
    return json.loads(path.read_text(encoding="utf-8"))
