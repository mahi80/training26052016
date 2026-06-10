"""Offline tests for the PageIndex-style vectorless RAG engine (src/pageindex).

WHY THIS EXISTS
---------------
These tests pin down the offline contract of ARCHITECTURE.md §4.4 — the part of the
RAG engine that must work in a classroom with NO API keys:

- ``build_index`` parses all 6 contracts into trees and persists the JSON index;
- node ids are unique (they double as citations, so collisions would be lies);
- the lexical fallback retriever actually ranks the *right* section first for the
  canonical demo question about SwiftShip's late-delivery penalty;
- ``outline()`` exposes a ToC covering every document (the supervisor's agents and
  the labs both lean on it).

``OFFLINE=1`` is forced for the whole module so results never depend on whether a
developer happens to have a key in their environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest

from src.config import PROJECT_ROOT
from src.pageindex import PageIndexRetriever, RetrievedSection, build_index

INDEX_PATH = PROJECT_ROOT / "data" / "contracts_index.json"
EXPECTED_DOCS = 6
MIN_NODES_PER_DOC = 5


@pytest.fixture(scope="module", autouse=True)
def force_offline() -> Iterator[None]:
    """Pin OFFLINE=1 for every test here, restoring the prior value afterwards."""
    previous = os.environ.get("OFFLINE")
    os.environ["OFFLINE"] = "1"
    yield
    if previous is None:
        os.environ.pop("OFFLINE", None)
    else:
        os.environ["OFFLINE"] = previous


@pytest.fixture(scope="module")
def index() -> dict:
    """Build the index once for the module (heuristic summaries, no LLM)."""
    return build_index()


@pytest.fixture(scope="module")
def retriever(index: dict) -> PageIndexRetriever:
    return PageIndexRetriever()


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(child) for child in node["children"])


def _collect_ids(node: dict, bucket: list[str]) -> None:
    bucket.append(node["node_id"])
    for child in node["children"]:
        _collect_ids(child, bucket)


def test_index_has_six_docs_each_with_min_nodes(index: dict) -> None:
    assert len(index["docs"]) == EXPECTED_DOCS
    for entry in index["docs"]:
        n_nodes = sum(_count_nodes(node) for node in entry["nodes"])
        assert n_nodes >= MIN_NODES_PER_DOC, f"{entry['doc']} has only {n_nodes} nodes"


def test_index_persisted_as_json(index: dict) -> None:
    assert INDEX_PATH.exists()
    on_disk = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert set(on_disk.keys()) == {"docs"}
    assert len(on_disk["docs"]) == EXPECTED_DOCS
    sample = on_disk["docs"][0]["nodes"][0]
    assert {"node_id", "title", "level", "doc", "summary", "text", "children"} <= set(sample)


def test_node_ids_are_unique(index: dict) -> None:
    all_ids: list[str] = []
    for entry in index["docs"]:
        for node in entry["nodes"]:
            _collect_ids(node, all_ids)
    duplicates = {nid for nid in all_ids if all_ids.count(nid) > 1}
    assert not duplicates, f"duplicate node ids: {duplicates}"


def test_search_returns_topk_sections_with_text(retriever: PageIndexRetriever) -> None:
    results = retriever.search("late delivery penalty cap", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RetrievedSection) for r in results)
    assert all(r.text.strip() for r in results), "every hit must carry section text"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "results must be ranked by score"


def test_swiftship_penalty_query_hits_right_contract(retriever: PageIndexRetriever) -> None:
    """The canonical demo question must land in the SwiftShip penalty section."""
    results = retriever.search("What penalty applies if SwiftShip delivers late?", top_k=3)
    assert results, "query returned no sections"
    top = results[0]
    assert top.doc == "swiftship_express_msa.md", f"top-1 came from {top.doc} ({top.path})"
    assert "2%" in top.text or "2 %" in top.text, "expected the 2%-per-late-day clause"


def test_outline_lists_all_documents(index: dict, retriever: PageIndexRetriever) -> None:
    toc = retriever.outline()
    for entry in index["docs"]:
        assert entry["title"] in toc, f"outline missing {entry['title']}"
