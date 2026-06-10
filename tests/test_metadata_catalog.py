"""Contract tests for the OpenMetadata-style catalog client (§4.5).

WHY THIS EXISTS
---------------
The NL2SQL agent is only as grounded as the text this class renders. These tests
pin the promises the SQL tools (and the lab prompts) rely on: all four warehouse
tables are visible, schema text carries column-level semantics (not just names),
keyword search ranks the right table first, and the glossary surfaces the business
vocabulary ("Late Delivery") an analyst would actually use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata.catalog import MetadataCatalog  # noqa: E402

CATALOG_PATH: Path = PROJECT_ROOT / "data" / "metadata_catalog.json"
EXPECTED_TABLES: set[str] = {"orders", "customers", "products", "carriers"}


@pytest.fixture(scope="module")
def catalog() -> MetadataCatalog:
    return MetadataCatalog()


def test_lists_four_tables(catalog: MetadataCatalog) -> None:
    assert set(catalog.list_tables()) == EXPECTED_TABLES


def test_missing_catalog_file_raises_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_database"):
        MetadataCatalog(catalog_path=tmp_path / "nope.json")


def test_orders_schema_has_target_column_and_descriptions(catalog: MetadataCatalog) -> None:
    schema = catalog.get_table_schema("orders")
    assert "CREATE TABLE orders" in schema
    assert "late_delivery" in schema
    # Column *descriptions* must be rendered, not just names — that is the whole
    # point of a catalog. Check against the source JSON rather than hardcoding prose.
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    orders_entry = next(t for t in raw["tables"] if t["name"] == "orders")
    assert orders_entry["description"] in schema, "table description missing from schema text"
    late_col = next(c for c in orders_entry["columns"] if c["name"] == "late_delivery")
    assert late_col["description"] in schema, "column description missing from schema text"


def test_unknown_table_raises_keyerror_listing_options(catalog: MetadataCatalog) -> None:
    with pytest.raises(KeyError, match="orders"):
        catalog.get_table_schema("warehouse_facts")


def test_full_schema_is_prompt_ready(catalog: MetadataCatalog) -> None:
    full = catalog.get_full_schema()
    for table in EXPECTED_TABLES:
        assert f"CREATE TABLE {table}" in full
    assert "glossary" in full.lower(), "full schema must include the glossary"


def test_search_ranks_carriers_first_for_sla_question(catalog: MetadataCatalog) -> None:
    ranked = catalog.search("penalty carrier sla")
    assert ranked, "search returned nothing"
    assert ranked[0] == "carriers"
    assert set(ranked) <= EXPECTED_TABLES


def test_search_with_no_match_still_returns_tables(catalog: MetadataCatalog) -> None:
    ranked = catalog.search("zzz qqq xyzzy")
    assert set(ranked) == EXPECTED_TABLES, "agent must always get somewhere to look next"


def test_glossary_mentions_late_delivery(catalog: MetadataCatalog) -> None:
    glossary = catalog.get_glossary()
    assert "Late Delivery" in glossary
    assert "orders.late_delivery" in glossary, "glossary must map terms to columns"
