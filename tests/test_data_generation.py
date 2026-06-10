"""Contract tests for the data layer (CSV, contracts corpus, warehouse, catalog).

WHY THIS EXISTS
---------------
Every downstream module (ML pipeline, RAG, NL2SQL, the agents) trusts the shapes
promised in ARCHITECTURE.md §3. These tests pin that contract: column names and
target semantics for the CSV, file count/size/structure for the contracts, table
set + resolvable foreign keys for the warehouse, and the catalog's table/glossary
coverage. If a generator change breaks any consumer, it should fail HERE first,
with a message that names the broken promise — not three modules downstream.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CSV_PATH: Path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"
CONTRACTS_DIR: Path = PROJECT_ROOT / "data" / "contracts"
DB_PATH: Path = PROJECT_ROOT / "data" / "warehouse.db"
CATALOG_PATH: Path = PROJECT_ROOT / "data" / "metadata_catalog.json"

# §3.1 canonical column order — downstream code may rely on names, not order,
# but we pin both so accidental schema drift is loud.
CANONICAL_COLUMNS: list[str] = [
    "order_id", "order_date", "shipping_date", "scheduled_shipping_days",
    "actual_shipping_days", "shipping_mode", "carrier_name", "customer_id",
    "customer_segment", "market", "order_region", "order_country",
    "category_name", "product_name", "order_item_quantity", "sales",
    "discount", "profit", "late_delivery",
]

# §3.3 contract corpus — exact filenames.
CONTRACT_FILES: list[str] = [
    "swiftship_express_msa.md",
    "atlas_freight_msa.md",
    "pacific_crest_carriers_msa.md",
    "nordhaul_logistics_msa.md",
    "meridian_supply_procurement.md",
    "helios_components_procurement.md",
]


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ----------------------------------------------------------------------------------
# Flat CSV (§3.1)
# ----------------------------------------------------------------------------------

def test_csv_row_count(orders: pd.DataFrame) -> None:
    assert 11_000 <= len(orders) <= 13_000, f"expected ~12,000 rows, got {len(orders)}"


def test_csv_canonical_columns(orders: pd.DataFrame) -> None:
    assert list(orders.columns) == CANONICAL_COLUMNS


def test_csv_order_ids_unique_from_10000(orders: pd.DataFrame) -> None:
    assert orders["order_id"].is_unique
    assert orders["order_id"].min() == 10_000


def test_late_rate_around_30pct(orders: pd.DataFrame) -> None:
    late_rate = orders["late_delivery"].mean()
    assert 0.25 <= late_rate <= 0.35, f"late rate {late_rate:.3f} outside [0.25, 0.35]"


def test_target_consistent_with_shipping_days(orders: pd.DataFrame) -> None:
    derived = (orders["actual_shipping_days"] > orders["scheduled_shipping_days"]).astype(int)
    assert (orders["late_delivery"] == derived).all(), (
        "late_delivery must equal (actual_shipping_days > scheduled_shipping_days) on every row"
    )


def test_scheduled_days_follow_mode(orders: pd.DataFrame) -> None:
    expected = {"Same Day": 1, "First Class": 2, "Second Class": 4, "Standard Class": 6}
    by_mode = orders.groupby("shipping_mode")["scheduled_shipping_days"].unique()
    assert set(by_mode.index) == set(expected)
    for mode, values in by_mode.items():
        assert list(values) == [expected[mode]], f"{mode}: scheduled days {values}"


def test_planted_signal_direction(orders: pd.DataFrame) -> None:
    """The documented drivers must actually show up, or the ML labs fall flat."""
    by_mode = orders.groupby("shipping_mode")["late_delivery"].mean()
    assert by_mode["Same Day"] > by_mode["Standard Class"]
    by_carrier = orders.groupby("carrier_name")["late_delivery"].mean()
    assert by_carrier.idxmax() == "SwiftShip Express"
    assert by_carrier.idxmin() == "NordHaul Logistics"
    by_market = orders.groupby("market")["late_delivery"].mean()
    assert by_market["Africa"] > by_market["USCA"]


# ----------------------------------------------------------------------------------
# Contracts corpus (§3.3)
# ----------------------------------------------------------------------------------

def test_six_contracts_exist_with_substance() -> None:
    for filename in CONTRACT_FILES:
        path = CONTRACTS_DIR / filename
        assert path.exists(), f"missing contract: {filename}"
        text = path.read_text(encoding="utf-8")
        assert len(text) > 4_000, f"{filename}: only {len(text)} chars — too thin for RAG"
        assert "## " in text, f"{filename}: no '## ' markdown sections"
        assert "On-Time Delivery Commitment" in text, f"{filename}: missing SLA subsection"


def test_contract_numbers_are_distinct() -> None:
    """Each contract must be quotably different — same SLA text twice breaks the quiz."""
    swiftship = (CONTRACTS_DIR / "swiftship_express_msa.md").read_text(encoding="utf-8")
    assert "96.0%" in swiftship and "15%" in swiftship  # flagship demo numbers
    sla_lines = set()
    for filename in CONTRACT_FILES:
        text = (CONTRACTS_DIR / filename).read_text(encoding="utf-8")
        commitment = text.split("On-Time Delivery Commitment")[1][:400]
        sla_lines.add(commitment)
    assert len(sla_lines) == len(CONTRACT_FILES), "two contracts share identical SLA wording"


# ----------------------------------------------------------------------------------
# Warehouse (§3.2)
# ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def warehouse() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_warehouse_has_four_tables(warehouse: sqlite3.Connection) -> None:
    rows = warehouse.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert {r[0] for r in rows} == {"orders", "customers", "products", "carriers"}


def test_warehouse_row_counts(warehouse: sqlite3.Connection, orders: pd.DataFrame) -> None:
    n_orders = warehouse.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert n_orders == len(orders)
    for table in ("customers", "products", "carriers"):
        assert warehouse.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    assert warehouse.execute("SELECT COUNT(*) FROM carriers").fetchone()[0] == 4


@pytest.mark.parametrize("dim_table, fk", [
    ("customers", "customer_id"),
    ("products", "product_id"),
    ("carriers", "carrier_id"),
])
def test_foreign_keys_resolve(warehouse: sqlite3.Connection, dim_table: str, fk: str) -> None:
    """Spot-join: every FK value in orders must exist in its dimension table."""
    orphans = warehouse.execute(
        f"SELECT COUNT(*) FROM orders o LEFT JOIN {dim_table} d ON o.{fk} = d.{fk} "
        f"WHERE d.{fk} IS NULL"
    ).fetchone()[0]
    assert orphans == 0, f"{orphans} orders rows have a dangling {fk}"


def test_normalization_is_lossless(warehouse: sqlite3.Connection, orders: pd.DataFrame) -> None:
    """Joining back through the dimensions must reproduce the flat file's facts."""
    row = warehouse.execute(
        "SELECT c.carrier_name, p.product_name, cu.customer_segment "
        "FROM orders o "
        "JOIN carriers c ON o.carrier_id = c.carrier_id "
        "JOIN products p ON o.product_id = p.product_id "
        "JOIN customers cu ON o.customer_id = cu.customer_id "
        "WHERE o.order_id = ?", (int(orders.iloc[0]["order_id"]),)
    ).fetchone()
    expected = orders.iloc[0]
    assert row == (expected["carrier_name"], expected["product_name"],
                   expected["customer_segment"])


# ----------------------------------------------------------------------------------
# Metadata catalog (§3.4)
# ----------------------------------------------------------------------------------

def test_catalog_structure() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert catalog["service"]["name"] == "supply_chain_warehouse"
    table_names = {t["name"] for t in catalog["tables"]}
    assert table_names == {"orders", "customers", "products", "carriers"}
    for table in catalog["tables"]:
        assert table["description"], f"{table['name']}: empty table description"
        assert len(table["sampleQueries"]) >= 2, f"{table['name']}: needs >= 2 sample queries"
        for column in table["columns"]:
            assert column["description"], f"{table['name']}.{column['name']}: empty description"
    assert len(catalog["glossary"]) >= 6
    terms = {entry["term"] for entry in catalog["glossary"]}
    assert {"Late Delivery", "On-Time SLA", "Scheduled Shipping Days", "Market"} <= terms
