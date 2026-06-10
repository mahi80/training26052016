"""Builds the SQLite warehouse and its OpenMetadata-style catalog from the flat CSV.

WHY THIS EXISTS
---------------
The flat CSV is perfect for ML, but the Week-4 **NL2SQL agent** needs what real
analysts have: a *normalized warehouse* (fact table + dimensions) and a *metadata
catalog* describing it. This script plays the role of the ELT job + the catalog
ingestion pipeline:

1. **Normalize**: split the one-big-table CSV into ``orders`` (fact) plus
   ``customers``, ``products``, ``carriers`` (dimensions), exactly per
   ARCHITECTURE.md §3.2. Splitting is lossless because the generator guarantees one
   segment per customer and one category per product.
2. **Catalog**: emit ``data/metadata_catalog.json`` — table + column descriptions,
   tags, sample queries, and a business glossary. The JSON mirrors OpenMetadata's
   Table and GlossaryTerm entities, so ``src/metadata/catalog.py`` can later swap
   this file for a live OpenMetadata server (``OPENMETADATA_HOST_PORT`` + JWT)
   behind the same interface.

WHY A CATALOG MATTERS FOR NL2SQL (the Week-4 lesson)
----------------------------------------------------
An LLM writing SQL fails on real warehouses not because of SQL syntax but because it
doesn't know *what the columns mean* ("is late_delivery a flag or a count?", "which
table holds the SLA?"). Schema-grounding the agent with rich, human-written column
descriptions and a glossary is the single highest-leverage fix — that's exactly what
this catalog provides.

Idempotent: drops and recreates all tables on every run (safe to re-run).
Run: ``python data/build_database.py`` (from the project root).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CSV_PATH: Path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"
DB_PATH: Path = PROJECT_ROOT / "data" / "warehouse.db"
CATALOG_PATH: Path = PROJECT_ROOT / "data" / "metadata_catalog.json"

# carrier_id, carrier_name, contract_file, on_time_sla_pct
# NOTE: names match generate_supply_chain_data.CARRIERS; SLA values match the
# §3.3 contract texts produced by generate_contracts.py — keep all three in sync.
CARRIERS: list[tuple[int, str, str, float]] = [
    (1, "SwiftShip Express", "swiftship_express_msa.md", 96.0),
    (2, "Atlas Freight Co.", "atlas_freight_msa.md", 92.5),
    (3, "Pacific Crest Carriers", "pacific_crest_carriers_msa.md", 94.0),
    (4, "NordHaul Logistics", "nordhaul_logistics_msa.md", 97.5),
]

# Exact DDL from ARCHITECTURE.md §3.2 — the NL2SQL agent is tested against this shape.
DDL: list[str] = [
    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_segment TEXT);",
    "CREATE TABLE products  (product_id INTEGER PRIMARY KEY, product_name TEXT,\n"
    "                        category_name TEXT);",
    "CREATE TABLE carriers  (carrier_id INTEGER PRIMARY KEY, carrier_name TEXT,\n"
    "                        contract_file TEXT, on_time_sla_pct REAL);",
    """CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT, shipping_date TEXT,
    customer_id INTEGER REFERENCES customers(customer_id),
    product_id  INTEGER REFERENCES products(product_id),
    carrier_id  INTEGER REFERENCES carriers(carrier_id),
    shipping_mode TEXT,
    scheduled_shipping_days INTEGER, actual_shipping_days INTEGER,
    late_delivery INTEGER,
    order_item_quantity INTEGER, sales REAL, discount REAL, profit REAL,
    market TEXT, order_region TEXT, order_country TEXT
);""",
]


def _ensure_csv() -> None:
    """Generate the flat CSV first if it does not exist (standalone-run friendly)."""
    if CSV_PATH.exists():
        return
    try:  # package import (pytest / `python -m` from the project root)
        from data import generate_supply_chain_data
    except ImportError:  # direct script run from inside data/
        import generate_supply_chain_data  # type: ignore[no-redef]
    generate_supply_chain_data.main()


def build_warehouse(df: pd.DataFrame) -> dict[str, int]:
    """Drop & recreate the four tables, load them from the flat frame, return row counts."""
    customers = df[["customer_id", "customer_segment"]].drop_duplicates().sort_values("customer_id")
    if customers["customer_id"].duplicated().any():
        raise ValueError("customer_id maps to multiple segments — generator contract broken")

    products = (
        df[["product_name", "category_name"]].drop_duplicates().sort_values("product_name")
        .reset_index(drop=True)
    )
    products.insert(0, "product_id", products.index + 1)
    product_id_of = dict(zip(products["product_name"], products["product_id"]))
    carrier_id_of = {name: cid for cid, name, _, _ in CARRIERS}

    orders = pd.DataFrame({
        "order_id": df["order_id"],
        "order_date": df["order_date"],
        "shipping_date": df["shipping_date"],
        "customer_id": df["customer_id"],
        "product_id": df["product_name"].map(product_id_of),
        "carrier_id": df["carrier_name"].map(carrier_id_of),
        "shipping_mode": df["shipping_mode"],
        "scheduled_shipping_days": df["scheduled_shipping_days"],
        "actual_shipping_days": df["actual_shipping_days"],
        "late_delivery": df["late_delivery"],
        "order_item_quantity": df["order_item_quantity"],
        "sales": df["sales"],
        "discount": df["discount"],
        "profit": df["profit"],
        "market": df["market"],
        "order_region": df["order_region"],
        "order_country": df["order_country"],
    })
    if orders[["product_id", "carrier_id"]].isna().any().any():
        raise ValueError("unmapped product or carrier name — generator contract broken")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for table in ("orders", "customers", "products", "carriers"):
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        for statement in DDL:
            conn.execute(statement)
        customers.to_sql("customers", conn, if_exists="append", index=False)
        products.to_sql("products", conn, if_exists="append", index=False)
        conn.executemany("INSERT INTO carriers VALUES (?, ?, ?, ?);", CARRIERS)
        orders.to_sql("orders", conn, if_exists="append", index=False)
        conn.commit()
    return {"customers": len(customers), "products": len(products),
            "carriers": len(CARRIERS), "orders": len(orders)}


def _col(name: str, dtype: str, description: str, tags: list[str] | None = None) -> dict:
    """One OpenMetadata-style column entity."""
    return {"name": name, "dataType": dtype, "description": description, "tags": tags or []}


def build_catalog() -> dict:
    """Assemble the OpenMetadata-style catalog dict (§3.4)."""
    return {
        "service": {"name": "supply_chain_warehouse", "type": "sqlite",
                    "database": "data/warehouse.db"},
        "tables": [
            {
                "name": "orders",
                "description": (
                    "Fact table: one row per shipped order line, 2024-01-01 to 2025-12-31. "
                    "Grain = order_id. Joins to customers, products, and carriers via FK ids. "
                    "Primary asset for late-delivery analysis: late_delivery is the label "
                    "the ML pipeline predicts and the KPI operations tracks."
                ),
                "columns": [
                    _col("order_id", "INTEGER", "Surrogate key, unique per order line; starts at 10000.", ["PK"]),
                    _col("order_date", "TEXT", "Date the order was placed, ISO YYYY-MM-DD."),
                    _col("shipping_date", "TEXT",
                         "Date the order was actually delivered = order_date + actual_shipping_days. "
                         "Leakage for ML (encodes the outcome); fine for SQL reporting.", ["ml-leakage"]),
                    _col("customer_id", "INTEGER", "FK to customers.customer_id.", ["FK"]),
                    _col("product_id", "INTEGER", "FK to products.product_id.", ["FK"]),
                    _col("carrier_id", "INTEGER", "FK to carriers.carrier_id — which carrier moved the shipment.", ["FK"]),
                    _col("shipping_mode", "TEXT",
                         "Service tier: Same Day, First Class, Second Class, or Standard Class. "
                         "Determines scheduled_shipping_days (1/2/4/6 respectively)."),
                    _col("scheduled_shipping_days", "INTEGER",
                         "Promised transit time in days, fixed by shipping_mode: Same Day=1, "
                         "First Class=2, Second Class=4, Standard Class=6."),
                    _col("actual_shipping_days", "INTEGER",
                         "Realized transit time in days. Leakage for ML — together with the "
                         "schedule it determines the target.", ["ml-leakage"]),
                    _col("late_delivery", "INTEGER",
                         "1 if actual_shipping_days > scheduled_shipping_days, else 0. "
                         "The business KPI and the ML target.", ["target", "label"]),
                    _col("order_item_quantity", "INTEGER", "Units ordered on the line, 1-5."),
                    _col("sales", "REAL", "Gross sales value of the line in USD."),
                    _col("discount", "REAL", "Fractional discount applied, 0.00-0.25."),
                    _col("profit", "REAL", "Profit on the line in USD; can be negative."),
                    _col("market", "TEXT",
                         "Top level of the geo hierarchy: LATAM, Europe, Pacific Asia, USCA, Africa."),
                    _col("order_region", "TEXT", "Region nested under market (e.g. Western Europe)."),
                    _col("order_country", "TEXT", "Destination country nested under order_region."),
                ],
                "tags": ["fact", "core"],
                "sampleQueries": [
                    "SELECT shipping_mode, COUNT(*) AS n_orders, ROUND(AVG(late_delivery), 3) AS late_rate "
                    "FROM orders GROUP BY shipping_mode ORDER BY late_rate DESC",
                    "SELECT c.carrier_name, COUNT(*) AS n_orders, ROUND(AVG(o.late_delivery), 3) AS late_rate "
                    "FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id "
                    "GROUP BY c.carrier_name ORDER BY late_rate DESC",
                    "SELECT market, ROUND(SUM(sales), 2) AS total_sales, ROUND(SUM(profit), 2) AS total_profit "
                    "FROM orders GROUP BY market ORDER BY total_sales DESC",
                ],
            },
            {
                "name": "customers",
                "description": (
                    "Dimension: one row per customer. Currently a slim dimension carrying "
                    "the customer's segment; join from orders.customer_id."
                ),
                "columns": [
                    _col("customer_id", "INTEGER", "Customer surrogate key; starts at 1000.", ["PK"]),
                    _col("customer_segment", "TEXT", "Consumer, Corporate, or Home Office."),
                ],
                "tags": ["dimension"],
                "sampleQueries": [
                    "SELECT customer_segment, COUNT(*) AS n_customers FROM customers GROUP BY customer_segment",
                    "SELECT cu.customer_segment, ROUND(AVG(o.late_delivery), 3) AS late_rate "
                    "FROM orders o JOIN customers cu ON o.customer_id = cu.customer_id "
                    "GROUP BY cu.customer_segment",
                ],
            },
            {
                "name": "products",
                "description": (
                    "Dimension: one row per distinct product, with its category. "
                    "Join from orders.product_id."
                ),
                "columns": [
                    _col("product_id", "INTEGER", "Product surrogate key assigned during warehouse load.", ["PK"]),
                    _col("product_name", "TEXT", "Human-readable product name, unique."),
                    _col("category_name", "TEXT", "One of ~10 product categories (Electronics, Furniture, ...)."),
                ],
                "tags": ["dimension"],
                "sampleQueries": [
                    "SELECT category_name, COUNT(*) AS n_products FROM products GROUP BY category_name "
                    "ORDER BY n_products DESC",
                    "SELECT p.category_name, ROUND(AVG(o.late_delivery), 3) AS late_rate "
                    "FROM orders o JOIN products p ON o.product_id = p.product_id "
                    "GROUP BY p.category_name ORDER BY late_rate DESC",
                ],
            },
            {
                "name": "carriers",
                "description": (
                    "Dimension: one row per contracted carrier. Bridges the warehouse to the "
                    "contracts corpus: contract_file names the governing MSA in data/contracts/, "
                    "and on_time_sla_pct is the contractual on-time commitment from that MSA — "
                    "compare it to the observed late rate in orders to find SLA breaches."
                ),
                "columns": [
                    _col("carrier_id", "INTEGER", "Carrier surrogate key.", ["PK"]),
                    _col("carrier_name", "TEXT",
                         "SwiftShip Express, Atlas Freight Co., Pacific Crest Carriers, or NordHaul Logistics."),
                    _col("contract_file", "TEXT",
                         "Filename of the governing contract in data/contracts/ (markdown)."),
                    _col("on_time_sla_pct", "REAL",
                         "Contractual on-time delivery commitment in percent (e.g. 96.0 = 96%)."),
                ],
                "tags": ["dimension", "contracts-bridge"],
                "sampleQueries": [
                    "SELECT carrier_name, on_time_sla_pct, contract_file FROM carriers "
                    "ORDER BY on_time_sla_pct DESC",
                    "SELECT c.carrier_name, ROUND(100 * (1 - AVG(o.late_delivery)), 1) AS observed_on_time_pct, "
                    "c.on_time_sla_pct AS contractual_sla_pct "
                    "FROM carriers c JOIN orders o ON o.carrier_id = c.carrier_id "
                    "GROUP BY c.carrier_id ORDER BY observed_on_time_pct ASC",
                ],
            },
        ],
        "glossary": [
            {"term": "Late Delivery",
             "definition": ("An order whose realized transit time exceeded its promise: "
                            "actual_shipping_days > scheduled_shipping_days (late_delivery = 1). "
                            "The central KPI of this project and the ML target."),
             "mappedAssets": ["orders.late_delivery"]},
            {"term": "On-Time SLA",
             "definition": ("The contractual minimum percentage of shipments a carrier must deliver "
                            "on or before the scheduled date in a measurement period; falling below "
                            "it triggers the penalty/credit clauses of the carrier's contract."),
             "mappedAssets": ["carriers.on_time_sla_pct"]},
            {"term": "Scheduled Shipping Days",
             "definition": ("Promised transit time in days, fixed by shipping mode: Same Day=1, "
                            "First Class=2, Second Class=4, Standard Class=6."),
             "mappedAssets": ["orders.scheduled_shipping_days", "orders.shipping_mode"]},
            {"term": "Market",
             "definition": ("Top level of the geographic hierarchy market > region > country. "
                            "One of: LATAM, Europe, Pacific Asia, USCA, Africa."),
             "mappedAssets": ["orders.market", "orders.order_region", "orders.order_country"]},
            {"term": "Shipping Mode",
             "definition": ("The delivery service tier sold to the customer (Same Day, First Class, "
                            "Second Class, Standard Class). Tighter modes promise fewer days and "
                            "are empirically more often late."),
             "mappedAssets": ["orders.shipping_mode"]},
            {"term": "Carrier",
             "definition": ("The contracted logistics provider that physically moves a shipment. "
                            "Each carrier has a signed MSA in data/contracts/ governing penalties "
                            "and SLAs."),
             "mappedAssets": ["carriers.carrier_name", "orders.carrier_id"]},
            {"term": "Late Delivery Penalty",
             "definition": ("The contractual financial remedy owed when a carrier or supplier misses "
                            "its delivery commitment. Formulas differ by contract (per-day %, flat "
                            "fee, service credits, liquidated damages) — see the contract_file of "
                            "the relevant carrier."),
             "mappedAssets": ["carriers.contract_file"]},
        ],
    }


def main() -> None:
    """Build warehouse.db and metadata_catalog.json from the flat CSV."""
    _ensure_csv()
    df = pd.read_csv(CSV_PATH)
    counts = build_warehouse(df)
    print(f"Wrote {DB_PATH} — rows: " + ", ".join(f"{t}={n:,}" for t, n in counts.items()))

    catalog = build_catalog()
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    n_terms = len(catalog["glossary"])
    print(f"Wrote {CATALOG_PATH} — {len(catalog['tables'])} tables, {n_terms} glossary terms")


if __name__ == "__main__":
    main()
