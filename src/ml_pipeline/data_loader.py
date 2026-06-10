"""Load the canonical supply-chain orders dataset (synthetic or real Kaggle DataCo).

WHY THIS EXISTS
---------------
Every ML engagement starts with three unglamorous-but-critical loader decisions, and
this module makes each one explicit so trainees see them as *design choices*:

1. **One canonical schema.** All downstream code (EDA, features, training, the
   ml_engineer agent) speaks the column names in ARCHITECTURE.md §3.1. Whether the
   bytes on disk come from our 12k-row synthetic generator or the real 180k-row
   Kaggle "DataCo Smart Supply Chain" CSV, the DataFrame leaving ``load_orders``
   looks identical. ``DATACO_COLUMN_MAP`` is the translation layer.

2. **Leakage is handled at the door.** ``shipping_date`` and
   ``actual_shipping_days`` are only knowable *after* delivery — a model trained on
   them would score perfectly in the lab and be useless in production (the target
   ``late_delivery`` is literally derived from ``actual_shipping_days``). Passing
   ``drop_leakage=True`` removes them before they can contaminate a feature set.
   This is the single most common real-world ML failure mode; we drill it early.

3. **Dates are parsed once, here.** Downstream feature engineering assumes
   ``order_date`` is a real ``datetime64``, not a string.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

# Canonical location of the synthetic dataset (built by data/generate_supply_chain_data.py).
DEFAULT_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"

# Columns only knowable AFTER delivery → never legal as model inputs.
LEAKAGE_COLUMNS: list[str] = ["shipping_date", "actual_shipping_days"]

# Columns to parse to datetime when present.
DATE_COLUMNS: list[str] = ["order_date", "shipping_date"]

# Real Kaggle DataCo headers → canonical schema (ARCHITECTURE.md §3.1).
# Trainees can download DataCoSupplyChainDataset.csv and pass its path to
# ``load_orders`` — the rename below makes the rest of the pipeline "just work".
# (DataCo has no carrier column; the ColumnTransformer in preprocess.py tolerates
# missing categorical columns for exactly this reason.)
DATACO_COLUMN_MAP: dict[str, str] = {
    "Order Id": "order_id",
    "order date (DateOrders)": "order_date",
    "shipping date (DateOrders)": "shipping_date",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Days for shipping (real)": "actual_shipping_days",
    "Shipping Mode": "shipping_mode",
    "Customer Id": "customer_id",
    "Customer Segment": "customer_segment",
    "Market": "market",
    "Order Region": "order_region",
    "Order Country": "order_country",
    "Category Name": "category_name",
    "Product Name": "product_name",
    "Order Item Quantity": "order_item_quantity",
    "Sales": "sales",
    "Order Item Discount Rate": "discount",
    "Order Profit Per Order": "profit",
    "Late_delivery_risk": "late_delivery",
}


def load_orders(
    path: str | Path | None = None, drop_leakage: bool = False
) -> pd.DataFrame:
    """Load the orders dataset into the canonical schema.

    Parameters
    ----------
    path:
        CSV to load. Defaults to ``data/raw/supply_chain_orders.csv``. May also be
        the raw Kaggle DataCo export — its headers are renamed via
        ``DATACO_COLUMN_MAP`` automatically.
    drop_leakage:
        When True, drop ``shipping_date`` and ``actual_shipping_days`` — the
        post-delivery columns that would leak the answer into the features.
        **Always use True when building a feature matrix.**

    Returns
    -------
    pd.DataFrame with canonical column names and parsed date columns.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Orders CSV not found at {csv_path}. "
            "Run: python data/generate_supply_chain_data.py (from the project root)."
        )

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        # The real Kaggle DataCo export is latin-1 encoded.
        df = pd.read_csv(csv_path, encoding="latin-1")

    # Translate real-world (DataCo) headers to the canonical schema; a no-op for
    # the synthetic file, which is already canonical.
    rename = {old: new for old, new in DATACO_COLUMN_MAP.items() if old in df.columns}
    if rename:
        df = df.rename(columns=rename)

    if "late_delivery" not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'late_delivery' column after renaming — "
            "is this really an orders dataset? See ARCHITECTURE.md §3.1."
        )

    for col in DATE_COLUMNS:
        if col in df.columns:
            # errors="coerce": tolerate the odd malformed date in real exports
            # (DataCo uses 'M/D/YYYY HH:MM' strings) rather than crash the class.
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    if drop_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    return df
