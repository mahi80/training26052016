"""Feature engineering for late-delivery prediction.

WHY THIS EXISTS
---------------
Raw order rows contain the *signal* for lateness, but not in a shape a model can
use. This module turns domain knowledge into columns — the highest-leverage step
in any classical-ML project (far more than swapping algorithms):

- **Seasonality**: the generator plants a Q4/holiday late-spike, so we expose
  ``order_month``, ``order_weekday``, ``is_weekend``, ``is_q4``.
- **Schedule pressure**: a Same-Day promise (1 scheduled day) is far harder to hit
  than Standard (6 days). ``mode_schedule_tightness`` normalizes scheduled days per
  shipping mode into a 0–1 "how tight is this promise?" score (1.0 = tightest mode).
- **Order size**: lateness rises with quantity, but not linearly — ``quantity_bucket``
  bins it so a linear model can pick up the step-change.

THE GOLDEN RULE — NO LEAKAGE
----------------------------
``shipping_date`` and ``actual_shipping_days`` are only known *after* delivery and
the target is derived from them. This function never reads, writes, or derives from
them; if they are present in the input they pass through untouched (and are dropped
later by ``load_orders(drop_leakage=True)`` / ``train_model``).

The three module constants below are the *contract* between feature engineering,
``preprocess.build_preprocessor`` (which scales/encodes them) and ``explain`` (which
reports importance against them). Change a feature → change it in exactly one place.
"""

from __future__ import annotations

import pandas as pd

# --- The feature contract (used by preprocess.py and explain.py) -------------------
NUMERIC_FEATURES: list[str] = [
    "scheduled_shipping_days",
    "order_item_quantity",
    "sales",
    "discount",
    "profit",
    "order_month",
    "order_weekday",
    "is_weekend",
    "is_q4",
    "mode_schedule_tightness",
]

# order_country and product_name are deliberately excluded: hundreds of one-hot
# levels add noise + training time without beating market/order_region/category.
CATEGORICAL_FEATURES: list[str] = [
    "shipping_mode",
    "carrier_name",
    "customer_segment",
    "market",
    "order_region",
    "category_name",
    "quantity_bucket",
]

TARGET: str = "late_delivery"

# Columns that must never be used as features (see data_loader.LEAKAGE_COLUMNS).
_LEAKAGE = ("shipping_date", "actual_shipping_days")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered feature columns added.

    Adds: ``order_month``, ``order_weekday``, ``is_weekend``, ``is_q4``,
    ``quantity_bucket``, ``mode_schedule_tightness``.

    Pure function: the input frame is never mutated, and leakage columns are never
    read — only ``order_date``, ``order_item_quantity``, ``shipping_mode`` and
    ``scheduled_shipping_days`` feed the new columns.
    """
    out = df.copy()

    # --- calendar features (from the order date — known at order time) ---
    if "order_date" not in out.columns:
        raise KeyError(
            "engineer_features needs an 'order_date' column — "
            "load data via data_loader.load_orders() first."
        )
    dates = pd.to_datetime(out["order_date"], errors="coerce")
    out["order_month"] = dates.dt.month.astype("Int64").fillna(0).astype(int)
    out["order_weekday"] = dates.dt.weekday.astype("Int64").fillna(0).astype(int)
    out["is_weekend"] = (out["order_weekday"] >= 5).astype(int)
    out["is_q4"] = out["order_month"].isin([10, 11, 12]).astype(int)

    # --- order size bucket (categorical, so OneHotEncoder gives each band its own
    # weight — a linear model can then learn the non-linear "bulk orders are risky"
    # step that a single numeric column would smooth over) ---
    out["quantity_bucket"] = pd.cut(
        out["order_item_quantity"],
        bins=[0, 1, 3, float("inf")],
        labels=["single", "small_2_3", "bulk_4_plus"],
    ).astype(str)

    # --- schedule tightness: scheduled days normalized per shipping mode ---
    # Median scheduled days per mode (robust to noise in real DataCo data), then
    # tightest_mode_days / this_mode_days → Same Day=1.0 ... Standard Class≈0.17.
    # Higher = tighter promise = structurally harder to deliver on time.
    mode_median = out.groupby("shipping_mode")["scheduled_shipping_days"].transform(
        "median"
    )
    tightest = mode_median.min()
    out["mode_schedule_tightness"] = (tightest / mode_median).astype(float).round(4)

    # Belt-and-braces: prove we did not invent leakage columns.
    assert all(c in df.columns or c not in out.columns for c in _LEAKAGE)
    return out
