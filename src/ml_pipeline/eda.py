"""Exploratory data analysis helpers for the late-delivery dataset.

WHY THIS EXISTS
---------------
Before any model, a consultant must answer four questions a stakeholder will ask in
the first meeting — and each function here answers exactly one of them:

- "Where is the problem concentrated?"      → ``late_rate_by(df, dimension)``
- "How imbalanced is the target?"           → ``class_balance(df)``  (drives the
  whole Week-3 imbalance lesson: ~30 % late means accuracy is a misleading metric)
- "Is it getting better or worse?"          → ``monthly_trend(df)``
- "What does the dataset even look like?"   → ``summary_stats(df)``

DESIGN RULE: every function returns a plain ``pd.DataFrame`` or a JSON-serializable
``dict`` — never a plot, never prints. That makes the same functions reusable from
(a) the lab notebooks, (b) unit tests, and (c) the ``data_analyst`` LangGraph agent,
whose tools must hand the LLM *text*, not matplotlib figures.
"""

from __future__ import annotations

import pandas as pd

TARGET = "late_delivery"


def late_rate_by(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Late-delivery rate broken down by one categorical dimension.

    Returns a DataFrame with columns ``[<dimension>, "n_orders", "late_rate"]``,
    sorted worst-first — the slide-ready "where does it hurt?" table.
    """
    if dimension not in df.columns:
        raise KeyError(
            f"Dimension {dimension!r} not in dataset. "
            f"Available columns: {sorted(df.columns)}"
        )
    out = (
        df.groupby(dimension, observed=True)[TARGET]
        .agg(n_orders="size", late_rate="mean")
        .reset_index()
        .sort_values("late_rate", ascending=False)
        .reset_index(drop=True)
    )
    out["late_rate"] = out["late_rate"].round(4)
    return out


def class_balance(df: pd.DataFrame) -> dict:
    """Target distribution: ``{"on_time": int, "late": int, "late_rate": float}``.

    With ~30 % positives, a model that predicts "on time" for everything scores
    ~70 % accuracy while catching zero late shipments — which is why training and
    evaluation downstream optimize *recall* (with a precision floor), not accuracy.
    """
    y = df[TARGET].astype(int)
    return {
        "on_time": int((y == 0).sum()),
        "late": int((y == 1).sum()),
        "late_rate": round(float(y.mean()), 4),
    }


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Order volume and late rate per calendar month.

    Returns columns ``["month", "n_orders", "late_rate"]`` with month as an ISO
    ``YYYY-MM`` string (JSON/markdown friendly), sorted chronologically. Look for
    the planted Q4 / holiday spike — it motivates the ``is_q4`` feature.
    """
    dates = pd.to_datetime(df["order_date"], errors="coerce")
    out = (
        df.assign(month=dates.dt.to_period("M").astype(str))
        .groupby("month")[TARGET]
        .agg(n_orders="size", late_rate="mean")
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )
    out["late_rate"] = out["late_rate"].round(4)
    return out


def summary_stats(df: pd.DataFrame) -> dict:
    """One JSON-serializable snapshot of the dataset for prompts and reports."""
    dates = pd.to_datetime(df["order_date"], errors="coerce")
    stats: dict = {
        "n_orders": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "date_min": str(dates.min().date()) if dates.notna().any() else None,
        "date_max": str(dates.max().date()) if dates.notna().any() else None,
        "late_rate": round(float(df[TARGET].mean()), 4),
        "shipping_modes": sorted(df["shipping_mode"].dropna().unique().tolist())
        if "shipping_mode" in df.columns
        else [],
        "markets": sorted(df["market"].dropna().unique().tolist())
        if "market" in df.columns
        else [],
        "n_carriers": int(df["carrier_name"].nunique())
        if "carrier_name" in df.columns
        else 0,
        "avg_sales": round(float(df["sales"].mean()), 2) if "sales" in df.columns else None,
        "avg_profit": round(float(df["profit"].mean()), 2) if "profit" in df.columns else None,
    }
    return stats
