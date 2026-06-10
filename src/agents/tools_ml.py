"""WHY THIS EXISTS
---------------
LLM agents cannot call ``sklearn.fit()`` — they can only emit *tool calls* with
JSON arguments. This module is the adapter: each ``@tool`` function is a thin,
typed wrapper over ``src/ml_pipeline`` that

1. takes only JSON-friendly arguments (strings, ints, floats),
2. returns a **string** (markdown table / JSON dump) — never a DataFrame,
   because tool output is pasted straight into the LLM's context window,
3. catches predictable user errors (bad column name, missing model artifact)
   and returns a *helpful message string* instead of raising — a ReAct agent
   can read an error string and self-correct; an exception just kills the run.

The docstring of each tool is not decoration: ``create_agent`` sends it to the
LLM verbatim as the tool's description, so write it for the model.

Imports from ``src.ml_pipeline`` are deliberately *lazy* (inside functions):
this module stays importable even before the ML pipeline exists or when its
optional dependencies are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.config import PROJECT_ROOT

MODELS_DIR: Path = PROJECT_ROOT / "models"

# Risk bands for predict_order_risk: LOW < 0.3 <= MEDIUM < 0.6 <= HIGH.
LOW_RISK_MAX = 0.3
MEDIUM_RISK_MAX = 0.6

# Sensible defaults for the one-row prediction frame. Geography is nested
# (market -> region -> country), so we pick the most common plausible pair per
# market from the generated dataset rather than mixing incompatible values.
_MARKET_GEO: dict[str, tuple[str, str]] = {
    "Africa": ("Southern Africa", "South Africa"),
    "Europe": ("Western Europe", "Netherlands"),
    "LATAM": ("Caribbean", "Jamaica"),
    "Pacific Asia": ("Oceania", "Australia"),
    "USCA": ("East of USA", "United States"),
}

# Scheduled shipping days are determined by the mode (ARCHITECTURE.md §3.1).
_MODE_SCHEDULE: dict[str, int] = {
    "Same Day": 1,
    "First Class": 2,
    "Second Class": 4,
    "Standard Class": 6,
}


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table (LLM-friendly), with a fallback."""
    try:
        return df.to_markdown(index=False)
    except ImportError:  # tabulate missing — plain text still works
        return df.to_string(index=False)


def _json(obj: Any) -> str:
    """JSON-dump pipeline results, coercing numpy scalars/arrays to plain types."""

    def _default(o: Any) -> Any:
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    return json.dumps(obj, indent=2, default=_default)


def _risk_band(p_late: float) -> str:
    if p_late < LOW_RISK_MAX:
        return "LOW"
    if p_late < MEDIUM_RISK_MAX:
        return "MEDIUM"
    return "HIGH"


def _load_pipeline(model_name: str = "hist_gb") -> tuple[Any, str]:
    """Load the persisted sklearn pipeline; train on demand if the artifact is missing.

    Returns ``(fitted_pipeline, note)`` where ``note`` is "" when the artifact
    existed, or a human-readable explanation that we trained one just now.
    """
    import joblib

    path = MODELS_DIR / f"{model_name}.joblib"
    note = ""
    if not path.exists():
        from src.ml_pipeline.train import train_model

        result = train_model(model_name=model_name)
        note = (
            f"(No saved model at models/{model_name}.joblib — trained one on demand: "
            f"cv_recall={result.get('cv_recall')}, cv_roc_auc={result.get('cv_roc_auc')}.)\n"
        )
        path = Path(result.get("model_path", path))
    artifact = joblib.load(path)
    # train_model persists {"pipeline": ..., "split": {...}} (ARCHITECTURE.md §4.3).
    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    return pipeline, note


@tool
def get_data_overview() -> str:
    """Get the orders dataset shape, column names, and late/on-time class balance.

    Call this first when you need to know what data is available.
    """
    from src.ml_pipeline.data_loader import load_orders
    from src.ml_pipeline.eda import class_balance

    df = load_orders()
    balance = class_balance(df)
    return "\n".join(
        [
            f"Orders dataset: {len(df):,} rows x {df.shape[1]} columns.",
            "Columns: " + ", ".join(map(str, df.columns)),
            "Class balance (target=late_delivery): " + _json(balance),
        ]
    )


@tool
def analyze_late_rate(dimension: str) -> str:
    """Late-delivery rate broken down by one column of the orders dataset.

    Useful dimensions: shipping_mode, carrier_name, market, order_region,
    category_name, customer_segment. Returns a markdown table with
    n_orders and late_rate per value, sorted worst-first.
    """
    from src.ml_pipeline.data_loader import load_orders
    from src.ml_pipeline.eda import late_rate_by

    df = load_orders()
    if dimension not in df.columns:
        return (
            f"Unknown dimension '{dimension}'. Pick one of these columns: "
            + ", ".join(map(str, df.columns))
        )
    table = late_rate_by(df, dimension)
    if "late_rate" in table.columns:
        table = table.sort_values("late_rate", ascending=False)
    return _df_to_md(table.round(4))


@tool
def analyze_monthly_trend() -> str:
    """Monthly order volume and late-delivery rate over time, as a markdown table.

    Use this for seasonality questions (Q4 spikes, holiday effects, trends).
    """
    from src.ml_pipeline.data_loader import load_orders
    from src.ml_pipeline.eda import monthly_trend

    table = monthly_trend(load_orders())
    return _df_to_md(table.round(4))


@tool
def train_late_delivery_model(model_name: str = "hist_gb") -> str:
    """Train a late-delivery classifier and persist it to models/<name>.joblib.

    model_name is one of: logreg, random_forest, hist_gb (gradient boosting,
    the recommended default). Returns cross-validated recall / ROC-AUC and the
    artifact path as JSON.
    """
    from src.ml_pipeline.train import train_model

    try:
        result = train_model(model_name=model_name)
    except KeyError:
        return (
            f"Unknown model_name '{model_name}'. "
            "Use one of: logreg, random_forest, hist_gb."
        )
    return _json(result)


@tool
def evaluate_late_delivery_model(model_name: str = "hist_gb", threshold: float = 0.5) -> str:
    """Evaluate a trained model on the held-out test split at a decision threshold.

    Returns recall, precision, f1, roc_auc, pr_auc and the confusion matrix
    [[tn, fp], [fn, tp]] as JSON. Recall on the late class is the primary
    business metric. Train the model first if it does not exist yet.
    """
    from src.ml_pipeline.evaluate import evaluate_model

    if not (MODELS_DIR / f"{model_name}.joblib").exists():
        return (
            f"No trained model at models/{model_name}.joblib. "
            f"Call train_late_delivery_model(model_name='{model_name}') first."
        )
    return _json(evaluate_model(model_name=model_name, threshold=threshold))


@tool
def tune_decision_threshold(model_name: str = "hist_gb", min_precision: float = 0.5) -> str:
    """Find the decision threshold that maximizes recall subject to a precision floor.

    Late shipments trigger contract penalties, so we usually accept more false
    alarms (lower threshold) to catch more true lates. Returns the best
    threshold and its metrics as JSON.
    """
    from src.ml_pipeline.evaluate import tune_threshold

    if not (MODELS_DIR / f"{model_name}.joblib").exists():
        return (
            f"No trained model at models/{model_name}.joblib. "
            f"Call train_late_delivery_model(model_name='{model_name}') first."
        )
    return _json(tune_threshold(model_name=model_name, min_precision=min_precision))


@tool
def explain_model_drivers(model_name: str = "hist_gb", top_k: int = 10) -> str:
    """Top-k feature drivers of late delivery (permutation importance), as a table.

    Use this to answer 'WHY are shipments late / what drives the risk?'.
    Train the model first if it does not exist yet.
    """
    from src.ml_pipeline.explain import feature_drivers

    if not (MODELS_DIR / f"{model_name}.joblib").exists():
        return (
            f"No trained model at models/{model_name}.joblib. "
            f"Call train_late_delivery_model(model_name='{model_name}') first."
        )
    return _df_to_md(feature_drivers(model_name=model_name, top_k=top_k).round(4))


@tool
def predict_order_risk(
    shipping_mode: str,
    market: str,
    category_name: str,
    order_month: int,
    order_item_quantity: int = 1,
    carrier_name: str = "SwiftShip Express",
    customer_segment: str = "Consumer",
    sales: float = 200.0,
    discount: float = 0.05,
) -> str:
    """Score ONE hypothetical order for late-delivery risk.

    shipping_mode: 'Same Day' | 'First Class' | 'Second Class' | 'Standard Class'.
    market: 'Africa' | 'Europe' | 'LATAM' | 'Pacific Asia' | 'USCA'.
    order_month: 1-12. Unspecified fields get sensible defaults (region/country
    inferred from market, scheduled days from mode). Returns P(late) plus a
    risk band: LOW < 0.30 <= MEDIUM < 0.60 <= HIGH.
    """
    from src.ml_pipeline.features import engineer_features

    if not 1 <= int(order_month) <= 12:
        return f"order_month must be 1-12, got {order_month}."
    region, country = _MARKET_GEO.get(market, ("Western Europe", "Netherlands"))
    row = {
        "order_id": 999_999,
        # mid-month date in the dataset's range so month/weekday features derive cleanly
        "order_date": pd.Timestamp(year=2025, month=int(order_month), day=15),
        "scheduled_shipping_days": _MODE_SCHEDULE.get(shipping_mode, 4),
        "shipping_mode": shipping_mode,
        "carrier_name": carrier_name,
        "customer_id": 1,
        "customer_segment": customer_segment,
        "market": market,
        "order_region": region,
        "order_country": country,
        "category_name": category_name,
        "product_name": f"{category_name} item",
        "order_item_quantity": int(order_item_quantity),
        "sales": float(sales),
        "discount": float(discount),
        "profit": float(sales) * 0.10,
        "late_delivery": 0,  # placeholder — never read at predict time
    }
    frame = engineer_features(pd.DataFrame([row]))

    pipeline, note = _load_pipeline("hist_gb")
    # The ColumnTransformer inside the pipeline selects its feature columns by
    # name; restrict to the canonical feature list when available.
    try:
        from src.ml_pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

        features = frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    except (ImportError, KeyError):
        features = frame
    p_late = float(pipeline.predict_proba(features)[0, 1])
    band = _risk_band(p_late)

    echo = {k: str(v) for k, v in row.items() if k not in ("order_id", "late_delivery")}
    return (
        f"{note}Predicted late-delivery probability: {p_late:.1%} -> risk band: {band}\n"
        f"(bands: LOW < 30% <= MEDIUM < 60% <= HIGH)\n"
        f"Scored order: {json.dumps(echo, indent=2)}"
    )
