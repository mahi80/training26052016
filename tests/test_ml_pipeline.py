"""Tests for src/ml_pipeline — loader, EDA, features, train/evaluate/tune/explain.

WHY THIS EXISTS
---------------
These tests double as executable documentation of the ml_pipeline contract
(ARCHITECTURE.md §4.3). They run fully OFFLINE (no LLM, no network) and use
``logreg`` for the end-to-end path — it trains in seconds, while still exercising
the exact same Pipeline/artifact machinery as hist_gb and random_forest.

The end-to-end tests (train → evaluate → tune → explain) share a single
module-scoped training run via the ``logreg_run`` fixture so the model is fitted
once, not four times.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml_pipeline import (
    CATEGORICAL_FEATURES,
    DATACO_COLUMN_MAP,
    MODEL_REGISTRY,
    NUMERIC_FEATURES,
    TARGET,
    class_balance,
    engineer_features,
    evaluate_model,
    feature_drivers,
    late_rate_by,
    load_orders,
    monthly_trend,
    split_data,
    train_model,
    tune_threshold,
)

# ----------------------------------------------------------------------------------
# data_loader
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    return load_orders()


def test_load_orders_shape_and_schema(orders: pd.DataFrame) -> None:
    assert len(orders) > 1_000
    for col in ("order_id", "order_date", "shipping_mode", "late_delivery"):
        assert col in orders.columns
    # Dates parsed, target binary.
    assert pd.api.types.is_datetime64_any_dtype(orders["order_date"])
    assert set(orders["late_delivery"].unique()) <= {0, 1}


def test_load_orders_drops_leakage() -> None:
    df = load_orders(drop_leakage=True)
    assert "shipping_date" not in df.columns
    assert "actual_shipping_days" not in df.columns
    # The target itself must survive the leakage drop.
    assert TARGET in df.columns


def test_dataco_column_map_contract() -> None:
    # The two renames called out in ARCHITECTURE.md §3.1 explicitly.
    assert DATACO_COLUMN_MAP["Days for shipment (scheduled)"] == "scheduled_shipping_days"
    assert DATACO_COLUMN_MAP["Late_delivery_risk"] == "late_delivery"
    # Every mapped value must be a canonical schema column name.
    canonical = set(load_orders().columns)
    assert set(DATACO_COLUMN_MAP.values()) <= canonical


# ----------------------------------------------------------------------------------
# eda
# ----------------------------------------------------------------------------------


def test_late_rate_by_shipping_mode(orders: pd.DataFrame) -> None:
    out = late_rate_by(orders, "shipping_mode")
    assert list(out.columns) == ["shipping_mode", "n_orders", "late_rate"]
    assert out["n_orders"].sum() == len(orders)
    assert ((out["late_rate"] >= 0) & (out["late_rate"] <= 1)).all()


def test_class_balance_and_monthly_trend(orders: pd.DataFrame) -> None:
    balance = class_balance(orders)
    assert set(balance) == {"on_time", "late", "late_rate"}
    assert balance["on_time"] + balance["late"] == len(orders)
    assert 0.1 < balance["late_rate"] < 0.6  # planted ~30% late rate

    trend = monthly_trend(orders)
    assert list(trend.columns) == ["month", "n_orders", "late_rate"]
    assert len(trend) >= 12  # two years of data


# ----------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------


def test_engineer_features_adds_expected_columns(orders: pd.DataFrame) -> None:
    df = load_orders(drop_leakage=True)
    engineered = engineer_features(df)
    for col in (
        "order_month",
        "order_weekday",
        "is_weekend",
        "is_q4",
        "quantity_bucket",
        "mode_schedule_tightness",
    ):
        assert col in engineered.columns, f"missing engineered column {col}"

    # No leakage columns invented, input not mutated.
    assert "actual_shipping_days" not in engineered.columns
    assert "shipping_date" not in engineered.columns
    assert "order_month" not in df.columns

    # Sanity of values.
    assert engineered["order_month"].between(1, 12).all()
    assert set(engineered["is_q4"].unique()) <= {0, 1}
    tightness = engineered["mode_schedule_tightness"]
    assert ((tightness > 0) & (tightness <= 1)).all()
    # Same Day is the tightest schedule → tightness 1.0.
    same_day = engineered.loc[engineered["shipping_mode"] == "Same Day", "mode_schedule_tightness"]
    assert (same_day == 1.0).all()


def test_feature_contract_constants() -> None:
    assert TARGET == "late_delivery"
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert not overlap
    for leak in ("shipping_date", "actual_shipping_days"):
        assert leak not in NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_split_data_is_stratified() -> None:
    df = engineer_features(load_orders(drop_leakage=True))
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02  # stratified ratios match
    # Feature matrix contains only contract columns — never the target or ids.
    assert TARGET not in X_train.columns
    assert "order_id" not in X_train.columns


# ----------------------------------------------------------------------------------
# train → evaluate → tune → explain (end-to-end on the fast logreg baseline)
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def logreg_run() -> dict:
    """Train logreg exactly once for all end-to-end tests below (slow-ish: ~5-15s)."""
    return train_model("logreg")


@pytest.mark.slow
class TestEndToEnd:
    def test_registry_has_core_models(self) -> None:
        for name in ("logreg", "random_forest", "hist_gb"):
            assert name in MODEL_REGISTRY

    def test_train_model_returns_metrics_and_persists(self, logreg_run: dict) -> None:
        assert set(logreg_run) == {
            "model_name",
            "model_path",
            "cv_recall",
            "cv_roc_auc",
            "train_seconds",
        }
        assert logreg_run["model_name"] == "logreg"
        from pathlib import Path

        assert Path(logreg_run["model_path"]).exists()
        assert logreg_run["cv_recall"] > 0.55
        assert logreg_run["cv_roc_auc"] > 0.55

    def test_evaluate_model_keys_and_quality(self, logreg_run: dict) -> None:
        result = evaluate_model("logreg", threshold=0.5)
        for key in (
            "recall",
            "precision",
            "f1",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
            "threshold",
        ):
            assert key in result, f"missing evaluate_model key {key}"
        assert result["recall"] > 0.55
        assert result["roc_auc"] > 0.55
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2
        assert sum(sum(row) for row in cm) == result["n_test"]

    def test_tune_threshold_returns_valid_threshold(self, logreg_run: dict) -> None:
        result = tune_threshold("logreg", min_precision=0.5)
        assert 0.0 < result["threshold"] < 1.0
        for key in ("recall", "precision", "f1", "met_precision_floor"):
            assert key in result
        if result["met_precision_floor"]:
            assert result["precision"] >= 0.5
        else:
            assert result["note"]  # fallback must be flagged

    def test_feature_drivers_top_k(self, logreg_run: dict) -> None:
        top_k = 5
        drivers = feature_drivers("logreg", top_k=top_k)
        assert isinstance(drivers, pd.DataFrame)
        assert list(drivers.columns) == ["feature", "importance"]
        assert 0 < len(drivers) <= top_k
        # Importances sorted descending; top driver should genuinely matter.
        assert drivers["importance"].is_monotonic_decreasing
        assert drivers["importance"].iloc[0] > 0
