"""Classical-ML pipeline for late-delivery prediction — the ml_engineer's toolbox.

WHY THIS EXISTS
---------------
This package is the Week-3 spine of the course: a complete, production-shaped
scikit-learn workflow that the Week-4 ``ml_engineer`` LangGraph agent then wraps
as tools. Read the modules in pipeline order:

1. ``data_loader``  — canonical schema, DataCo header mapping, leakage dropping
2. ``eda``          — stakeholder-question analytics (late rate by X, trends, balance)
3. ``features``     — domain knowledge → columns; the NUMERIC/CATEGORICAL contract
4. ``preprocess``   — stratified split + ColumnTransformer (inside the Pipeline!)
5. ``train``        — model registry, 3-fold CV, joblib artifact persistence
6. ``evaluate``     — imbalance-aware metrics + decision-threshold tuning
7. ``explain``      — permutation importance (and optional SHAP)

Everything below is the public API — import from the package root:

    from src.ml_pipeline import train_model, evaluate_model, tune_threshold
"""

from src.ml_pipeline.data_loader import (
    DATACO_COLUMN_MAP,
    DEFAULT_DATA_PATH,
    LEAKAGE_COLUMNS,
    load_orders,
)
from src.ml_pipeline.eda import class_balance, late_rate_by, monthly_trend, summary_stats
from src.ml_pipeline.evaluate import evaluate_model, tune_threshold
from src.ml_pipeline.explain import HAS_SHAP, feature_drivers, shap_drivers
from src.ml_pipeline.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    engineer_features,
)
from src.ml_pipeline.preprocess import build_preprocessor, split_data
from src.ml_pipeline.train import MODEL_REGISTRY, MODELS_DIR, load_artifact, train_model

__all__ = [
    # data_loader
    "DATACO_COLUMN_MAP",
    "DEFAULT_DATA_PATH",
    "LEAKAGE_COLUMNS",
    "load_orders",
    # eda
    "late_rate_by",
    "class_balance",
    "monthly_trend",
    "summary_stats",
    # features
    "engineer_features",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET",
    # preprocess
    "split_data",
    "build_preprocessor",
    # train
    "MODEL_REGISTRY",
    "MODELS_DIR",
    "train_model",
    "load_artifact",
    # evaluate
    "evaluate_model",
    "tune_threshold",
    # explain
    "feature_drivers",
    "shap_drivers",
    "HAS_SHAP",
]
