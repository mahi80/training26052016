"""Model explainability: which order attributes actually drive lateness?

WHY THIS EXISTS
---------------
A model the operations team cannot interrogate will not be adopted. "It's 87 %
ROC-AUC" convinces nobody; "Same-Day promises and the SwiftShip lanes drive most
of the late risk" changes carrier negotiations. This module produces that second
sentence.

WHY PERMUTATION IMPORTANCE (the default here)
---------------------------------------------
``permutation_importance`` shuffles one feature at a time on the **held-out** split
and measures how much ROC-AUC drops. Three properties make it the consultant-safe
default:
- *Model-agnostic*: identical code for logreg, forests, boosting, xgboost.
- *Honest*: computed on unseen data, so it measures what the model actually uses
  for generalization (tree impurity importances are biased toward high-cardinality
  features and are computed on training data).
- *Original-feature granularity*: because we permute the RAW columns and score the
  whole Pipeline (preprocessor included), importance lands on "shipping_mode" —
  not on 4 separate one-hot dummies nobody can present to a stakeholder.

Cost control for the classroom: a 1,500-row sample of the test split with
``n_repeats=5`` — seconds, not minutes, with stable rankings.

SHAP (optional extra): if the ``shap`` package is installed, ``shap_drivers`` gives
per-prediction additive attributions — the production-grade tool for "why was THIS
order flagged?". It is a guarded extra; nothing in the course requires it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE
from src.ml_pipeline.train import load_artifact

# Guarded optional dependency — never a hard import (ARCHITECTURE.md §7).
try:  # pragma: no cover - environment dependent
    import shap  # type: ignore  # noqa: F401

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

# Sample size for permutation importance: large enough for stable rankings,
# small enough to keep the lab feedback loop in seconds.
_PERM_SAMPLE_ROWS = 1_500
_PERM_N_REPEATS = 5


def feature_drivers(model_name: str = "hist_gb", top_k: int = 10) -> pd.DataFrame:
    """Top-k feature importances via permutation on the held-out split.

    Returns a DataFrame with columns ``["feature", "importance"]`` sorted by
    importance (mean ROC-AUC drop when the feature is shuffled), highest first.
    An importance near 0 means the model would do just as well without the column.
    """
    artifact = load_artifact(model_name)
    pipeline = artifact["pipeline"]
    X_test: pd.DataFrame = artifact["split"]["X_test"]
    y_test = artifact["split"]["y_test"]

    n = min(_PERM_SAMPLE_ROWS, len(X_test))
    X_sample = X_test.sample(n=n, random_state=RANDOM_STATE)
    y_sample = pd.Series(np.asarray(y_test), index=X_test.index).loc[X_sample.index]

    result = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        n_repeats=_PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )

    drivers = (
        pd.DataFrame(
            {
                "feature": list(X_sample.columns),
                "importance": np.round(result.importances_mean, 5),
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return drivers


def shap_drivers(
    model_name: str = "hist_gb", top_k: int = 10, n_samples: int = 300
) -> pd.DataFrame:
    """Optional SHAP-based drivers (requires ``pip install shap``).

    Returns ``["feature", "importance"]`` where importance = mean(|SHAP value|)
    over a test-split sample — i.e. the average absolute contribution of each
    *encoded* feature to individual predictions. Note the granularity difference
    vs ``feature_drivers``: SHAP attributes to post-encoding columns (one-hot
    dummies like ``shipping_mode_Same Day``), which is exactly what you want for
    per-order explanations.
    """
    if not HAS_SHAP:
        raise ImportError(
            "shap is not installed — this is an optional extra. "
            "Run `pip install shap` to use shap_drivers; feature_drivers() "
            "covers the curriculum without it."
        )

    artifact = load_artifact(model_name)
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_test: pd.DataFrame = artifact["split"]["X_test"]

    X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=RANDOM_STATE)
    X_enc = np.asarray(preprocessor.transform(X_sample))
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.Explainer(model, X_enc, feature_names=feature_names)
    values = explainer(X_enc).values
    if values.ndim == 3:  # (rows, features, classes) → take the "late" class
        values = values[:, :, -1]
    importance = np.abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": feature_names, "importance": np.round(importance, 5)})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
