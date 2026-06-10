"""Train/test splitting and the sklearn preprocessing ColumnTransformer.

WHY THIS EXISTS
---------------
Two ideas every consultant must internalize before touching an estimator:

1. **Stratified splitting.** With ~30 % late orders, a naive random split can hand
   you folds with materially different late-rates, making metrics noisy and
   threshold tuning unreliable. ``stratify=y`` keeps the class ratio identical in
   train and test — cheap insurance you should reach for by default on imbalanced
   targets.

2. **Preprocessing belongs INSIDE the model pipeline.** The ``ColumnTransformer``
   built here is fitted on training data only, *as step one of a Pipeline*
   (see train.py). Fitting a scaler/encoder on the full dataset before splitting is
   the subtle cousin of data leakage: test-set statistics bleed into training.
   Pipelines make that mistake impossible — and they ship to production as one
   serializable object (one ``joblib.dump``, one ``predict_proba``).

Implementation notes:
- ``OneHotEncoder(handle_unknown="ignore")`` — a category seen only at inference
  (new carrier, new region) encodes to all-zeros instead of crashing the service.
- ``sparse_output=False`` — HistGradientBoostingClassifier requires dense input;
  at ~12k rows the dense matrix is trivially small.
- Column lists are passed as *callables* that select whichever contract columns are
  present, so the same preprocessor works on the real Kaggle DataCo file (which has
  no ``carrier_name``) without edits.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.ml_pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split → ``(X_train, X_test, y_train, y_test)``.

    Expects an *engineered* frame (run ``features.engineer_features`` first).
    Only contract feature columns present in ``df`` enter X — leakage columns and
    identifiers are excluded by construction.
    """
    if TARGET not in df.columns:
        raise KeyError(f"Target column {TARGET!r} missing from DataFrame.")

    feature_cols = [
        c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns
    ]
    if "order_month" not in df.columns:
        raise KeyError(
            "Engineered columns missing — call features.engineer_features(df) "
            "before split_data(df)."
        )

    X = df[feature_cols]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


class _PresentColumns:
    """Column selector: of ``self.columns``, those actually present in X at fit time.

    A module-level class (not a closure) so the fitted pipeline stays picklable —
    ``joblib.dump`` cannot serialize ``<locals>`` functions.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def __call__(self, X: pd.DataFrame) -> list[str]:
        return [c for c in self.columns if c in X.columns]


def _present(columns: list[str]) -> Callable[[pd.DataFrame], list[str]]:
    return _PresentColumns(columns)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: StandardScaler on numerics, OneHotEncoder on categoricals.

    Designed to be step one of a ``Pipeline`` so it is fitted on training folds
    only — never on the held-out test set.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _present(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _present(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",  # ids / stray columns silently excluded from the model
        verbose_feature_names_out=False,
    )
