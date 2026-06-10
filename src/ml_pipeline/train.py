"""Model registry, cross-validated training, and artifact persistence.

WHY THIS EXISTS
---------------
This is the heart of the Week-3 classical-ML arc. Three deliberate teaching choices:

1. **A model REGISTRY, not a model.** ``MODEL_REGISTRY`` maps a name to a factory.
   The ml_engineer agent (and you, in the labs) can say ``train_model("hist_gb")``
   without touching code — exactly how model selection should work on a client
   project. The three core entries cover the canonical progression:

   - ``logreg`` — the interpretable baseline you ALWAYS fit first.
   - ``random_forest`` — bagged trees: strong, low-tuning benchmark.
   - ``hist_gb`` — ``HistGradientBoostingClassifier``: sklearn's native equivalent
     of **LightGBM** (histogram-binned gradient-boosted trees: features are bucketed
     into 255 bins, so split-finding is O(bins) not O(rows) — same trick that makes
     LightGBM fast). Usually the accuracy winner on tabular data, with zero extra
     dependencies. If ``xgboost`` happens to be installed, it is registered too
     (guarded import) so the labs can compare — but nothing requires it.

2. **Imbalance handled at train time.** All factories use ``class_weight="balanced"``
   where the estimator supports it (sklearn ≥1.2 supports it on hist_gb). If a very
   old sklearn lacks it, ``train_model`` falls back to per-row ``sample_weight`` via
   ``compute_sample_weight("balanced", y)`` — mathematically the same correction.

3. **Train and evaluate are SEPARATE steps sharing one artifact.** ``train_model``
   persists a joblib dict — fitted pipeline + the held-out test split + feature
   names — to ``models/<name>.joblib``. ``evaluate.py`` and ``explain.py`` reload
   it, so evaluation is always on data the model never saw, even across separate
   agent tool calls or Python sessions.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.ml_pipeline.data_loader import LEAKAGE_COLUMNS, load_orders
from src.ml_pipeline.features import engineer_features
from src.ml_pipeline.preprocess import build_preprocessor, split_data

MODELS_DIR: Path = PROJECT_ROOT / "models"

# Does this sklearn's HistGradientBoostingClassifier accept class_weight?
# (Added in sklearn 1.2; True on the 1.8 used in class. The False branch keeps the
# code portable to older clients via the sample_weight fallback in train_model.)
_HIST_GB_HAS_CLASS_WEIGHT: bool = "class_weight" in inspect.signature(
    HistGradientBoostingClassifier.__init__
).parameters


def _make_hist_gb() -> HistGradientBoostingClassifier:
    if _HIST_GB_HAS_CLASS_WEIGHT:
        return HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        )
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE)


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    "logreg": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "hist_gb": _make_hist_gb,
}

# Optional extra: register xgboost only if importable (never a hard dependency).
try:  # pragma: no cover - environment dependent
    from xgboost import XGBClassifier

    MODEL_REGISTRY["xgboost"] = lambda: XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        # ≈ (1 - late_rate) / late_rate for our ~30% late rate — xgboost's
        # equivalent of class_weight="balanced".
        scale_pos_weight=2.3,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


def train_model(model_name: str = "hist_gb", df: pd.DataFrame | None = None) -> dict:
    """Cross-validate, fit, and persist one registry model.

    Steps (each one a Week-3 lesson):
    1. Load + drop leakage + engineer features (skipped if caller pre-engineered).
    2. Stratified 80/20 split — the 20 % test set is sealed into the artifact and
       only ever touched by evaluate/explain.
    3. 3-fold cross-validation on the TRAINING split, scoring recall and ROC-AUC
       (3 folds, not 5/10, to keep the classroom feedback loop under a minute).
    4. Fit ``Pipeline(preprocessor, model)`` on the full training split.
    5. Persist ``models/<name>.joblib``.

    Returns ``{"model_name", "model_path", "cv_recall", "cv_roc_auc",
    "train_seconds"}``.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )

    t0 = time.perf_counter()

    if df is None:
        df = load_orders(drop_leakage=True)
    else:
        df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    if "order_month" not in df.columns:  # not yet engineered
        df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", MODEL_REGISTRY[model_name]())]
    )

    # sample_weight fallback for estimators without class_weight (old-sklearn hist_gb).
    needs_sample_weight = (
        model_name == "hist_gb" and not _HIST_GB_HAS_CLASS_WEIGHT
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "roc_auc": "roc_auc"}
    if needs_sample_weight:  # pragma: no cover - only on sklearn < 1.2
        weights = compute_sample_weight("balanced", y_train)
        try:
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring,
                params={"model__sample_weight": weights},
            )
        except (TypeError, ValueError):
            # Very old sklearn: CV without weights (final fit below still weighted).
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring
            )
        pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring
        )
        pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.joblib"
    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "split": {
            "X_test": X_test,
            "y_test": y_test,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        },
        "feature_names": list(X_train.columns),
        "cv_recall": float(cv_results["test_recall"].mean()),
        "cv_roc_auc": float(cv_results["test_roc_auc"].mean()),
    }
    joblib.dump(artifact, model_path)

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "cv_recall": round(artifact["cv_recall"], 4),
        "cv_roc_auc": round(artifact["cv_roc_auc"], 4),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }


def load_artifact(model_name: str = "hist_gb") -> dict:
    """Reload a persisted training artifact (pipeline + sealed test split).

    Shared by evaluate.py and explain.py so every downstream metric is computed
    on exactly the held-out rows from training time.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained artifact at {model_path}. "
            f'Run train_model("{model_name}") first.'
        )
    return joblib.load(model_path)
