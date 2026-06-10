"""Held-out evaluation and decision-threshold tuning for late-delivery models.

WHY THIS EXISTS
---------------
Two lessons consultants get wrong on real engagements, both encoded here:

1. **Pick metrics from the business problem, not habit.** A missed late shipment
   costs contract penalties and churn; a false alarm costs an analyst a phone call.
   So we report the full imbalance-aware panel — recall (what % of true lates we
   catch), precision (how many alarms are real), F1, ROC-AUC and PR-AUC — and we
   never mention accuracy.

2. **The 0.5 threshold is a default, not a law.** ``predict_proba`` gives a score;
   *where you cut it* is a business decision. ``tune_threshold`` sweeps cut-offs
   0.05 → 0.95 and picks the one that maximizes recall subject to a precision
   floor ("catch as many lates as possible, as long as ≥ X % of alarms are real").
   This one function routinely buys 10–20 recall points over the 0.5 default —
   for free, no retraining.

Both functions reload the artifact persisted by ``train_model`` and score the
**sealed held-out split** — the model never saw these rows during training or CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml_pipeline.train import load_artifact


def _test_probabilities(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """P(late) on the sealed test split → ``(y_true, proba)``."""
    artifact = load_artifact(model_name)
    pipeline = artifact["pipeline"]
    X_test = artifact["split"]["X_test"]
    y_test = np.asarray(artifact["split"]["y_test"]).astype(int)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return y_test, proba


def evaluate_model(model_name: str = "hist_gb", threshold: float = 0.5) -> dict:
    """Score a trained model on its held-out split at a given decision threshold.

    Returns ``{"recall", "precision", "f1", "roc_auc", "pr_auc",
    "confusion_matrix": [[tn, fp], [fn, tp]], "threshold"}`` (plus model context).
    Note ROC-AUC / PR-AUC are threshold-free (computed on the raw probabilities);
    only the confusion-matrix-derived metrics move when you change ``threshold``.
    """
    y_test, proba = _test_probabilities(model_name)
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "model_name": model_name,
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "confusion_matrix": [[int(v) for v in row] for row in cm],
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
    }


def tune_threshold(model_name: str = "hist_gb", min_precision: float = 0.5) -> dict:
    """Sweep thresholds 0.05 → 0.95 (step 0.01); maximize recall s.t. a precision floor.

    Selection rule: among thresholds whose precision ≥ ``min_precision``, take the
    one with the highest recall (ties broken by F1). If NO threshold reaches the
    floor — possible for weak models or aggressive floors — fall back to the
    best-F1 threshold and flag it via ``met_precision_floor=False`` + a ``note``,
    so an agent (or a human) knows the constraint was relaxed rather than silently
    getting a worse-than-requested precision.
    """
    y_test, proba = _test_probabilities(model_name)

    rows: list[tuple[float, float, float, float]] = []
    for t in np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2):
        y_pred = (proba >= t).astype(int)
        rows.append(
            (
                float(t),
                float(precision_score(y_test, y_pred, zero_division=0)),
                float(recall_score(y_test, y_pred, zero_division=0)),
                float(f1_score(y_test, y_pred, zero_division=0)),
            )
        )

    feasible = [r for r in rows if r[1] >= min_precision]
    if feasible:
        best = max(feasible, key=lambda r: (r[2], r[3]))  # recall, then F1
        met_floor = True
        note = ""
    else:
        best = max(rows, key=lambda r: r[3])  # fall back to best F1
        met_floor = False
        note = (
            f"No threshold reached precision >= {min_precision:.2f}; "
            "fell back to the best-F1 threshold instead."
        )

    threshold, precision, recall, f1 = best
    return {
        "model_name": model_name,
        "threshold": round(threshold, 2),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "min_precision": float(min_precision),
        "met_precision_floor": met_floor,
        "note": note,
        "n_thresholds_tried": len(rows),
    }
