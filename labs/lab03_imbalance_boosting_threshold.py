"""WHY THIS EXISTS
=================
LAB 03 — Class Imbalance, Boosting, and the Threshold the Business Chooses

A model is not a deliverable; a DECISION RULE is. With ~30% late shipments,
accuracy flatters do-nothing models, so we optimize what the client actually
feels: recall (late shipments caught) at a precision the ops team can tolerate
(false alarms they'll chase). This lab compares three contracted models, tunes
the decision threshold, and turns feature importances into a VP-ready story.

LEARNING OBJECTIVES
-------------------
- Show numerically why accuracy lies when 30% of labels are positive
- Build intuition for ``class_weight="balanced"``
- Train and compare random_forest and hist_gb (sklearn's LightGBM-equivalent)
- Tune the decision threshold against a precision floor (``tune_threshold``)
- Translate ``feature_drivers`` output into one slide for a logistics VP

DURATION: ~90 minutes
PREREQUISITES: Labs 01-02. No API key needed — fully offline.
"""

# %% ── Banner: paths, offline status, data check ──────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles often default to cp1252 — force UTF-8 so emoji/box chars print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive cells: run from the project root
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import is_offline  # noqa: E402

print("=" * 72)
print("LAB 03 — Imbalance, boosting, threshold tuning")
print(f"OFFLINE mode : {is_offline()}")
print("API key needed: NO — every section of this lab runs fully offline.")
print("=" * 72)

if not (PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv").exists():
    print("Dataset missing — run:  python data/generate_supply_chain_data.py")
    raise SystemExit(1)

# %% ── 1. Why accuracy lies at 30% positives ───────────────────────────────────
import pandas as pd  # noqa: E402

from src.ml_pipeline.data_loader import load_orders  # noqa: E402
from src.ml_pipeline.eda import class_balance  # noqa: E402

balance = class_balance(load_orders(drop_leakage=True))
late_rate = balance["late_rate"]
print(f"Late rate: {late_rate:.1%}")
print(f'A "predict ON-TIME for everything" model scores {1 - late_rate:.1%} accuracy')
print("…while catching 0 of the late shipments (recall = 0%).")

# 💡 CONSULTANT'S NOTE: "97% accurate" sells in steering committees and fails in
# operations. Anchor the conversation on RECALL ("of the shipments that WILL be
# late, what share do we flag in advance?") and PRECISION ("when we raise an
# alarm, how often is it real?"). Those map 1:1 to money; accuracy does not.

# %% ── 2. class_weight="balanced" — the cheap first fix ────────────────────────
# Every model in MODEL_REGISTRY uses class weighting where supported. Intuition:
# misclassifying a LATE order costs ~(1/0.3) ≈ 3.3× more in the loss function
# than misclassifying an on-time one. The model stops treating the minority
# class as ignorable noise. No resampling, no synthetic data — one parameter.
from src.ml_pipeline.train import MODEL_REGISTRY, train_model  # noqa: E402

print(f"Models in the registry: {list(MODEL_REGISTRY)}")

# %% ── 3. Train the contenders ─────────────────────────────────────────────────
# random_forest: bagging — many deep trees on bootstrap samples, variance killer.
# hist_gb: HistGradientBoostingClassifier — sklearn's LightGBM-equivalent.
#   Boosting fits shallow trees SEQUENTIALLY, each correcting the last one's
#   errors; histogram binning makes it fast on wide one-hot matrices.
for name in ("logreg", "random_forest", "hist_gb"):
    info = train_model(name)
    print(
        f"  {name:<14} cv_recall={info['cv_recall']:.3f}  "
        f"cv_roc_auc={info['cv_roc_auc']:.3f}  "
        f"train_seconds={info['train_seconds']:.1f}"
    )

# %% ── 4. Compare on the held-out test set ─────────────────────────────────────
from src.ml_pipeline.evaluate import evaluate_model, tune_threshold  # noqa: E402

rows = []
for name in ("logreg", "random_forest", "hist_gb"):
    m = evaluate_model(name, threshold=0.5)
    rows.append({"model": name, **{k: v for k, v in m.items()
                                   if k != "confusion_matrix"}})
comparison = pd.DataFrame(rows).set_index("model").round(3)
print(comparison.to_string())

# How to read this: roc_auc/pr_auc rank RANKING quality (threshold-free);
# recall/precision/f1 describe one specific threshold (0.5). A model can win on
# AUC and still look bad at 0.5 — which is exactly why the next cell exists.

# %% ── 5. tune_threshold — let the BUSINESS pick the operating point ───────────
# The 0.5 default is a convention, not a law. We sweep thresholds and take the
# best recall subject to a precision floor the ops team agreed to: at least half
# of all alarms must be real (min_precision=0.5), or they'll ignore the system.
tuned = tune_threshold("hist_gb", min_precision=0.5)
print(tuned)

base = evaluate_model("hist_gb", threshold=0.5)
best = evaluate_model("hist_gb", threshold=tuned["threshold"])
print(f"\n threshold 0.50  → recall {base['recall']:.1%}  precision {base['precision']:.1%}")
print(f" threshold {tuned['threshold']:.2f}  → recall {best['recall']:.1%}  "
      f"precision {best['precision']:.1%}")

# %% ── 6. The recall-vs-precision tradeoff, in dollars ─────────────────────────
# Put illustrative costs on each error (numbers you'd get from the client):
#   missed late shipment (FN): contract penalty + churn risk  ≈ $400
#   false alarm (FP): an expeditor reviews/calls the carrier  ≈ $40
COST_FN, COST_FP = 400, 40
for label, m in (("default 0.50", base), (f"tuned {tuned['threshold']:.2f}", best)):
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    cost = fn * COST_FN + fp * COST_FP
    print(f"  {label:<13} FN={fn:>4}  FP={fp:>4}  expected error cost ≈ ${cost:>9,}")

# 💡 CONSULTANT'S NOTE: a missed late shipment costs ~10× a false alarm here, so
# trading some precision for recall is usually money-positive — UP TO the point
# where alarm fatigue sets in and the ops team stops trusting the queue. That
# ceiling is the min_precision floor: it is a negotiated, human constraint.
# The threshold is a BUSINESS decision wearing a technical costume.

# %% ── 7. feature_drivers — what to tell a logistics VP ────────────────────────
from src.ml_pipeline.explain import feature_drivers  # noqa: E402

drivers = feature_drivers("hist_gb", top_k=10)
print(drivers.to_string(index=False))

# Presenting drivers to a VP — three rules:
# 1. Translate features to operations language: "mode_schedule_tightness" becomes
#    "how aggressive the delivery promise is". Never show raw column names.
# 2. Importance = predictive contribution, NOT causation. Say "the model leans
#    on X", never "X causes lateness" — or someone will 'fix' X and blame you.
# 3. Pair every driver with an action: tight promises → re-tier the SLA;
#    carrier signal → renegotiate (the Week-4 contracts agent quotes the clause).

# %% ── 8. Optional: xgboost / SMOTE — guarded, environment-dependent ────────────
# The registry self-extends when xgboost is installed (ARCHITECTURE §4.3).
if "xgboost" in MODEL_REGISTRY:
    info = train_model("xgboost")
    m = evaluate_model("xgboost")
    print(f"xgboost: cv_roc_auc={info['cv_roc_auc']:.3f}  "
          f"test recall={m['recall']:.1%}  precision={m['precision']:.1%}")
else:
    print("xgboost not installed — skipping (hist_gb is the curriculum default).")

try:
    from imblearn.over_sampling import SMOTE  # noqa: F401
    print("imbalanced-learn available — try SMOTE as a stretch exercise:")
except ImportError:
    print("imbalanced-learn not installed — SMOTE note below is read-only.")
# SMOTE sketch (resampling alternative to class_weight):
#   SMOTE synthesizes minority samples by interpolating between neighbors.
#   Apply to TRAINING data only — resampling the test set is leakage — and on
#   one-hot data prefer SMOTENC, which respects categorical columns.
#   In practice on tabular problems: try class_weight + threshold tuning FIRST;
#   they're simpler, faster, and usually within noise of SMOTE.

# %% ── 9. 🎯 EXERCISE — tighten the precision floor ─────────────────────────────
# TODO (10 min): the ops team got stricter — they now demand min_precision=0.6.
#   a) Re-run tune_threshold("hist_gb", min_precision=0.6).
#   b) How much recall did the stricter floor cost?
#   c) Recompute the expected error cost. Was stricter cheaper?
#
# region 📁 ANSWER (unfold)
# strict = tune_threshold("hist_gb", min_precision=0.6)
# m6 = evaluate_model("hist_gb", threshold=strict["threshold"])
# print(f"recall {best['recall']:.1%} → {m6['recall']:.1%} at precision "
#       f"{m6['precision']:.1%}")
# (tn, fp), (fn, tp) = m6["confusion_matrix"]
# print(f"cost ≈ ${fn * COST_FN + fp * COST_FP:,} "
#       f"(vs ${best['confusion_matrix'][1][0] * COST_FN + best['confusion_matrix'][0][1] * COST_FP:,})")
# # Typical finding: stricter precision REDUCES alarms but the extra misses cost
# # more than the saved reviews — with FN:FP at 10:1 you usually want recall.
# endregion

# %% ── 10. Wrap-up ──────────────────────────────────────────────────────────────
print(
    """
LAB 03 TAKEAWAYS
----------------
1. At 30% positives, accuracy is marketing; recall & precision are operations.
2. class_weight="balanced" is the cheapest imbalance fix — start there.
3. Boosting (hist_gb) typically edges out bagging here; CHECK, don't assume.
4. The threshold is chosen by the business via a precision floor, not by 0.5.
5. Drivers are presented as actions, in ops language, with causation caveats.

Week 3 ML arc complete. Next: Lab 04 — LangGraph from first principles, where
these pipeline functions become TOOLS an agent can call on demand.
"""
)
