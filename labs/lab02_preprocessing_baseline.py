"""WHY THIS EXISTS
=================
LAB 02 — Leakage, Features, and an Honest Baseline

The fastest way to lose a client's trust is to demo a 99%-accurate model that
collapses in production. That failure has a name: **data leakage** — training on
columns that won't exist at prediction time. This lab makes you commit the crime,
catch yourself, then build a defensible baseline with the contracted pipeline
(``src/ml_pipeline``): features → stratified split → ColumnTransformer → logistic
regression.

LEARNING OBJECTIVES
-------------------
- Detect and remove leakage columns (``shipping_date``, ``actual_shipping_days``)
- Walk through ``engineer_features`` and explain each derived column
- Verify stratification in ``split_data`` (why class ratios must match)
- Dissect the ``build_preprocessor`` ColumnTransformer (OneHot + StandardScaler)
- Train ``logreg`` via ``train_model`` and read a confusion matrix out loud

DURATION: ~75 minutes
PREREQUISITES: Lab 01. No API key needed — fully offline.
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
print("LAB 02 — Preprocessing & baseline model")
print(f"OFFLINE mode : {is_offline()}")
print("API key needed: NO — every section of this lab runs fully offline.")
print("=" * 72)

if not (PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv").exists():
    print("Dataset missing — run:  python data/generate_supply_chain_data.py")
    raise SystemExit(1)

# %% ── 1. THE LEAKAGE LESSON — commit the crime first ─────────────────────────
# Load WITH the leakage columns still present (the default) and look closely.
import pandas as pd  # noqa: E402

from src.ml_pipeline.data_loader import load_orders  # noqa: E402

df_leaky = load_orders()  # drop_leakage defaults to False
cols = ["order_date", "shipping_date", "scheduled_shipping_days",
        "actual_shipping_days", "late_delivery"]
print(df_leaky[cols].head(8).to_string(index=False))

# Now the smoking gun: the target is LITERALLY computed from these columns.
reconstructed = (
    df_leaky["actual_shipping_days"] > df_leaky["scheduled_shipping_days"]
).astype(int)
match = (reconstructed == df_leaky["late_delivery"]).mean()
print(f"\n(actual > scheduled) reproduces the target on {match:.1%} of rows.")

# 💡 CONSULTANT'S NOTE — why a model trained with these columns is FRAUD:
# `shipping_date` / `actual_shipping_days` are only known AFTER the shipment
# arrives. At prediction time ("will order 104872 be late?") they don't exist yet.
# A model using them scores ~100% in your slide deck and ~coin-flip in production.
# Clients call that fraud; courts have, too. The fix is procedural, not heroic:
# for every feature ask "is this knowable at decision time?" — the *timestamp test*.

# %% ── 2. Drop the leakage — through the contract, not by hand ─────────────────
df = load_orders(drop_leakage=True)
dropped = set(df_leaky.columns) - set(df.columns)
print(f"Columns dropped by drop_leakage=True: {sorted(dropped)}")
assert "actual_shipping_days" not in df.columns
assert "shipping_date" not in df.columns
# The drop lives inside load_orders() so EVERY consumer (labs, tools, agents)
# inherits the fix. Leakage policy in one place — that's pipeline thinking.

# %% ── 3. engineer_features — turning EDA insight into columns ─────────────────
from src.ml_pipeline.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    engineer_features,
)

df_feat = engineer_features(df)
new_cols = sorted(set(df_feat.columns) - set(df.columns))
print(f"New engineered columns: {new_cols}")
print(df_feat[new_cols].head(5).to_string(index=False))

# Each new column encodes a Lab-01 finding:
#   order_month / is_q4 ............ the Q4 peak-season hump
#   order_weekday / is_weekend ..... weekend cutoffs change effective lead time
#   quantity_bucket ................ big orders are harder to fulfill on time
#   mode_schedule_tightness ........ Same Day's 1-day promise is the real risk
print(f"\nNUMERIC_FEATURES    : {NUMERIC_FEATURES}")
print(f"CATEGORICAL_FEATURES: {CATEGORICAL_FEATURES}")
print(f"TARGET              : {TARGET}")

# 💡 CONSULTANT'S NOTE: feature engineering is where domain interviews pay off.
# Every feature above came from a question an ops manager can answer in plain
# English. Features nobody can explain don't survive the client's model-risk review.

# %% ── 4. split_data — why "stratified" is non-negotiable ──────────────────────
from src.ml_pipeline.preprocess import build_preprocessor, split_data  # noqa: E402

X_train, X_test, y_train, y_test = split_data(df_feat)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")
print(f"Late rate — train: {y_train.mean():.3%}   test: {y_test.mean():.3%}")
# Near-identical rates = stratification worked. With a random (unstratified) split
# on an imbalanced target, your test set can drift a few points away from reality,
# and every metric you report afterwards is quietly biased.

# %% ── 5. build_preprocessor — anatomy of a ColumnTransformer ───────────────────
pre = build_preprocessor()
print(pre)
# Read it like a routing table:
#   numeric columns      → StandardScaler   (logreg needs comparable scales)
#   categorical columns  → OneHotEncoder(handle_unknown="ignore")
# handle_unknown="ignore" matters in production: a NEW carrier appearing next
# quarter encodes as all-zeros instead of crashing the pipeline at 2 a.m.

fitted = pre.fit(X_train)
n_out = len(fitted.get_feature_names_out())
print(f"\n{X_train.shape[1]} input columns → {n_out} model features after encoding")

# %% ── 6. Train the baseline: logistic regression ──────────────────────────────
# Why logreg first? It is fast, explainable, and sets the bar every fancier model
# must beat. Consultants who skip the baseline can't answer "compared to what?".
from src.ml_pipeline.train import train_model  # noqa: E402

train_info = train_model("logreg")
for k, v in train_info.items():
    print(f"  {k:>14}: {v}")
# Note cv_recall / cv_roc_auc come from cross-validation on TRAIN only.
# The test set stays untouched until evaluation — no peeking.

# %% ── 7. evaluate_model — reading the confusion matrix OUT LOUD ────────────────
from src.ml_pipeline.evaluate import evaluate_model  # noqa: E402

metrics = evaluate_model("logreg", threshold=0.5)
print({k: v for k, v in metrics.items() if k != "confusion_matrix"})

(tn, fp), (fn, tp) = metrics["confusion_matrix"]
total = tn + fp + fn + tp
print(f"""
CONFUSION MATRIX, NARRATED  (threshold = {metrics['threshold']})
                          predicted on-time   predicted LATE
  actually on-time   TN = {tn:>5,}            FP = {fp:>5,}
  actually LATE      FN = {fn:>5,}            TP = {tp:>5,}

* TP {tp:,} — late shipments we caught: ops can intervene (expedite, re-route).
* FN {fn:,} — late shipments we MISSED: penalties + angry customers. The
              expensive box. Recall = TP/(TP+FN) = {tp / (tp + fn):.1%}.
* FP {fp:,} — false alarms: wasted expediting effort. Precision = TP/(TP+FP)
              = {tp / (tp + fp):.1%}.
* Accuracy = (TN+TP)/total = {(tn + tp) / total:.1%} — and Lab 03 shows why that
  number is the least interesting one on this slide.
""")

# %% ── 8. 🎯 EXERCISE — the timestamp test ─────────────────────────────────────
# TODO (5 min): For each column below, decide: usable at prediction time, or leakage?
#   (a) scheduled_shipping_days   (b) discount   (c) profit   (d) order_date
# Hint: "profit" is booked when? Could it be revised after delivery?
#
# region 📁 ANSWER (unfold)
# (a) USABLE  — the promise is known the moment the order is placed.
# (b) USABLE  — the discount is set at order time.
# (c) GRAY ZONE — if profit is finalized post-delivery (returns, penalty charges),
#     it's partial leakage. In a real engagement you ASK the finance team how and
#     when this field is written. "It depends on the source system" is the single
#     most billable sentence in feature review.
# (d) USABLE  — known at order time; we only use derived parts (month, weekday).
# endregion

# %% ── 9. Wrap-up ───────────────────────────────────────────────────────────────
print(
    """
LAB 02 TAKEAWAYS
----------------
1. Leakage check FIRST — the timestamp test on every column.
2. One loader, one feature module: fixes propagate to every consumer.
3. Stratified split or your metrics lie by a few quiet points.
4. ColumnTransformer = preprocessing as code, deployable as one artifact.
5. logreg baseline: respectable recall, modest precision — the bar to beat.

Next: Lab 03 — why accuracy lies at 30% positives, boosting, and picking a
decision threshold the business can live with.
"""
)
