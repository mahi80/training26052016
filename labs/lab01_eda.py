"""WHY THIS EXISTS
=================
LAB 01 — Exploratory Data Analysis: Where Do Late Shipments Hide?

Before anyone says "machine learning" in a client meeting, a consultant must be able
to answer: *which slices of the business are bleeding?* This lab uses the project's
contracted EDA functions (``src/ml_pipeline/eda.py``) — the same functions the
``data_analyst`` agent will call as tools in Week 4. Calling the contracted functions
IS the lesson: you learn the API surface your agents will use.

LEARNING OBJECTIVES
-------------------
- Load the canonical orders dataset through ``load_orders`` (never raw ``pd.read_csv``)
- Quantify class balance and understand why ~30% late is "imbalanced"
- Slice late rates by shipping mode, market, category and carrier
- Read a monthly trend and connect seasonality to operations reality
- Produce and save client-ready charts to ``labs/outputs/``

DURATION: ~60 minutes
PREREQUISITES: Python + pandas basics (Weeks 1-2). No API key needed — fully offline.
"""

# %% ── Banner: paths, offline status, data check ──────────────────────────────
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows consoles often default to cp1252 — force UTF-8 so emoji/box chars print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:  # script mode: this file lives in <root>/labs/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive `# %%` cell mode: run cells from the project root
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import is_offline  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "labs" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("LAB 01 — EDA on supply-chain orders")
print(f"Project root : {PROJECT_ROOT}")
print(f"OFFLINE mode : {is_offline()}")
print("API key needed: NO — every section of this lab runs fully offline.")
print("=" * 72)

_CSV = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"
if not _CSV.exists():
    print(f"Dataset missing: {_CSV}")
    print("Generate it first:  python data/generate_supply_chain_data.py")
    raise SystemExit(1)

# %% ── Matplotlib setup: headless-safe ─────────────────────────────────────────
# 💡 CONSULTANT'S NOTE: training rooms and CI servers often have no display.
# We render to files always, and only pop windows when LABS_SHOW=1 is set.
import matplotlib  # noqa: E402

if os.getenv("LABS_SHOW", "0") != "1":
    matplotlib.use("Agg")  # pure file rendering, never blocks a headless run

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)


def finish_chart(fig: plt.Figure, filename: str) -> None:
    """Save to labs/outputs/ and show only when LABS_SHOW=1 (headless-safe)."""
    out = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  chart saved → {out.relative_to(PROJECT_ROOT)}")
    if os.getenv("LABS_SHOW", "0") == "1":
        plt.show()
    else:
        plt.close(fig)


# %% ── 1. Load the data through the contract ──────────────────────────────────
# Everyone — labs, tests, agents — loads through load_orders(). One loader means
# one place to fix dtypes, parse dates, and (in lab02) drop leakage columns.
from src.ml_pipeline.data_loader import load_orders  # noqa: E402
from src.ml_pipeline.eda import (  # noqa: E402
    class_balance,
    late_rate_by,
    monthly_trend,
    summary_stats,
)

df = load_orders()
print(f"Loaded {len(df):,} orders × {df.shape[1]} columns")
print(df.head(3).to_string())

# %% ── 2. Summary statistics: the 10,000-foot view ─────────────────────────────
from pprint import pprint  # noqa: E402

stats = summary_stats(df)
pprint(stats)

# 💡 CONSULTANT'S NOTE: in a first client workshop you have ~5 minutes to prove you
# understand their data. Row count, date coverage, and the target rate are the three
# numbers that buy you credibility before you show a single model.

# %% ── 3. Class balance: the number that shapes Week 3 ─────────────────────────
balance = class_balance(df)
print(f"On-time orders : {balance['on_time']:,}")
print(f"Late orders    : {balance['late']:,}")
print(f"Late rate      : {balance['late_rate']:.1%}")

# ~30% late means a model that predicts "always on time" is ~70% accurate while
# catching ZERO late shipments. Hold that thought — it is the core of Lab 03.

# %% ── 4. Which segments scream risk? Late rate by dimension ───────────────────
for dim in ("shipping_mode", "market", "category_name", "carrier_name"):
    table = late_rate_by(df, dim).sort_values("late_rate", ascending=False)
    print(f"\n── late rate by {dim} " + "─" * (50 - len(dim)))
    print(table.to_string(index=False))

# 💡 CONSULTANT'S NOTE — reading these tables like an operator, not a data scientist:
#  * shipping_mode: Same Day / First Class promise 1-2 days. Tight promises break
#    first — the risk is in the COMMITMENT, not the distance.
#  * market: Africa and LATAM lanes run hot — infrastructure + customs variance.
#  * carrier_name: if one carrier (SwiftShip…) is consistently worst, that's not a
#    modeling insight, that's a CONTRACT conversation (see Week 4 contracts agent).
#  * category_name: flat late rates here tell you product mix is NOT the driver —
#    knowing what *doesn't* matter is billable insight too.

# %% ── 5. Monthly trend: seasonality is an operations story ─────────────────────
trend = monthly_trend(df)
print(trend.to_string(index=False))
# Look for the Q4 hump: peak season overloads carrier networks every year.

# %% ── 6. Chart 1 — late rate by shipping mode ─────────────────────────────────
mode_table = late_rate_by(df, "shipping_mode").sort_values("late_rate", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=mode_table, x="shipping_mode", y="late_rate", hue="shipping_mode",
            legend=False, palette="rocket", ax=ax)
ax.axhline(balance["late_rate"], ls="--", c="gray",
           label=f"overall {balance['late_rate']:.0%}")
ax.set_title("Late-delivery rate by shipping mode (tight promises break first)")
ax.set_ylabel("late rate")
ax.set_xlabel("")
ax.legend()
finish_chart(fig, "lab01_late_rate_by_mode.png")

# %% ── 7. Chart 2 — monthly late-rate trend ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(trend["month"].astype(str), trend["late_rate"], marker="o", lw=2)
ax.axhline(balance["late_rate"], ls="--", c="gray")
ax.set_title("Monthly late-delivery rate — spot the Q4 peak-season stress")
ax.set_ylabel("late rate")
ax.tick_params(axis="x", rotation=60)
finish_chart(fig, "lab01_monthly_trend.png")

# %% ── 8. Chart 3 — market × mode heatmap (where two risks stack) ───────────────
# This pivot is presentation-only logic, so it's OK to do inline (pipeline logic
# stays in src/). Two stacked risk factors = the cells your client should fear.
pivot = (
    df.groupby(["market", "shipping_mode"], observed=True)["late_delivery"]
    .mean()
    .unstack()
)
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds", ax=ax)
ax.set_title("Late rate: market × shipping mode — risk factors compound")
ax.set_xlabel("")
ax.set_ylabel("")
finish_chart(fig, "lab01_market_mode_heatmap.png")

# %% ── 9. 🎯 EXERCISE — find the next risk dimension ───────────────────────────
# TODO (5 min):
#   a) Compute the late rate by "customer_segment" and by "order_region".
#   b) Which single order_region has the highest late rate, and is its n_orders
#      large enough to act on? (A 60% late rate on 12 orders is noise, not signal.)
#
# Write your code below, then unfold the answer to compare.

# region 📁 ANSWER (unfold)
# seg = late_rate_by(df, "customer_segment").sort_values("late_rate", ascending=False)
# reg = late_rate_by(df, "order_region").sort_values("late_rate", ascending=False)
# print(seg.to_string(index=False))
# print(reg.head(5).to_string(index=False))
# worst = reg.iloc[0]
# print(f"Worst region: {worst.iloc[0]} at {worst['late_rate']:.0%} "
#       f"on n={worst['n_orders']:,} orders")
# # Rule of thumb: flag a segment only when n_orders is in the hundreds — otherwise
# # you are presenting sampling noise to a VP as if it were strategy.
# endregion

# %% ── 10. Narrative wrap-up: which segments scream risk? ──────────────────────
print(
    """
WHAT WE'D TELL THE CLIENT AFTER LAB 01
--------------------------------------
1. ~30% of orders ship late — concentrated, not uniform.
2. Tight-promise modes (Same Day, First Class) are the #1 risk slice.
3. Africa & LATAM lanes underperform every mode they touch.
4. One carrier drags the average down → contract/SLA review (Week 4 agent).
5. Q4 spikes every year → staff and route BEFORE peak, not during.

Next: Lab 02 turns these slices into features — after we hunt down leakage.
"""
)
