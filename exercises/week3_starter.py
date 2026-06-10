"""WEEK 3 ASSESSMENT — "The Trial": classical ML + LangGraph fundamentals (300 XP).

WHY THIS EXISTS
---------------
The labs walked you through solutions; this assessment proves you can produce them
unaided. It is a gauntlet of 5 tasks covering everything Week 3 taught:

    TASK 1 (40 XP)  Leakage-safe data loading
    TASK 2 (50 XP)  EDA: find the worst carrier
    TASK 3 (80 XP)  Train a baseline model + tune the decision threshold for recall
    TASK 4 (60 XP)  Wrap an analysis function as a LangChain @tool
    TASK 5 (70 XP)  Build a 2-node LangGraph with a conditional keyword router

Each task is a function stub whose docstring tells you exactly what to build,
followed by a ``_check_*`` function (DO NOT EDIT) whose asserts you must make pass.
Run the file after every task — the trial stops at the first failure with a hint:

    python exercises/week3_starter.py        # run from the week3&4 project root

Everything here runs OFFLINE — no API key needed. TASK 5 deliberately uses a pure
keyword router function instead of an LLM so you can be graded without credentials:
the graph mechanics (state, nodes, conditional edges) are identical either way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

# Make `src` importable when this file is run directly from the project root
# (or from anywhere else — we resolve relative to this file, never the cwd).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# TASK 1 (40 XP) — Load a leakage-safe modelling frame
# ============================================================================

def load_clean_frame() -> pd.DataFrame:
    """Return the canonical orders dataset, safe to build features on.

    Requirements:
    - Use ``load_orders`` from ``src.ml_pipeline`` (NOT a raw ``pd.read_csv`` —
      the loader also parses dates and maps real-world DataCo headers).
    - The returned frame must contain NO post-delivery (leakage) columns:
      ``shipping_date`` and ``actual_shipping_days`` must be gone.
    - The target ``late_delivery`` must still be present, and ``order_date``
      must be a real datetime, not a string.

    Hint: the loader has a keyword argument that does the leakage drop for you.
    """
    raise NotImplementedError(
        "TASK 1: implement load_clean_frame() — "
        "from src.ml_pipeline import load_orders; call it with the keyword "
        "argument that drops the two post-delivery leakage columns."
    )


def _check_task1() -> None:
    df = load_clean_frame()
    assert isinstance(df, pd.DataFrame), "must return a pandas DataFrame"
    assert df.shape == (12000, 17), f"expected (12000, 17), got {df.shape}"
    assert "shipping_date" not in df.columns, "LEAKAGE: shipping_date still present"
    assert "actual_shipping_days" not in df.columns, (
        "LEAKAGE: actual_shipping_days still present — the target is derived "
        "from it, a model could just read off the answer"
    )
    assert "late_delivery" in df.columns, "you dropped the TARGET, not the leakage"
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"]), (
        "order_date must be datetime — did you bypass load_orders?"
    )


# ============================================================================
# TASK 2 (50 XP) — EDA: which carrier hurts us most?
# ============================================================================

def worst_carrier(df: pd.DataFrame) -> tuple[str, float]:
    """Return ``(carrier_name, late_rate)`` for the WORST carrier.

    Requirements:
    - Reuse ``late_rate_by`` from ``src.ml_pipeline`` (don't re-derive the
      groupby by hand — on an engagement you reuse the vetted analytics layer).
    - "Worst" = highest late_rate. Return a plain ``(str, float)`` tuple.
    """
    raise NotImplementedError(
        "TASK 2: implement worst_carrier(df) — "
        "from src.ml_pipeline import late_rate_by; break down by 'carrier_name' "
        "and return the name + late_rate of the top (worst) row."
    )


def _check_task2() -> None:
    name, rate = worst_carrier(load_clean_frame())
    assert isinstance(name, str) and isinstance(rate, float), (
        f"expected (str, float), got ({type(name).__name__}, {type(rate).__name__})"
    )
    assert name == "SwiftShip Express", (
        f"got {name!r} — the planted signal makes one carrier clearly worst. "
        "Did you sort ascending by mistake?"
    )
    assert 0.55 <= rate <= 0.70, f"late_rate {rate} outside plausible range"


# ============================================================================
# TASK 3 (80 XP) — Train a baseline + tune the threshold for recall
# ============================================================================

def train_and_tune() -> dict[str, Any]:
    """Train logistic regression, evaluate at threshold 0.4, tune the threshold.

    Requirements (all from ``src.ml_pipeline``):
    1. ``train_model("logreg")``        — fits + persists models/logreg.joblib.
    2. ``evaluate_model("logreg", threshold=0.4)`` — held-out metrics at 0.4.
    3. ``tune_threshold("logreg", min_precision=0.5)`` — best-recall threshold
       subject to a 50% precision floor.

    Return ``{"recall_at_040": <float from step 2>, "tuned": <dict from step 3>}``.

    The business logic: missed late shipments cost penalties, so we maximize
    recall — but with a precision floor so ops doesn't drown in false alarms.
    """
    raise NotImplementedError(
        "TASK 3: implement train_and_tune() — "
        "from src.ml_pipeline import train_model, evaluate_model, tune_threshold; "
        "train 'logreg', evaluate at threshold=0.4, tune with min_precision=0.5."
    )


def _check_task3() -> None:
    out = train_and_tune()
    assert set(out) >= {"recall_at_040", "tuned"}, f"missing keys, got {sorted(out)}"
    assert out["recall_at_040"] >= 0.60, (
        f"recall {out['recall_at_040']} < 0.60 at threshold 0.4 — did you "
        "evaluate the right model at the right threshold?"
    )
    tuned = out["tuned"]
    assert tuned["met_precision_floor"] is True, "tuner could not hold the floor?"
    assert tuned["precision"] >= 0.5, f"precision {tuned['precision']} below floor"
    assert 0.05 <= tuned["threshold"] <= 0.95, "threshold outside the sweep range"


# ============================================================================
# TASK 4 (60 XP) — Wrap monthly_trend as a LangChain @tool
# ============================================================================

def make_monthly_trend_tool() -> Any:
    """Build and return a ``@tool`` the data_analyst agent could use.

    Requirements:
    - Decorate a zero-argument function named exactly ``monthly_late_trend``
      with ``@tool`` (``from langchain_core.tools import tool``).
    - Inside: load the leakage-safe frame, call ``monthly_trend`` from
      ``src.ml_pipeline``, and return the result as a STRING (e.g.
      ``df.to_string(index=False)``) — tools hand the LLM text, never DataFrames.
    - Write a real docstring containing the word "trend": the docstring IS the
      API documentation the LLM reads when deciding whether to call your tool.
    - Return the decorated tool object itself.
    """
    raise NotImplementedError(
        "TASK 4: implement make_monthly_trend_tool() — define "
        "monthly_late_trend() -> str inside, decorate it with @tool, "
        "give it a docstring mentioning 'trend', and return the tool object."
    )


def _check_task4() -> None:
    from langchain_core.tools import BaseTool

    trend_tool = make_monthly_trend_tool()
    assert isinstance(trend_tool, BaseTool), (
        "not a tool — did you forget the @tool decorator, or call the function "
        "instead of returning the decorated object?"
    )
    assert trend_tool.name == "monthly_late_trend", f"name is {trend_tool.name!r}"
    assert trend_tool.description, "empty description — the docstring is the API"
    assert "trend" in trend_tool.description.lower(), "docstring must say what it does"
    out = trend_tool.invoke({})
    assert isinstance(out, str), "tools must return strings, not DataFrames"
    assert "late_rate" in out and "2024-01" in out, "output should be the trend table"


# ============================================================================
# TASK 5 (70 XP) — A 2-node StateGraph with a conditional keyword router
# ============================================================================

class TriageState(TypedDict):
    """State for the mini triage graph. Provided — build the graph around it."""

    question: str
    answer: str


def route_question(state: TriageState) -> str:
    """Return ``"model"`` for ML questions, ``"data"`` for everything else.

    Pure keyword routing — no LLM. Questions mentioning training, models,
    prediction, thresholds, recall/precision or tuning go to ``"model"``;
    anything else (late rates, trends, breakdowns) goes to ``"data"``.
    This is exactly what the Week-4 supervisor does, minus the LLM.
    """
    raise NotImplementedError(
        "TASK 5a: implement route_question(state) — lowercase the question, "
        "return 'model' if any ML keyword (train/model/predict/threshold/"
        "recall/tune/...) appears, else 'data'."
    )


def build_router_graph() -> Any:
    """Compile a StateGraph: START --router--> data_node | model_node --> END.

    Requirements:
    - ``StateGraph(TriageState)`` with two nodes, ``data_node`` and
      ``model_node``. Each returns ``{"answer": "..."}`` where the answer
      STARTS WITH ``[data_analyst]`` / ``[ml_engineer]`` respectively.
    - Conditional entry: ``add_conditional_edges(START, route_question,
      {"data": "data_node", "model": "model_node"})`` — the router returns a
      LABEL, the dict maps labels to node names.
    - Both nodes edge to END. Return the COMPILED graph.
    """
    raise NotImplementedError(
        "TASK 5b: implement build_router_graph() — "
        "from langgraph.graph import StateGraph, START, END; two nodes, "
        "conditional entry via route_question, compile() and return."
    )


def _check_task5() -> None:
    assert route_question({"question": "Train a model for me", "answer": ""}) == "model"
    assert route_question({"question": "Late rate by market?", "answer": ""}) == "data"
    app = build_router_graph()
    out1 = app.invoke({"question": "What is the late delivery rate by market?"})
    assert out1["answer"].startswith("[data_analyst]"), f"got: {out1['answer']!r}"
    out2 = app.invoke({"question": "Train a model and tune the recall threshold."})
    assert out2["answer"].startswith("[ml_engineer]"), f"got: {out2['answer']!r}"


# ============================================================================
# The Trial runner — do not edit below this line
# ============================================================================

TASKS: list[tuple[str, int, Any]] = [
    ("TASK 1 — leakage-safe loading", 40, _check_task1),
    ("TASK 2 — worst carrier (EDA)", 50, _check_task2),
    ("TASK 3 — train + threshold tuning", 80, _check_task3),
    ("TASK 4 — @tool wrapping", 60, _check_task4),
    ("TASK 5 — LangGraph keyword router", 70, _check_task5),
]


def main() -> int:
    print("=" * 72)
    print("WEEK 3 TRIAL — classical ML + LangGraph fundamentals (300 XP)")
    print("=" * 72)
    earned = 0
    for label, xp, check in TASKS:
        print(f"\n--- {label} [{xp} XP] ---")
        try:
            check()
        except NotImplementedError as exc:
            print(f"NOT IMPLEMENTED: {exc}")
            print("\nThe trial stops here. Implement the TODO above and re-run.")
            print(f"XP so far: {earned} / 300")
            return 1
        except AssertionError as exc:
            print(f"ASSERT FAILED: {exc}")
            print("\nThe trial stops here. Fix the implementation and re-run.")
            print(f"XP so far: {earned} / 300")
            return 1
        earned += xp
        print(f"PASS (+{xp} XP)")
    print("\n" + "=" * 72)
    print(f"TRIAL COMPLETE — {earned} / 300 XP. Week 3 skills confirmed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
