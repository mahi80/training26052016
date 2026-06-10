"""WEEK 3 TRIAL — TRAINER SOLUTIONS. Do not distribute to trainees.

WHY THIS EXISTS
---------------
Reference implementations for every task in ``week3_starter.py``, with TRAINER
NOTES documenting the mistakes trainees actually make, so you can spot them fast
during review. The checks are byte-for-byte the same asserts as the starter.

Run it exactly like the starter (must print 300/300 and exit 0, fully offline):

    python exercises/week3_solutions_TRAINER_ONLY.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# TASK 1 (40 XP) — Load a leakage-safe modelling frame
# ============================================================================
# TRAINER NOTES — common trainee mistakes:
# * pd.read_csv directly: passes the shape assert but fails the datetime assert
#   (order_date stays a string) — good teaching moment about reusing the loader.
# * df.drop(columns=["late_delivery"]) — dropping the TARGET instead of the
#   leakage columns. The check has a dedicated message for this one.
# * Dropping leakage manually AFTER load_orders() — works, but point out the
#   loader's drop_leakage flag exists precisely so nobody forgets a column.

def load_clean_frame() -> pd.DataFrame:
    """Canonical orders frame with post-delivery (leakage) columns removed."""
    from src.ml_pipeline import load_orders

    return load_orders(drop_leakage=True)


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
# TRAINER NOTES:
# * late_rate_by sorts worst-first already; trainees who re-sort ascending and
#   take .iloc[0] return NordHaul (the BEST carrier) — the assert calls it out.
# * Hand-rolled groupby returning numpy types: the (str, float) isinstance
#   check fails on np.float64 — teach float(...) casting at API boundaries.
#   (np.float64 subclasses float since NumPy 2, but np.str_ is not str.)

def worst_carrier(df: pd.DataFrame) -> tuple[str, float]:
    """(carrier_name, late_rate) of the worst carrier, via the analytics layer."""
    from src.ml_pipeline import late_rate_by

    table = late_rate_by(df, "carrier_name")  # already sorted worst-first
    top = table.iloc[0]
    return str(top["carrier_name"]), float(top["late_rate"])


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
# TRAINER NOTES:
# * Calling tune_threshold BEFORE train_model -> FileNotFoundError (no artifact).
#   The order matters because evaluate/tune reload the persisted joblib.
# * Confusing min_precision (the floor constraint) with threshold (the cut on
#   predict_proba) — ask trainees to explain the difference out loud.
# * Reporting cv_recall from train_model instead of held-out recall at 0.4 —
#   cv_recall (~0.68) happens to pass the 0.60 floor here, so probe verbally:
#   "which split is that number computed on?"
# Expected solution numbers (seed 42): recall@0.4 ~ 0.78, tuned threshold 0.42
# with recall ~0.75 at precision ~0.51.

def train_and_tune() -> dict[str, Any]:
    """Train logreg, evaluate held-out at 0.4, tune threshold for recall."""
    from src.ml_pipeline import evaluate_model, train_model, tune_threshold

    train_model("logreg")  # persists models/logreg.joblib + sealed test split
    at_040 = evaluate_model("logreg", threshold=0.4)
    tuned = tune_threshold("logreg", min_precision=0.5)
    return {"recall_at_040": at_040["recall"], "tuned": tuned}


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
# TRAINER NOTES:
# * Returning monthly_late_trend() (CALLED) instead of the object — the
#   isinstance(BaseTool) assert catches it. @tool turns the function into a
#   StructuredTool; you invoke it with .invoke({}), not with parentheses.
# * Returning the DataFrame itself — "tools must return strings" assert fires.
#   Reinforce: the LLM consumes text; a repr'd DataFrame object is garbage.
# * One-word docstrings — technically passes "trend" if they write "trend",
#   but review for quality: the docstring is the ONLY thing the routing LLM
#   reads. We use to_string(index=False), not to_markdown (needs `tabulate`).

def make_monthly_trend_tool() -> Any:
    """Return monthly_trend wrapped as a LangChain tool for the data_analyst."""
    from langchain_core.tools import tool

    from src.ml_pipeline import load_orders, monthly_trend

    @tool
    def monthly_late_trend() -> str:
        """Monthly trend of order volume and late-delivery rate.

        Use this to answer "is lateness getting better or worse?" or any
        question about seasonality (look for the Q4 spike). Returns a text
        table with columns: month (YYYY-MM), n_orders, late_rate.
        """
        df = load_orders(drop_leakage=True)
        return monthly_trend(df).to_string(index=False)

    return monthly_late_trend


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
# TRAINER NOTES:
# * The router returns a LABEL ("data"/"model"); the dict in
#   add_conditional_edges maps labels -> node names. Trainees who return node
#   names directly AND pass the mapping dict get a KeyError — both work alone,
#   mixing them doesn't. This is the #1 confusion of the week.
# * Returning the whole mutated state from nodes instead of a partial update
#   {"answer": ...} — works here, but breaks once reducers (add_messages) are
#   involved in Week 4. Insist on partial updates now.
# * Forgetting .compile() — "StateGraph object has no attribute invoke".

class TriageState(TypedDict):
    """State for the mini triage graph. Provided — build the graph around it."""

    question: str
    answer: str


_MODEL_KEYWORDS: tuple[str, ...] = (
    "train", "model", "predict", "threshold", "recall", "precision",
    "tune", "evaluate", "classifier", "auc",
)


def route_question(state: TriageState) -> str:
    """'model' for ML questions, 'data' for everything else. Pure keywords."""
    question = state["question"].lower()
    return "model" if any(k in question for k in _MODEL_KEYWORDS) else "data"


def build_router_graph() -> Any:
    """START --route_question--> data_node | model_node --> END (compiled)."""
    from langgraph.graph import END, START, StateGraph

    def data_node(state: TriageState) -> dict:
        return {
            "answer": f"[data_analyst] I would answer {state['question']!r} "
            "with late_rate_by / monthly_trend on the orders frame."
        }

    def model_node(state: TriageState) -> dict:
        return {
            "answer": f"[ml_engineer] I would answer {state['question']!r} "
            "with train_model / evaluate_model / tune_threshold."
        }

    graph = StateGraph(TriageState)
    graph.add_node("data_node", data_node)
    graph.add_node("model_node", model_node)
    graph.add_conditional_edges(
        START, route_question, {"data": "data_node", "model": "model_node"}
    )
    graph.add_edge("data_node", END)
    graph.add_edge("model_node", END)
    return graph.compile()


def _check_task5() -> None:
    assert route_question({"question": "Train a model for me", "answer": ""}) == "model"
    assert route_question({"question": "Late rate by market?", "answer": ""}) == "data"
    app = build_router_graph()
    out1 = app.invoke({"question": "What is the late delivery rate by market?"})
    assert out1["answer"].startswith("[data_analyst]"), f"got: {out1['answer']!r}"
    out2 = app.invoke({"question": "Train a model and tune the recall threshold."})
    assert out2["answer"].startswith("[ml_engineer]"), f"got: {out2['answer']!r}"


# ============================================================================
# The Trial runner — identical to the starter's
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
    print("WEEK 3 TRIAL (TRAINER SOLUTIONS) — must score 300/300 offline")
    print("=" * 72)
    earned = 0
    for label, xp, check in TASKS:
        print(f"\n--- {label} [{xp} XP] ---")
        try:
            check()
        except (NotImplementedError, AssertionError) as exc:
            print(f"FAILED: {exc}")
            print(f"XP so far: {earned} / 300")
            return 1
        earned += xp
        print(f"PASS (+{xp} XP)")
    print("\n" + "=" * 72)
    print(f"TRIAL COMPLETE — {earned} / 300 XP.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
