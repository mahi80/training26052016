"""WEEK 3 QUIZ — 10 MCQs on classical ML + LangGraph fundamentals.

WHY THIS EXISTS
---------------
The Trial (week3_starter.py) tests whether you can BUILD it; this quiz tests
whether you can EXPLAIN it — the skill you actually need in front of a client.
Topics: leakage, stratified splits, class imbalance, recall/precision economics,
threshold tuning, HistGradientBoosting vs LightGBM, @tool anatomy, StateGraph
state + reducers + conditional edges, and the ReAct loop.

HOW TO RUN (self-contained, stdlib only, no API key, no input() by default):

    python exercises/week3_quiz.py                       # print the questions
    python exercises/week3_quiz.py --answers ABCDABCDAB  # grade your 10 letters
    python exercises/week3_quiz.py --interactive         # answer one by one

The grader prints per-question explanations — read them even for questions you
got right; the WHY is the teaching payload. Pass mark: 7/10.
"""

from __future__ import annotations

import argparse
import sys

PASS_MARK = 7

QUESTIONS: list[dict] = [
    {
        "topic": "Data leakage",
        "q": "Your late-delivery model scores a suspicious 99.9% ROC-AUC. Which "
             "input column is the prime suspect, and why?",
        "options": {
            "A": "order_date — calendar information is never a legitimate feature",
            "B": "actual_shipping_days — only known after delivery, and the target is computed from it",
            "C": "carrier_name — high-cardinality categoricals always overfit",
            "D": "sales — monetary columns leak customer intent",
        },
        "answer": "B",
        "explain": "late_delivery = actual_shipping_days > scheduled_shipping_days, "
                   "so the column literally contains the answer — and it does not "
                   "exist at prediction time. Dates, carriers and sales are all "
                   "legitimate pre-shipment facts.",
    },
    {
        "topic": "Stratified split",
        "q": "Why does split_data stratify the train/test split on late_delivery?",
        "options": {
            "A": "It makes training faster by balancing CPU load across folds",
            "B": "It removes leakage columns automatically",
            "C": "It oversamples the minority class so the model sees more late orders",
            "D": "It keeps the ~30% late rate identical in train and test, so metrics aren't skewed by an unlucky split",
        },
        "answer": "D",
        "explain": "Stratification preserves class proportions in both splits — "
                   "nothing more. It does NOT resample (that's oversampling, option "
                   "C describes SMOTE-style techniques) and has nothing to do with "
                   "leakage or speed.",
    },
    {
        "topic": "Class imbalance",
        "q": "With a 30% late rate, a 'model' that always predicts ON TIME gets "
             "which scores — and what is the lesson?",
        "options": {
            "A": "70% accuracy, 0% recall on lates — accuracy is misleading on imbalanced targets",
            "B": "30% accuracy, 100% recall — always-on-time is a strong baseline",
            "C": "70% accuracy, 70% recall — accuracy and recall agree",
            "D": "50% accuracy, 50% recall — chance level",
        },
        "answer": "A",
        "explain": "It is right on the 70% majority class (70% accuracy) while "
                   "catching zero late shipments (0% recall). That single example "
                   "is why the whole pipeline reports recall/precision/PR-AUC and "
                   "never mentions accuracy.",
    },
    {
        "topic": "Metric choice",
        "q": "A missed late shipment costs contract penalties; a false alarm costs "
             "an analyst a phone call. Which evaluation policy fits?",
        "options": {
            "A": "Maximize accuracy — it balances both error types",
            "B": "Maximize precision — alarms must never be wrong",
            "C": "Maximize recall subject to a precision floor — catch lates, keep alarms mostly real",
            "D": "Maximize ROC-AUC and ship with the default 0.5 threshold",
        },
        "answer": "C",
        "explain": "Asymmetric costs mean you optimize for the expensive error "
                   "(missed lates -> recall) while constraining the cheap one "
                   "(precision floor). That is exactly what tune_threshold("
                   "min_precision=...) implements.",
    },
    {
        "topic": "Threshold tuning",
        "q": "You lower the decision threshold from 0.5 to 0.35 on an already-"
             "trained classifier. What happens?",
        "options": {
            "A": "Recall rises, precision typically falls, ROC-AUC is unchanged — no retraining involved",
            "B": "Recall and precision both rise",
            "C": "ROC-AUC rises because more positives are predicted",
            "D": "Nothing, until you retrain the model with the new threshold",
        },
        "answer": "A",
        "explain": "The threshold only moves where you cut predict_proba: more "
                   "orders flagged late -> more true lates caught (recall up) but "
                   "more false alarms (precision down). ROC-AUC is threshold-free, "
                   "and no retraining happens — that is why tuning is 'free'.",
    },
    {
        "topic": "HistGB vs LightGBM",
        "q": "What is sklearn's HistGradientBoostingClassifier, relative to LightGBM?",
        "options": {
            "A": "A thin wrapper that imports lightgbm under the hood",
            "B": "sklearn's native take on the same idea — histogram-binned gradient-boosted trees, zero extra dependencies",
            "C": "A bagging ensemble like RandomForest, just renamed",
            "D": "A neural-network approximation of gradient boosting",
        },
        "answer": "B",
        "explain": "Both bucket continuous features into ~255 histogram bins so "
                   "split-finding costs O(bins) instead of O(rows) — LightGBM's "
                   "core trick. hist_gb gives you that speed natively in sklearn, "
                   "which is why it is our default tabular model.",
    },
    {
        "topic": "@tool anatomy",
        "q": "When a ReAct agent decides whether (and how) to call your @tool, "
             "what does the LLM actually read?",
        "options": {
            "A": "The function's source code, line by line",
            "B": "Your unit tests for the tool",
            "C": "The tool's name, docstring, and argument schema",
            "D": "Only the return type annotation",
        },
        "answer": "C",
        "explain": "@tool serializes name + docstring + a JSON schema of the "
                   "arguments into the prompt; the body is invisible to the model. "
                   "That is why our tool docstrings are written FOR the LLM — they "
                   "are the API documentation it routes on.",
    },
    {
        "topic": "State reducers",
        "q": "In SupplyChainState, messages is Annotated[list, add_messages]. "
             "What does that reducer change?",
        "options": {
            "A": "Nodes returning {'messages': [msg]} APPEND to the list instead of overwriting it",
            "B": "Messages are deduplicated by content hash",
            "C": "Messages are sorted by timestamp after every node",
            "D": "The list becomes immutable",
        },
        "answer": "A",
        "explain": "Without a reducer, a node's return value REPLACES the old "
                   "value. add_messages merges instead, so every agent sees the "
                   "full conversation history — the mechanism that lets four "
                   "workers share one context.",
    },
    {
        "topic": "Conditional edges",
        "q": "After our supervisor node runs, what determines which node executes next?",
        "options": {
            "A": "Node insertion order in the StateGraph",
            "B": "LangGraph picks the least-recently-used worker",
            "C": "Every worker runs in parallel and the supervisor merges results",
            "D": "The conditional-edge routing function reads state (e.g. next_agent) and returns the target's label",
        },
        "answer": "D",
        "explain": "add_conditional_edges(source, router_fn, mapping): after the "
                   "source node, LangGraph calls router_fn(state) and follows the "
                   "label it returns through the mapping. The supervisor writes "
                   "next_agent into state; the edge function reads it.",
    },
    {
        "topic": "ReAct loop",
        "q": "What terminates a ReAct agent's reason-act-observe loop in normal operation?",
        "options": {
            "A": "A fixed three-iteration limit baked into create_agent",
            "B": "The model responds with a final answer containing no tool call",
            "C": "The first tool that returns an empty string",
            "D": "The supervisor interrupts it after a timeout",
        },
        "answer": "B",
        "explain": "The loop continues while the model keeps emitting tool calls; "
                   "a plain assistant message ends it. Recursion limits exist as a "
                   "safety net against runaway loops, not as the normal stop "
                   "condition.",
    },
]


def print_questions() -> None:
    print(f"WEEK 3 QUIZ — {len(QUESTIONS)} questions. No peeking at the labs.")
    for i, item in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}. [{item['topic']}] {item['q']}")
        for letter in "ABCD":
            print(f"   {letter}) {item['options'][letter]}")
    print(
        "\nSubmit: python exercises/week3_quiz.py --answers "
        + "X" * len(QUESTIONS)
        + "   (your 10 letters A-D, in order)"
    )


def _clean(answers: str) -> str:
    return "".join(ch for ch in answers.upper() if ch not in " ,;-_").strip()


def grade(answers: str) -> int:
    cleaned = _clean(answers)
    if len(cleaned) != len(QUESTIONS) or any(c not in "ABCD" for c in cleaned):
        print(
            f"Invalid answer string {answers!r}: need exactly {len(QUESTIONS)} "
            "letters A-D, e.g. --answers ABCDABCDAB"
        )
        raise SystemExit(2)

    score = 0
    for i, (item, given) in enumerate(zip(QUESTIONS, cleaned), 1):
        correct = item["answer"]
        if given == correct:
            score += 1
            verdict = "CORRECT"
        else:
            verdict = f"WRONG (you said {given}, answer is {correct})"
        print(f"\nQ{i} [{item['topic']}] -> {verdict}")
        print(f"   {correct}) {item['options'][correct]}")
        print(f"   WHY: {item['explain']}")

    pct = 100 * score // len(QUESTIONS)
    print("\n" + "=" * 72)
    print(f"SCORE: {score}/{len(QUESTIONS)} ({pct}%) — "
          + ("PASS" if score >= PASS_MARK else f"below the {PASS_MARK}/10 pass mark")
          )
    print("=" * 72)
    return score


def interactive() -> int:
    """Prompt for one answer per question, then grade. Opt-in via --interactive."""
    collected: list[str] = []
    for i, item in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}. [{item['topic']}] {item['q']}")
        for letter in "ABCD":
            print(f"   {letter}) {item['options'][letter]}")
        while True:
            given = input("Your answer (A-D): ").strip().upper()
            if given in ("A", "B", "C", "D"):
                collected.append(given)
                break
            print("Please enter A, B, C, or D.")
    return grade("".join(collected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 3 quiz (10 MCQs)")
    parser.add_argument("--answers", help="10-letter answer string, e.g. ABCDABCDAB")
    parser.add_argument(
        "--interactive", action="store_true", help="answer question by question"
    )
    args = parser.parse_args(argv)

    if args.interactive:
        interactive()
    elif args.answers:
        grade(args.answers)
    else:
        print_questions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
