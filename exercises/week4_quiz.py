"""WEEK 4 QUIZ — 10 MCQs on RAG, NL2SQL, supervision, and production concerns.

WHY THIS EXISTS
---------------
The Trial (week4_starter.py) tests whether you can BUILD the multi-agent system;
this quiz tests whether you can DEFEND its design decisions in a client review.
Topics: vectorless (PageIndex) vs embedding RAG, tree navigation, hallucination
guards, metadata catalogs / OpenMetadata, NL2SQL guardrails and self-repair,
supervisor routing + FINISH, checkpointing / human-in-the-loop, observability.

HOW TO RUN (self-contained, stdlib only, no API key, no input() by default):

    python exercises/week4_quiz.py                       # print the questions
    python exercises/week4_quiz.py --answers ABCDABCDAB  # grade your 10 letters
    python exercises/week4_quiz.py --interactive         # answer one by one

The grader prints per-question explanations — the WHY is the teaching payload.
Pass mark: 7/10.
"""

from __future__ import annotations

import argparse
import sys

PASS_MARK = 7

QUESTIONS: list[dict] = [
    {
        "topic": "Vectorless RAG",
        "q": "What fundamentally distinguishes PageIndex-style retrieval from "
             "classic chunk-and-embed RAG?",
        "options": {
            "A": "It uses a bigger embedding model for higher accuracy",
            "B": "It stores the chunks in SQLite instead of a vector database",
            "C": "An LLM reasons over a hierarchical tree index (like a ToC) to navigate to sections — no embeddings or vector DB at all",
            "D": "It only works on PDF documents",
        },
        "answer": "C",
        "explain": "PageIndex keeps the document's own structure as a tree and "
                   "lets the model navigate it the way a human expert uses a "
                   "table of contents. No chunking heuristics, no embedding "
                   "model, no vector store to operate.",
    },
    {
        "topic": "Tree navigation",
        "q": "In the contracts index, every node carries a short summary. What is it for?",
        "options": {
            "A": "It is the 'scent' the navigating LLM reads in the outline to decide which branch to drill into",
            "B": "It is shown to end users instead of the contract text",
            "C": "It is embedded into vectors for similarity search",
            "D": "It pads the context window to a fixed size",
        },
        "answer": "A",
        "explain": "At query time the retriever shows the model the outline — "
                   "node ids, titles, summaries — and the model selects node ids. "
                   "Summaries make that outline informative enough to navigate; "
                   "offline they degrade to heading + first 240 characters.",
    },
    {
        "topic": "RAG tradeoffs",
        "q": "When would you prefer classic embedding RAG OVER a PageIndex-style tree?",
        "options": {
            "A": "Never — vectorless retrieval dominates in every scenario",
            "B": "When documents are highly structured, like contracts with numbered sections",
            "C": "When you need an explainable retrieval trace for auditors",
            "D": "For huge corpora of short, unstructured snippets with no reliable structure, where cheap similarity search shines",
        },
        "answer": "D",
        "explain": "Tree navigation needs structure worth navigating and spends "
                   "LLM tokens per query. Millions of unstructured tickets/chat "
                   "snippets favor embeddings. Options B and C describe exactly "
                   "where PageIndex is strongest — structured docs, auditable hops.",
    },
    {
        "topic": "Hallucination guards",
        "q": "Which practice best guards a contracts-RAG agent against hallucinated "
             "penalty numbers?",
        "options": {
            "A": "Raise the temperature so the model explores more phrasings",
            "B": "Require answers to quote the retrieved section text and cite doc + section path, and say 'not found' when retrieval returns nothing relevant",
            "C": "Let the model answer from its pretraining when retrieval is slow",
            "D": "Cap every answer at 50 words",
        },
        "answer": "B",
        "explain": "Grounding: the agent may only assert what the retrieved text "
                   "supports, with a citation (our node ids read like citations: "
                   "swiftship_express_msa/6/6.1) — and refusal is the correct "
                   "output when evidence is missing. A and C invite fabrication.",
    },
    {
        "topic": "Metadata catalog",
        "q": "Why does the sql_analyst consult the metadata catalog before writing "
             "SQL — and what is OpenMetadata's role in production?",
        "options": {
            "A": "The catalog caches query results so repeated SQL runs faster",
            "B": "It grounds the LLM in the authoritative schema + business glossary so it stops guessing column names; in production the same entities live in an OpenMetadata server",
            "C": "It stores the LLM provider's API keys securely",
            "D": "It replaces the database — queries execute against the catalog JSON",
        },
        "answer": "B",
        "explain": "Most NL2SQL failures are schema hallucinations (wrong column/"
                   "join). Descriptions, tags, sample queries and the glossary "
                   "('Late Delivery' -> orders.late_delivery) anchor generation. "
                   "Our JSON mirrors OpenMetadata's Table/Glossary entities, so "
                   "swapping in a real server keeps the same interface.",
    },
    {
        "topic": "Self-repair",
        "q": "run_sql_query returns 'SQL_ERROR: ...' strings instead of raising "
             "exceptions. Why?",
        "options": {
            "A": "An exception kills the agent loop; an error string becomes an observation the ReAct agent reads and self-repairs from (re-check schema, retry)",
            "B": "Python exceptions are slower than returning strings",
            "C": "SQLite is incapable of raising exceptions",
            "D": "So that errors stay hidden from the end user",
        },
        "answer": "A",
        "explain": "Tools never raise — that is a design rule. 'SQL_ERROR: no such "
                   "column carrier' flows back as a tool observation, the model "
                   "re-reads the schema and retries (max 2 repairs in the worker "
                   "prompt). Crashing the loop teaches the model nothing.",
    },
    {
        "topic": "NL2SQL guardrails",
        "q": "Which of these is NOT one of run_sql_query's guardrails?",
        "options": {
            "A": "Read-only connection (SQLite URI mode=ro)",
            "B": "Single statement only, first keyword SELECT/WITH, write keywords blocked",
            "C": "LIMIT 50 auto-appended when the query has no LIMIT",
            "D": "Automatic correction of misspelled column names before execution",
        },
        "answer": "D",
        "explain": "Wrong column names come back as SQL_ERROR strings for the "
                   "AGENT to fix — the tool never rewrites your SQL beyond "
                   "appending a LIMIT. A, B and C are real layers: treat "
                   "LLM-generated SQL as untrusted input, defense in depth.",
    },
    {
        "topic": "Supervisor routing",
        "q": "How does the supervisor reliably emit a valid route, and what does "
             "FINISH mean?",
        "options": {
            "A": "It parses the LLM's free text with regex and defaults to data_analyst",
            "B": "Workers vote on who should go next; FINISH triggers on a tie",
            "C": "llm.with_structured_output(Router) with next: Literal[workers + FINISH] constrains output to valid options; FINISH ends the turn and returns the answer",
            "D": "It calls every worker in sequence and FINISH selects the longest answer",
        },
        "answer": "C",
        "explain": "Structured output validates against the Router schema, so the "
                   "route is always a legal node name or FINISH — no string "
                   "parsing, no typos. FINISH maps to END at the conditional "
                   "edge: the supervisor decides the question is answered.",
    },
    {
        "topic": "Checkpointing & HITL",
        "q": "What does compiling the graph with a checkpointer enable?",
        "options": {
            "A": "State persists per thread_id, so conversations resume across calls and execution can pause at interrupts for human approval",
            "B": "Automatic GPU acceleration of every node",
            "C": "It makes LLM outputs deterministic",
            "D": "Tools no longer need guardrails because a human reviews everything",
        },
        "answer": "A",
        "explain": "A checkpointer snapshots state after every super-step keyed by "
                   "thread_id — the foundation for memory AND human-in-the-loop "
                   "(interrupt before a sensitive node, resume after approval). "
                   "D is the trap: HITL complements guardrails, never replaces them.",
    },
    {
        "topic": "Observability",
        "q": "For a production multi-agent system, what is the most useful thing "
             "to capture per request?",
        "options": {
            "A": "Only the final answer text",
            "B": "CPU and RAM utilization of the database server",
            "C": "The full trace: each supervisor decision, worker hop, tool call with inputs/outputs, plus tokens and latency (e.g. LangSmith)",
            "D": "A screenshot of the user's terminal",
        },
        "answer": "C",
        "explain": "When the answer is wrong you need to see WHERE it went wrong: "
                   "bad route? bad SQL? bad retrieval? Hop-by-hop traces (LangSmith, "
                   "or structured logs of supervisor -> worker -> tool events) make "
                   "agent debugging tractable; final answers alone tell you nothing.",
    },
]


def print_questions() -> None:
    print(f"WEEK 4 QUIZ — {len(QUESTIONS)} questions. Defend the design.")
    for i, item in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}. [{item['topic']}] {item['q']}")
        for letter in "ABCD":
            print(f"   {letter}) {item['options'][letter]}")
    print(
        "\nSubmit: python exercises/week4_quiz.py --answers "
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
    parser = argparse.ArgumentParser(description="Week 4 quiz (10 MCQs)")
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
