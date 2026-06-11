"""WHY THIS EXISTS
---------------
The **validator / judge** node from the architecture diagram: after the
supervisor says FINISH, one more gate inspects the draft answer *before* the
user sees it and rules:

- ``complete``    → auto-complete, the graph ends, the answer ships;
- ``needs_human`` → the graph pauses at the human-review node (interrupt).

Two judges, same contract — and the choice between them is the lesson:

1. **LLM-as-judge** (online): a structured-output call that grades the answer
   against the question. Smart, but it costs a call and can itself be wrong.
2. **Heuristic** (offline fallback): deterministic red flags — empty/too-short
   answers, or tool-failure markers (``SQL_ERROR`` / ``MCP_ERROR`` /
   ``A2A_ERROR``) leaking into the final text. Dumb, but explainable and free.

Production systems run BOTH: cheap deterministic checks first, model judges
on what passes. Like every node factory here, ``make_validator_node`` takes
an optional ``llm`` so tests inject a fake and run the real node offline.
"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_llm, is_offline
from src.state_v2 import SupplyChainStateV2

# Tool-failure markers that must never reach an end user unexplained.
_ERROR_MARKERS = ("SQL_ERROR", "MCP_ERROR", "A2A_ERROR")
_MIN_ANSWER_CHARS = 40

_JUDGE_PROMPT = (
    "You are the quality gate of a supply-chain analytics team. You see a "
    "user's question and the team's draft answer. Rule 'complete' when the "
    "answer actually addresses the question with concrete, grounded content. "
    "Rule 'needs_human' when the answer is empty, evasive, off-topic, exposes "
    "raw tool errors, or makes claims with no supporting figures. Be strict: "
    "when in doubt, send it to a human."
)


class Verdict(BaseModel):
    """The judge's ruling — the structured-output schema."""

    verdict: Literal["complete", "needs_human"] = Field(
        description="'complete' if the draft answer fully and credibly answers "
        "the question; 'needs_human' if a person should review it first."
    )
    reason: str = Field(description="One short sentence justifying the ruling.")


def heuristic_verdict(question: str, answer: str) -> Verdict:
    """Deterministic offline judge: cheap red-flag checks, fully explainable."""
    text = (answer or "").strip()
    if not text:
        return Verdict(verdict="needs_human", reason="The team produced no answer at all.")
    if len(text) < _MIN_ANSWER_CHARS:
        return Verdict(
            verdict="needs_human",
            reason=f"Answer is only {len(text)} characters — too short to trust.",
        )
    for marker in _ERROR_MARKERS:
        if marker in text:
            return Verdict(
                verdict="needs_human",
                reason=f"Answer contains a raw tool failure ({marker}) — a person should check it.",
            )
    return Verdict(verdict="complete", reason="Answer is substantive and contains no tool-failure markers.")


def _last_answer(state: SupplyChainStateV2) -> tuple[str, str]:
    """Extract (question, draft answer) from the conversation state."""
    messages = state.get("messages", [])
    question = str(messages[0].content) if messages else ""
    answers = [m for m in messages if isinstance(m, AIMessage)]
    answer = str(answers[-1].content) if answers else ""
    return question, answer


def make_validator_node(
    llm: BaseChatModel | None = None,
) -> Callable[[SupplyChainStateV2], dict]:
    """Build the validator node (dependency-injectable for tests).

    With an ``llm`` (or online with a key): LLM-as-judge via structured
    output. Offline with no injected llm: the heuristic — the node *degrades*
    rather than raising, mirroring the PageIndex fallback pattern, because a
    quality gate that crashes is worse than a dumb one.
    """
    if llm is None and is_offline():
        def heuristic_node(state: SupplyChainStateV2) -> dict:
            question, answer = _last_answer(state)
            ruling = heuristic_verdict(question, answer)
            return {"verdict": ruling.verdict, "verdict_reason": ruling.reason}

        return heuristic_node

    if llm is None:
        llm = get_llm()
    judge = llm.with_structured_output(Verdict)

    def judge_node(state: SupplyChainStateV2) -> dict:
        question, answer = _last_answer(state)
        prompt = [
            SystemMessage(content=_JUDGE_PROMPT),
            AIMessage(content=f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{answer}"),
        ]
        ruling: Verdict = judge.invoke(prompt)
        return {"verdict": ruling.verdict, "verdict_reason": ruling.reason}

    return judge_node
