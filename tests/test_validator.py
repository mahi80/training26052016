"""WHY THIS EXISTS
---------------
Proof the validator/judge node rules correctly offline. The heuristic is
tested directly (it is deterministic by design); the LLM-as-judge path is
tested with a fake structured-output model, mirroring FakeRouterLLM in
test_graph_routing.py — the real node code runs, only the model is scripted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # conftest does this too; keep standalone-safe
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.validator import Verdict, heuristic_verdict, make_validator_node


class _ScriptedVerdictRunnable:
    """Stand-in for ``llm.with_structured_output(Verdict)`` — replays a script."""

    def __init__(self, script: list[Verdict]) -> None:
        self._script = list(script)
        self._calls = 0

    def invoke(self, _messages: Any, config: Any = None) -> Verdict:
        ruling = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return ruling


class FakeVerdictLLM:
    """Fake chat model exposing only what the validator node actually uses."""

    def __init__(self, script: list[Verdict]) -> None:
        self._script = script

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _ScriptedVerdictRunnable:
        assert schema is Verdict, "validator must request the Verdict schema"
        return _ScriptedVerdictRunnable(self._script)


def _state(question: str, answer: str | None) -> dict:
    messages: list = [HumanMessage(content=question)]
    if answer is not None:
        messages.append(AIMessage(content=answer, name="sql_analyst"))
    return {"messages": messages, "next_agent": "FINISH", "verdict": "", "verdict_reason": ""}


# ---- Heuristic judge (deterministic) -----------------------------------------

def test_heuristic_passes_substantive_answer() -> None:
    ruling = heuristic_verdict(
        "Which carrier is worst?",
        "SwiftShip Express has the highest late rate at 34.2% across 2,841 orders, "
        "well above the fleet average of 29.8%.",
    )
    assert ruling.verdict == "complete"


def test_heuristic_flags_empty_answer() -> None:
    assert heuristic_verdict("Anything?", "").verdict == "needs_human"


def test_heuristic_flags_too_short_answer() -> None:
    ruling = heuristic_verdict("Which carrier is worst?", "SwiftShip.")
    assert ruling.verdict == "needs_human"
    assert "short" in ruling.reason.lower()


def test_heuristic_flags_leaked_tool_errors() -> None:
    for marker in ("SQL_ERROR", "MCP_ERROR", "A2A_ERROR"):
        answer = f"I tried to check but got {marker}: something broke underneath here."
        ruling = heuristic_verdict("Status?", answer)
        assert ruling.verdict == "needs_human", marker
        assert marker in ruling.reason


# ---- Node factory: offline default degrades to the heuristic ------------------

def test_offline_default_uses_heuristic_and_never_raises(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE", "1")
    node = make_validator_node()  # no llm, offline -> heuristic, no RuntimeError
    out = node(_state("Q?", "A long, grounded answer with figures: 34.2% late over 2,841 orders."))
    assert out["verdict"] == "complete"
    assert out["verdict_reason"]


# ---- Node factory: LLM-as-judge with a scripted fake ---------------------------

def test_llm_judge_writes_scripted_verdict_to_state() -> None:
    fake = FakeVerdictLLM([Verdict(verdict="needs_human", reason="Numbers lack a source.")])
    node = make_validator_node(llm=fake)
    out = node(_state("Q?", "Some long draft answer that the scripted judge will reject anyway."))
    assert out == {"verdict": "needs_human", "verdict_reason": "Numbers lack a source."}


def test_llm_judge_complete_verdict() -> None:
    fake = FakeVerdictLLM([Verdict(verdict="complete", reason="Grounded and on-topic.")])
    node = make_validator_node(llm=fake)
    out = node(_state("Q?", "A perfectly grounded answer with a 34.2% figure and a source."))
    assert out["verdict"] == "complete"
