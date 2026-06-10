"""Central LLM + environment configuration for the Week 3&4 multi-agent project.

WHY THIS EXISTS
---------------
Every agent in this project (supervisor, data analyst, ML engineer, contracts RAG,
NL2SQL) needs a chat model. Hardcoding a provider in each module makes the training
material brittle — different consulting engagements use different providers. So we
centralize provider selection behind one function, ``get_llm()``, driven by env vars.

Supported providers (set ``LLM_PROVIDER`` in your ``.env``):
- ``azure_openai``  (default — matches the rest of the Zero-to-Hero curriculum)
- ``openai``
- ``anthropic``

OFFLINE MODE
------------
Classrooms don't always have API keys. ``is_offline()`` returns True when ``OFFLINE=1``
is set or when no provider key is present. Modules that can degrade gracefully
(PageIndex retrieval, EDA, SQL guardrails) check this and fall back to non-LLM logic;
modules that fundamentally need an LLM (the supervisor graph) raise a clear error.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = the week3&4 directory (this file lives in week3&4/src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42

# Load .env if python-dotenv is available; fall back to a tiny manual parser so the
# project has no hard dependency on it.
def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        return
    except ImportError:
        pass
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def is_offline() -> bool:
    """True when the project should avoid all LLM calls.

    Triggered by ``OFFLINE=1`` or by the absence of any provider API key.
    """
    if os.getenv("OFFLINE", "0") == "1":
        return True
    has_key = any(
        os.getenv(k)
        for k in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    return not has_key


def get_llm(temperature: float = 0.0, **kwargs):
    """Return a configured LangChain chat model based on ``LLM_PROVIDER``.

    Raises a descriptive RuntimeError in offline mode — callers that can degrade
    should check ``is_offline()`` *before* calling this.
    """
    if is_offline():
        raise RuntimeError(
            "No LLM available: OFFLINE=1 or no API key set. "
            "Copy .env.example to .env and set LLM_PROVIDER + the matching key, "
            "or use the offline fallbacks documented in ARCHITECTURE.md §5."
        )

    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=temperature,
            **kwargs,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
            **kwargs,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
            **kwargs,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Use one of: azure_openai, openai, anthropic."
    )
