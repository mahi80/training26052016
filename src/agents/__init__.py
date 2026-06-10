"""WHY THIS EXISTS
---------------
This package is the *agent layer* of the project: it adapts every backing
capability (the sklearn ML pipeline, the PageIndex contracts RAG, the SQLite
warehouse + metadata catalog) into things a LangGraph multi-agent system can
use. It has three kinds of modules, and the separation is the lesson:

- ``tools_ml.py`` / ``tools_rag.py`` / ``tools_sql.py`` — **tools**: plain
  Python functions decorated with ``@tool`` that return *strings* (markdown
  tables, JSON). Tools are the only way an LLM touches real systems, so they
  must be deterministic, defensive, and self-describing (the docstring IS the
  interface the LLM reads).
- ``workers.py`` — **worker agents**: each one is a LangChain v1
  ``create_agent`` ReAct loop = one LLM + a small, focused toolbox + a role
  prompt. Small toolboxes route better than one mega-agent with 15 tools.
- ``supervisor.py`` — the **router**: a structured-output LLM call that picks
  which worker acts next (or FINISH). It holds no tools itself.

``src/graph.py`` wires these into the supervisor StateGraph. Nothing in this
package is imported at package-import time on purpose: tool modules pull heavy
or sibling dependencies lazily, so ``import src.agents`` always succeeds — even
offline, even while sibling modules are still being built.
"""

from __future__ import annotations

# Intentionally no re-exports: import the concrete module you need, e.g.
#   from src.agents.workers import make_sql_analyst
#   from src.agents.supervisor import make_supervisor_node, make_worker_node
# Re-exporting here would import every tool module (and its data dependencies)
# the moment anything touches the package — bad for offline mode and testability.
__all__: list[str] = [
    "supervisor",
    "tools_ml",
    "tools_rag",
    "tools_sql",
    "workers",
]
