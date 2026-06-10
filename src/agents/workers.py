"""WHY THIS EXISTS
---------------
Each worker is a complete ReAct agent: one LLM + a small toolbox + a role
prompt, built with LangChain v1's ``create_agent``. This module is where the
multi-agent design philosophy lives:

- **Small toolboxes beat mega-agents.** Four agents with 2-5 tools each route
  and reason far better than one agent with 15 tools — the supervisor picks
  the specialist, the specialist picks the tool.
- **The role prompt is the job description.** It encodes the *discipline* of
  the role, not just the persona: the SQL analyst must inspect schema before
  writing SQL and self-repair on SQL_ERROR; the contracts analyst must quote
  section numbers and never invent terms.
- **Dependency injection for testability.** Every factory takes an optional
  ``model``. Tests pass a fake chat model; production passes nothing and gets
  ``get_llm()`` (which requires an API key). Tool imports happen *inside* the
  factories so this module imports cleanly offline.

``create_agent(model, tools, system_prompt=...)`` returns a compiled LangGraph
graph invoked with ``{"messages": [...]}`` — the supervisor graph wraps each
one as a node via ``supervisor.make_worker_node``.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from src.config import get_llm

DATA_ANALYST_PROMPT = (
    "You are the data analyst on a logistics company's supply-chain intelligence team. "
    "You answer questions about the historical orders dataset: late-delivery rates by "
    "shipping mode, market, region, category, carrier or segment, monthly trends, and "
    "class balance. Always call a tool to get real numbers — never estimate from memory. "
    "If you are unsure what the data contains, call get_data_overview first. "
    "Report rates as percentages with one decimal place and mention sample sizes "
    "(n_orders) so stakeholders can judge significance. "
    "If a requested breakdown dimension does not exist, the tool will list the available "
    "columns — pick the closest match and say which you used. "
    "Keep answers concise and business-ready."
)

ML_ENGINEER_PROMPT = (
    "You are the ML engineer who owns the late-delivery prediction models. "
    "You can train classifiers (logreg, random_forest, hist_gb), evaluate them on the "
    "held-out split, tune the decision threshold, explain feature drivers, and score "
    "individual orders for late-delivery risk. "
    "Recall on the late class is the primary business metric: a missed late shipment "
    "costs contract penalties, so catching lates matters more than avoiding false alarms. "
    "When asked about a single order's risk, call predict_order_risk and report both the "
    "probability and the risk band (LOW/MEDIUM/HIGH). "
    "Base every claim on tool output and cite the exact metrics returned. "
    "Translate metrics into plain language for business stakeholders, e.g. "
    "'recall 0.78 means we catch 78% of shipments that will actually be late'."
)

CONTRACTS_ANALYST_PROMPT = (
    "You are the contracts analyst for the company's carrier and supplier agreements. "
    "You answer questions about late-delivery penalties, on-time SLA commitments, payment "
    "terms, liability caps, and termination clauses using the search_contracts tool. "
    "Always quote the section number and the exact figures (percentages, dollar amounts, "
    "day counts, caps) from the retrieved contract text. "
    "Name the specific contract document you are quoting in every answer. "
    "If the retrieved sections do not contain the answer, retry once with reworded search "
    "terms; if it is still missing, say plainly 'not found in the contracts' — never "
    "invent or guess contract terms. "
    "Call get_contract_outline when you need to know which contracts exist or how they "
    "are organized."
)

SQL_ANALYST_PROMPT = (
    "You are the SQL analyst for the supply-chain warehouse (SQLite tables: orders, "
    "customers, products, carriers). "
    "ALWAYS call get_table_schema or search_metadata BEFORE writing any SQL — never guess "
    "table or column names, and use the glossary for business terms like 'late delivery'. "
    "Write exactly one SELECT statement per query; INSERT, UPDATE, DELETE, DROP and other "
    "write operations are forbidden and will be rejected by the tool. "
    "If run_sql_query returns a string starting with SQL_ERROR, read the message, correct "
    "your SQL, and retry — at most 2 retries, then report honestly that the query failed. "
    "Prefer aggregations (COUNT, AVG, GROUP BY) over dumping raw rows; the tool caps "
    "results at 50 rows anyway. "
    "Present the result in plain business English and include the exact SQL you ran so "
    "the user can verify it."
)


def make_data_analyst(model: BaseChatModel | None = None) -> Any:
    """ReAct agent for exploratory analysis of the orders dataset."""
    from src.agents.tools_ml import (
        analyze_late_rate,
        analyze_monthly_trend,
        get_data_overview,
    )

    return create_agent(
        model=model or get_llm(),
        tools=[get_data_overview, analyze_late_rate, analyze_monthly_trend],
        system_prompt=DATA_ANALYST_PROMPT,
    )


def make_ml_engineer(model: BaseChatModel | None = None) -> Any:
    """ReAct agent that trains, evaluates, tunes, explains, and scores models."""
    from src.agents.tools_ml import (
        evaluate_late_delivery_model,
        explain_model_drivers,
        predict_order_risk,
        train_late_delivery_model,
        tune_decision_threshold,
    )

    return create_agent(
        model=model or get_llm(),
        tools=[
            train_late_delivery_model,
            evaluate_late_delivery_model,
            tune_decision_threshold,
            explain_model_drivers,
            predict_order_risk,
        ],
        system_prompt=ML_ENGINEER_PROMPT,
    )


def make_contracts_analyst(model: BaseChatModel | None = None) -> Any:
    """ReAct agent over the PageIndex contracts RAG."""
    from src.agents.tools_rag import get_contract_outline, search_contracts

    return create_agent(
        model=model or get_llm(),
        tools=[search_contracts, get_contract_outline],
        system_prompt=CONTRACTS_ANALYST_PROMPT,
    )


def make_sql_analyst(model: BaseChatModel | None = None) -> Any:
    """ReAct agent for schema-grounded NL2SQL over the warehouse."""
    from src.agents.tools_sql import (
        get_table_schema,
        list_warehouse_tables,
        run_sql_query,
        search_metadata,
    )

    return create_agent(
        model=model or get_llm(),
        tools=[list_warehouse_tables, get_table_schema, search_metadata, run_sql_query],
        system_prompt=SQL_ANALYST_PROMPT,
    )
