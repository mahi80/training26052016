"""LAB 08 — NL2SQL Done Right: Metadata-Grounded, Guard-Railed, Self-Repairing.

WHY THIS EXISTS
---------------
"Just let the LLM write SQL" is how demos die in production. Three things are
missing from the naive approach, and this lab builds all three:

1. A METADATA CATALOG — the model cannot guess your column names, your
   geographic hierarchy, or what 'late' means in THIS business. We feed it an
   OpenMetadata-style catalog (schemas + descriptions + a business glossary)
   and demand schema lookup BEFORE SQL.
2. GUARDRAILS — the database must survive a confused (or prompt-injected)
   agent: SELECT-only, single statement, auto-LIMIT, and errors returned as
   strings the agent can read.
3. SELF-REPAIR — first drafts fail. The ``SQL_ERROR:`` string contract turns
   a crash into a feedback loop: the agent reads the error, re-checks the
   schema, and fixes its own query.

OBJECTIVES
----------
1. Tour data/metadata_catalog.json raw, then through ``MetadataCatalog``
   (get_table_schema, search, glossary).
2. Exercise every guardrail in ``run_sql_query`` directly: good SELECT, DROP
   attempt, broken SQL (the SQL_ERROR contract), missing-LIMIT auto-cap.
3. Understand the production swap-in: this JSON mirrors OpenMetadata entities
   (documented in src/metadata/catalog.py).
4. Online: run ``make_sql_analyst()`` on two business questions and read the
   tool-call trace — schema lookup BEFORE SQL is the discipline.
5. Trigger and observe SQL self-repair.

DURATION: ~90 minutes
PREREQS : data/warehouse.db + data/metadata_catalog.json (regenerated here if
          missing). Catalog + guardrail cells run OFFLINE; live agent cells
          need an API key.
"""

# %% Setup — sys.path banner, offline status
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive `# %%` cell execution
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import is_offline  # noqa: E402

OFFLINE = is_offline()
print("=" * 72)
print("LAB 08 — sql_analyst: NL2SQL over the supply-chain warehouse")
print(f"OFFLINE mode: {OFFLINE}")
print("  - Catalog tour + SQL guardrails: work FULLY offline (no LLM involved).")
print("  - Live NL2SQL agent (make_sql_analyst + trace): needs an API key.")
print("=" * 72)

# Ensure warehouse + catalog exist (idempotent, seed 42).
_db = PROJECT_ROOT / "data" / "warehouse.db"
_cat = PROJECT_ROOT / "data" / "metadata_catalog.json"
if not _db.exists() or not _cat.exists():
    if not (PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv").exists():
        from data import generate_supply_chain_data

        generate_supply_chain_data.main()
    from data import build_database

    build_database.main()
    print("(regenerated data/warehouse.db + data/metadata_catalog.json)")

# %% The raw catalog — open data/metadata_catalog.json and look around
# Before any abstraction, see the artifact itself: an OpenMetadata-style
# document with a service block, table entities (columns + descriptions +
# tags + sampleQueries) and a business GLOSSARY. The glossary is the secret
# weapon: it defines what 'Late Delivery' MEANS, not just where it is stored.
raw = json.loads(_cat.read_text(encoding="utf-8"))
print(f"Top-level keys : {list(raw.keys())}")
print(f"Service        : {raw['service']}")
for t in raw["tables"]:
    print(f"  table {t['name']:<10} cols={len(t['columns']):>2}  tags={t.get('tags')}")
print(f"Glossary terms : {[g['term'] for g in raw['glossary']]}")

# %% MetadataCatalog tour — the prompt-ready view
# MetadataCatalog renders those JSON entities as TEXT, because text is what
# goes into the agent's context. Three calls matter:
from src.metadata.catalog import MetadataCatalog  # noqa: E402

catalog = MetadataCatalog()
print(f"\nlist_tables() → {catalog.list_tables()}")

print("\nget_table_schema('orders') →\n")
print(catalog.get_table_schema("orders"))

# search() ranks tables by keyword relevance. Watch the query phrasing:
# the single token 'penalty' matches weakly everywhere (a near-tie), while
# a richer phrase ranks `carriers` first — it is the contracts-bridge table
# (contract_file, on_time_sla_pct). Agents learn this the same way you just
# did: more specific search terms → sharper table ranking.
print(f"\nsearch('penalty')                     → {catalog.search('penalty')}")
print(f"search('carrier contract penalty SLA') → "
      f"{catalog.search('carrier contract penalty SLA')}")

print("\nget_glossary() →\n")
print(catalog.get_glossary())
# 💡 CONSULTANT'S NOTE — the glossary entry 'Late Delivery: actual > scheduled'
# is business logic that lives NOWHERE in the DDL. Without it, an LLM
# plausibly invents `WHERE shipping_date > order_date + 7`. Metadata is not
# documentation overhead; it is the difference between guessing and knowing.

# %% Guardrails — call run_sql_query DIRECTLY and try to break it
# Tools are just functions; test them like functions, before any LLM touches
# them. Four probes: the happy path, a destructive attempt, broken SQL, and
# a missing LIMIT.
from src.agents.tools_sql import run_sql_query  # noqa: E402

good_sql = (
    "SELECT c.carrier_name, COUNT(*) AS n_orders, "
    "ROUND(AVG(o.late_delivery), 3) AS late_rate "
    "FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id "
    "GROUP BY c.carrier_name ORDER BY late_rate DESC LIMIT 10"
)
print("\n[1] Good SELECT:\n")
out_good = run_sql_query.invoke({"sql": good_sql})
print(out_good)
assert "SQL_ERROR" not in out_good

print("\n[2] DROP attempt — must be rejected, never executed:\n")
out_drop = run_sql_query.invoke({"sql": "DROP TABLE orders"})
print(out_drop)
assert "SQL_ERROR" in out_drop, "guardrail must reject non-SELECT statements"

print("\n[3] Broken SQL — the SQL_ERROR string contract:\n")
out_broken = run_sql_query.invoke({"sql": "SELECT carrier_namez FROM carriers"})
print(out_broken)
assert out_broken.startswith("SQL_ERROR"), "errors must come back as SQL_ERROR: <msg>"
# 💡 Why a STRING and not an exception? An exception kills the agent loop; a
# string becomes a ToolMessage the model READS — and uses to fix its query.
# Error messages are prompts now. Write them like you want them acted on.

print("\n[4] Missing LIMIT — auto-cap protects context window and warehouse:\n")
out_nolimit = run_sql_query.invoke({"sql": "SELECT order_id, sales FROM orders"})
n_lines = len(out_nolimit.splitlines())
print(out_nolimit[:400])
print(f"... output is {n_lines} lines — LIMIT 50 was auto-appended; "
      "12,000 rows never hit the prompt.")
assert "SQL_ERROR" not in out_nolimit and n_lines < 80

# %% Production note — from JSON file to a real OpenMetadata server
# This catalog file mirrors OpenMetadata's Table and Glossary entities 1:1.
# The swap-in path is documented in src/metadata/catalog.py: same public
# interface (list_tables / get_table_schema / search / get_glossary), backed
# by the openmetadata-ingestion SDK via OPENMETADATA_HOST_PORT + a JWT token.
# What a REAL catalog buys the client beyond this file:
#   * SCHEMA DRIFT  — ingestion pipelines detect added/renamed/dropped columns
#     and version every change; your agent's "knowledge" stays current without
#     redeploys (stale schema = confidently wrong SQL).
#   * PII TAGS      — columns tagged PII.Sensitive can be masked or refused at
#     the tool layer; the agent physically cannot SELECT what governance bans.
#   * GOVERNANCE    — ownership, lineage, tiering: when the agent misbehaves,
#     you know which team owns the table and what feeds it.
print("\n[Production note] See src/metadata/catalog.py docstring for the "
      "OpenMetadata swap-in sketch (host/JWT, same interface).")

# %% Live agent — ONLINE ONLY: two questions, with the full tool-call trace
# The system prompt encodes the NL2SQL loop: inspect schema → draft SQL →
# run → on SQL_ERROR repair (max 2 retries) → summarize in plain English.
# The trace below is the deliverable: verify schema lookup happens BEFORE SQL.
AGENT_QUESTIONS = [
    "Which carrier had the worst late delivery rate last quarter?",
    "What is the average profit on late versus on-time orders, by market?",
]

if not OFFLINE:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from src.agents.workers import make_sql_analyst

    sql_agent = make_sql_analyst()
    for i, q in enumerate(AGENT_QUESTIONS, 1):
        print("\n" + "-" * 72)
        print(f"Q{i}: {q}")
        result = sql_agent.invoke({"messages": [HumanMessage(content=q)]})
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"  [tool call ] {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                print(f"  [tool result] {str(msg.content)[:140]}...")
        print(f"\nANSWER:\n{result['messages'][-1].content}")
    # 💡 Check the trace: get_table_schema/search_metadata BEFORE run_sql_query.
    # If the agent jumps straight to SQL, its system prompt needs tightening.
else:
    print("\n[SKIP] Live sql_analyst skipped: OFFLINE mode (no API key).")
    # The discipline, run by hand — this IS what the agent automates:
    print("\nManual NL2SQL walk-through for Q1 (offline):")
    print("  step 1 — find the tables:", catalog.search("carrier late rate"))
    print("  step 2 — read the schema (orders + carriers) … done above.")
    q1_sql = (
        "SELECT c.carrier_name, COUNT(*) AS n_orders, "
        "ROUND(AVG(o.late_delivery), 3) AS late_rate "
        "FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id "
        "WHERE o.order_date BETWEEN '2025-10-01' AND '2025-12-31' "
        "GROUP BY c.carrier_name ORDER BY late_rate DESC LIMIT 10"
    )
    print("  step 3 — draft SQL grounded in that schema ('last quarter' = Q4-2025,")
    print("           the last full quarter in the data):")
    print(f"           {q1_sql}")
    print("  step 4 — run it through the guarded tool:\n")
    print(run_sql_query.invoke({"sql": q1_sql}))
    print("  step 5 — summarize for the business: the top row is your answer.")

# %% 🎯 EXERCISE 8.1 — engineer a failure, watch the self-repair
# Write a business question whose FIRST SQL draft should fail — e.g. one that
# tempts the model to invent a column that does not exist — then (online) run
# it through the agent and count the run_sql_query calls in the trace.
# Offline, simulate both halves of the loop yourself. Try before unfolding.

# %% ✅ ANSWER 8.1 (fold this cell)
# region ANSWER ----------------------------------------------------------
# Question: "What is the average late fee charged per carrier?"
# Trap: there is no late_fee column anywhere — penalty terms live in the
# CONTRACTS, not the warehouse. A good first draft fails; a good agent reads
# the SQL_ERROR, re-checks the schema, and pivots.
bad_draft = "SELECT carrier_name, AVG(late_fee) FROM orders GROUP BY carrier_name"
print("First draft :", bad_draft)
out_bad = run_sql_query.invoke({"sql": bad_draft})
print("Tool returns:", out_bad[:200])
assert out_bad.startswith("SQL_ERROR")

repaired = (
    "SELECT c.carrier_name, SUM(o.late_delivery) AS late_orders, "
    "COUNT(*) AS total_orders, ROUND(AVG(o.late_delivery), 3) AS late_rate "
    "FROM orders o JOIN carriers c ON o.carrier_id = c.carrier_id "
    "GROUP BY c.carrier_name ORDER BY late_rate DESC LIMIT 10"
)
print("\nRepaired (after re-reading the schema):", repaired, "\n")
print(run_sql_query.invoke({"sql": repaired}))
print("""
LESSON: the repair has TWO parts. (1) Mechanical: the error says 'no such
column: carrier_name' — SQLite stops at the FIRST bad reference, and
carrier_name lives in `carriers`, not `orders`. Fixing the join would only
expose the next error: late_fee does not exist either. One SQL_ERROR per
round trip is exactly why the agent gets 2 repair retries. (2) Semantic:
fee AMOUNTS are simply not in this database — they are contract terms. The
honest answer combines warehouse late-counts with the contracts_analyst's
penalty clauses. That hand-off between agents is exactly what the
supervisor orchestrates in lab09.
""")
# endregion ---------------------------------------------------------------

# %% Wrap-up
# 💡 CONSULTANT'S NOTE — the NL2SQL pitch in three lines:
#   * Grounding: the agent queries what the catalog SAYS exists, with the
#     business meaning the glossary defines. No guessed columns, no folk SQL.
#   * Safety: SELECT-only + single statement + auto-LIMIT means the worst
#     possible outcome is a wrong answer — never a damaged warehouse.
#   * Resilience: SQL_ERROR-as-string turns failures into one more reasoning
#     step. Demo the self-repair to clients; it builds more trust than any
#     accuracy slide.
print("\n" + "=" * 72)
print("Lab 08 complete. Next: lab09 wires data_analyst, ml_engineer,")
print("contracts_analyst and sql_analyst under one supervisor.")
print("=" * 72)
