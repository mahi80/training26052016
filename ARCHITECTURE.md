# Week 3 & 4 — LangGraph Multi-Agent Supply Chain Intelligence
## Architecture & Module Contract (single source of truth)

This document is the **binding contract** for every module in this project. All code,
labs, exercises, and tests MUST follow the file paths, function signatures, and data
schemas defined here exactly.

---

## 1. Use Case

A logistics company suffers late shipments (customer penalties, churn). We build a
**LangGraph supervisor multi-agent system** that an operations analyst can talk to:

| Agent (worker)       | Capability                                                                  | Backing tech                          |
|----------------------|-----------------------------------------------------------------------------|---------------------------------------|
| `data_analyst`       | EDA on order data: late-rate by mode/region/category, trends, class balance | pandas tools over canonical CSV        |
| `ml_engineer`        | Train / evaluate / tune late-delivery classifiers, explain drivers          | scikit-learn pipeline (`src/ml_pipeline`) |
| `contracts_analyst`  | Answer questions about procurement / carrier contracts (penalties, SLAs)    | PageIndex-style reasoning RAG (`src/pageindex`) |
| `sql_analyst`        | NL → SQL over the warehouse, schema-grounded via a metadata catalog         | SQLite + OpenMetadata-style catalog (`src/metadata`) |

A **supervisor** node routes each user request to the right worker (or several in
sequence), then synthesizes the final answer.

**Flagship demo question** (touches 3 agents):
> "Order 104872 ships Same Day via SwiftShip Express to Western Europe — how likely is
> it to be late, what penalty applies under the SwiftShip contract if it is, and what
> was SwiftShip's late rate last quarter?"

---

## 2. Directory Layout (fixed)

```
week3&4/
├── ARCHITECTURE.md              ← this file
├── README.md                    ← setup + quickstart + repo tour
├── TRAINING_PLAN.md             ← 2-week day-by-day curriculum
├── CHALLENGES_GUIDE.md          ← production issues & challenges guide (whiteboarding companion)
├── BUILD_MANUAL.md              ← step-by-step build of the full architecture (v2 layers)
├── requirements.txt
├── .env.example
├── main.py                      ← CLI demo: python main.py --demo | --ask "..."
├── data/
│   ├── generate_supply_chain_data.py   # → data/raw/supply_chain_orders.csv
│   ├── generate_contracts.py           # → data/contracts/*.md  (6 contracts)
│   ├── contract_texts.py               # verbatim contract section bodies (imported by generate_contracts)
│   ├── render_contracts_pdf.py         # → data/contracts_pdf/*.pdf (realistic PDF versions)
│   ├── build_database.py               # → data/warehouse.db + data/metadata_catalog.json
│   ├── raw/                            # generated (gitignore-able)
│   ├── contracts/                      # generated (markdown — parsed by PageIndex)
│   └── contracts_pdf/                  # generated (PDF — realistic client-facing artifacts)
├── src/
│   ├── __init__.py
│   ├── config.py                # get_llm(), is_offline()  [ALREADY WRITTEN — do not change]
│   ├── state.py                 # SupplyChainState, WORKERS [ALREADY WRITTEN — do not change]
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── eda.py
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   ├── pageindex/
│   │   ├── __init__.py
│   │   ├── tree_builder.py
│   │   └── retriever.py
│   ├── metadata/
│   │   ├── __init__.py
│   │   └── catalog.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── tools_ml.py
│   │   ├── tools_rag.py
│   │   ├── tools_sql.py
│   │   ├── tools_mcp.py         # v2: @tool wrappers dialing MCP servers (§4.9)
│   │   ├── tools_a2a.py         # v2: ask_carrier_agent over A2A (§4.10)
│   │   ├── workers.py           # + v2: make_logistics_coordinator appended
│   │   ├── supervisor.py
│   │   ├── supervisor_v2.py     # v2: RouterV2 incl. logistics_coordinator (§4.8)
│   │   └── validator.py         # v2: Verdict judge node (§4.11)
│   ├── mcp_servers/             # v2: mock SAP/ServiceNow + SQL shim + sync client (§4.9)
│   │   ├── __init__.py
│   │   ├── sap_server.py
│   │   ├── servicenow_server.py
│   │   ├── sql_server.py
│   │   └── client.py
│   ├── a2a/                     # v2: external carrier partner agent service (§4.10)
│   │   ├── __init__.py
│   │   └── external_agent.py
│   ├── state_v2.py              # v2: SupplyChainStateV2 + WORKERS_V2 (§4.8)
│   ├── graph.py                 # build_graph() → compiled LangGraph app
│   ├── graph_v2.py              # v2: build_graph_v2() + validator/HITL wiring (§4.12)
│   └── gateway.py               # v2: FastAPI front door (§4.13)
├── labs/                        # 10 teaching labs, .py with `# %%` cell markers
│   ├── lab01_eda.py
│   ├── lab02_preprocessing_baseline.py
│   ├── lab03_imbalance_boosting_threshold.py
│   ├── lab04_langgraph_fundamentals.py
│   ├── lab05_supervisor_two_agents.py
│   ├── lab06_pageindex_build.py
│   ├── lab07_contracts_rag_agent.py
│   ├── lab08_nl2sql_agent.py
│   ├── lab09_full_system_integration.py
│   └── lab10_production_layers.py   # v2: MCP, A2A, validator, HITL
├── exercises/
│   ├── week3_starter.py
│   ├── week3_solutions_TRAINER_ONLY.py
│   ├── week3_quiz.py
│   ├── week4_starter.py
│   ├── week4_solutions_TRAINER_ONLY.py
│   └── week4_quiz.py
├── models/                      # saved model artifacts (created at runtime)
└── tests/
    ├── conftest.py
    ├── test_data_generation.py
    ├── test_ml_pipeline.py
    ├── test_pageindex.py
    ├── test_metadata_catalog.py
    ├── test_nl2sql_tools.py
    ├── test_graph_routing.py
    ├── test_mcp_servers.py      # v2
    ├── test_a2a.py              # v2
    ├── test_validator.py        # v2
    ├── test_graph_v2_hitl.py    # v2
    └── test_gateway.py          # v2
```

**Path convention:** all modules resolve paths relative to the project root via
`pathlib.Path(__file__).resolve()` — never hardcode absolute paths. Project root =
the `week3&4` directory. Define `PROJECT_ROOT` locally in each entry-point module as
needed (`Path(__file__).resolve().parents[N]`).

---

## 3. Canonical Data Schema

### 3.1 Flat ML dataset — `data/raw/supply_chain_orders.csv` (~12,000 rows, seed=42)

| column                   | dtype   | notes                                                            |
|--------------------------|---------|------------------------------------------------------------------|
| `order_id`               | int     | unique, starts 10000                                             |
| `order_date`             | date    | ISO `YYYY-MM-DD`, range 2024-01-01 … 2025-12-31                  |
| `shipping_date`          | date    | order_date + actual_shipping_days  **(LEAKAGE — drop for ML)**   |
| `scheduled_shipping_days`| int     | by mode: Same Day=1, First Class=2, Second Class=4, Standard=6   |
| `actual_shipping_days`   | int     | **(LEAKAGE — used only to derive target)**                       |
| `shipping_mode`          | str     | Same Day / First Class / Second Class / Standard Class           |
| `carrier_name`           | str     | one of the 4 carriers (§3.3)                                     |
| `customer_id`            | int     |                                                                  |
| `customer_segment`       | str     | Consumer / Corporate / Home Office                               |
| `market`                 | str     | LATAM / Europe / Pacific Asia / USCA / Africa                    |
| `order_region`           | str     | plausible regions nested under market                            |
| `order_country`          | str     | plausible countries nested under region                          |
| `category_name`          | str     | ~10 product categories                                           |
| `product_name`           | str     |                                                                  |
| `order_item_quantity`    | int     | 1–5                                                              |
| `sales`                  | float   |                                                                  |
| `discount`               | float   | 0–0.25                                                           |
| `profit`                 | float   | can be negative                                                  |
| `late_delivery`          | int 0/1 | TARGET = `actual_shipping_days > scheduled_shipping_days`        |

**Planted signal** (so models genuinely learn): late probability rises with
Same Day & First Class modes (tight schedules), Africa & LATAM markets, Q4 + holiday
months, high quantity, certain carriers (SwiftShip worst, NordHaul best). Overall late
rate ≈ 30 % (class imbalance is intentional, it drives the Week-3 imbalance lesson).

**Real-dataset swap-in:** `data_loader.py` exposes `DATACO_COLUMN_MAP` translating the
Kaggle DataCo CSV headers (e.g. `Days for shipment (scheduled)` →
`scheduled_shipping_days`, `Late_delivery_risk` → `late_delivery`) so trainees can drop
in the real 180k-row file.

### 3.2 Warehouse — `data/warehouse.db` (SQLite, built from the CSV)

```sql
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_segment TEXT);
CREATE TABLE products  (product_id INTEGER PRIMARY KEY, product_name TEXT,
                        category_name TEXT);
CREATE TABLE carriers  (carrier_id INTEGER PRIMARY KEY, carrier_name TEXT,
                        contract_file TEXT, on_time_sla_pct REAL);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT, shipping_date TEXT,
    customer_id INTEGER REFERENCES customers(customer_id),
    product_id  INTEGER REFERENCES products(product_id),
    carrier_id  INTEGER REFERENCES carriers(carrier_id),
    shipping_mode TEXT,
    scheduled_shipping_days INTEGER, actual_shipping_days INTEGER,
    late_delivery INTEGER,
    order_item_quantity INTEGER, sales REAL, discount REAL, profit REAL,
    market TEXT, order_region TEXT, order_country TEXT
);
```

### 3.3 Carriers & contracts (6 documents in `data/contracts/`)

| file                                   | party                    | type                         |
|----------------------------------------|--------------------------|------------------------------|
| `swiftship_express_msa.md`             | SwiftShip Express        | Carrier MSA (Same Day / First Class lanes) |
| `atlas_freight_msa.md`                 | Atlas Freight Co.        | Carrier MSA (Standard lanes) |
| `pacific_crest_carriers_msa.md`        | Pacific Crest Carriers   | Carrier MSA (Pacific Asia)   |
| `nordhaul_logistics_msa.md`            | NordHaul Logistics       | Carrier MSA (Europe)         |
| `meridian_supply_procurement.md`       | Meridian Supply Partners | Supplier procurement agreement |
| `helios_components_procurement.md`     | Helios Components Ltd.   | Supplier procurement agreement |

Each contract: markdown, 1,500–3,000 words, numbered sections (`## 1. Parties`,
`## 5. Service Levels`, `### 5.2 On-Time Delivery Commitment`, `## 6. Late Delivery
Penalties`, …) with **distinct, quotable numbers** (e.g. SwiftShip: 2 % of shipment
value per late day, capped 15 %, on-time SLA 96 %; Atlas: flat $250 per late
shipment >48 h; etc.) so RAG answers are verifiable. Penalty/SLA numbers MUST differ
across contracts.

### 3.4 Metadata catalog — `data/metadata_catalog.json` (OpenMetadata-style)

```json
{
  "service": {"name": "supply_chain_warehouse", "type": "sqlite",
               "database": "data/warehouse.db"},
  "tables": [
    {"name": "orders", "description": "...",
     "columns": [{"name": "order_id", "dataType": "INTEGER",
                   "description": "...", "tags": ["PK"]}, ...],
     "tags": ["fact"], "sampleQueries": ["SELECT ..."]}
  ],
  "glossary": [
    {"term": "Late Delivery",
     "definition": "actual_shipping_days > scheduled_shipping_days",
     "mappedAssets": ["orders.late_delivery"]},
    {"term": "On-Time SLA", "definition": "...", "mappedAssets": ["carriers.on_time_sla_pct"]}
  ]
}
```

Generated by `build_database.py`. Mention in docs: in production this lives in an
**OpenMetadata** server (the JSON mirrors its Table/Glossary entities; `catalog.py`
documents how to point at a real instance via `OPENMETADATA_HOST_PORT` + JWT, using the
same public interface).

### 3.5 Mock live-systems data (v2 — in-code, deterministic, no generators)

`src/mcp_servers/sap_server.py` — `PURCHASE_ORDERS: list[dict]`, 10 rows over the six
§3.3 contract parties; fields `po_number, vendor, material, quantity, value_usd,
promised_date, status` with `status ∈ {Open, In Transit, Delivered, Delayed}`.
SwiftShip Express has the most `Delayed` rows (consistent with the §3.1 planted signal).

`src/mcp_servers/servicenow_server.py` — `INCIDENTS: list[dict]`, 8 tickets
`INC0010001…INC0010008`; fields `number, carrier, short_description, priority (P1–P4),
state ∈ {New, In Progress, Resolved}, opened_at`. SwiftShip has the only open P1.

`src/a2a/external_agent.py` — `AGENT_CARD: dict` (name/description/version/url/skills)
and `CARRIER_OPS: dict[str, str]` (fleet / pickup / congestion narratives). All three
datasets are hand-written literals: deterministic by construction, no seeding needed.

---

## 4. Module Contracts (exact signatures)

### 4.1 `src/config.py`  — ALREADY WRITTEN, import from it, do not modify

```python
get_llm(temperature: float = 0.0, **kwargs) -> BaseChatModel   # provider from env
is_offline() -> bool   # True when OFFLINE=1 or no API keys → all modules must degrade
PROJECT_ROOT: Path     # the week3&4 directory
```

### 4.2 `src/state.py` — ALREADY WRITTEN

```python
WORKERS: tuple[str, ...]  # ("data_analyst","ml_engineer","contracts_analyst","sql_analyst")
ROUTE_OPTIONS: tuple[str, ...]  # WORKERS + ("FINISH",)
class SupplyChainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
```

### 4.3 `src/ml_pipeline/`

```python
# data_loader.py
DATACO_COLUMN_MAP: dict[str, str]
load_orders(path: str | Path | None = None, drop_leakage: bool = False) -> pd.DataFrame
    # default path data/raw/supply_chain_orders.csv; drop_leakage drops
    # shipping_date, actual_shipping_days; parses date columns to datetime

# eda.py  — every function returns a pd.DataFrame or dict (JSON-serializable-ish)
late_rate_by(df, dimension: str) -> pd.DataFrame       # columns: <dimension>, n_orders, late_rate
class_balance(df) -> dict                               # {"on_time": int, "late": int, "late_rate": float}
monthly_trend(df) -> pd.DataFrame                       # month, n_orders, late_rate
summary_stats(df) -> dict

# features.py
engineer_features(df) -> pd.DataFrame
    # adds: order_month, order_weekday, is_weekend, is_q4, quantity_bucket,
    # mode_schedule_tightness (scheduled days normalized per mode)
    # MUST NOT touch leakage columns
NUMERIC_FEATURES: list[str]; CATEGORICAL_FEATURES: list[str]; TARGET = "late_delivery"

# preprocess.py
split_data(df, test_size=0.2, random_state=42) -> tuple[X_train, X_test, y_train, y_test]
    # stratified on target
build_preprocessor() -> sklearn ColumnTransformer  # OneHotEncoder(handle_unknown="ignore") + StandardScaler

# train.py
MODEL_REGISTRY: dict[str, callable]   # {"logreg": ..., "random_forest": ..., "hist_gb": ...}
    # all with class_weight="balanced" where supported; hist_gb is the
    # LightGBM-equivalent sklearn HistGradientBoostingClassifier.
    # If xgboost importable, also register "xgboost" (guarded import).
train_model(model_name="hist_gb", df=None) -> dict
    # fits Pipeline(preprocessor, model), saves joblib to models/<name>.joblib,
    # returns {"model_name", "model_path", "cv_recall", "cv_roc_auc", "train_seconds"}

# evaluate.py
evaluate_model(model_name="hist_gb", threshold=0.5) -> dict
    # {"recall","precision","f1","roc_auc","pr_auc","confusion_matrix":[[tn,fp],[fn,tp]],"threshold"}
tune_threshold(model_name="hist_gb", min_precision=0.5) -> dict
    # sweeps thresholds, returns best-recall threshold subject to precision floor

# explain.py
feature_drivers(model_name="hist_gb", top_k=10) -> pd.DataFrame
    # permutation_importance on the held-out split; columns: feature, importance
    # If shap importable, optional shap_drivers() too (guarded).
```

State between train/evaluate: `train_model` persists the fitted pipeline AND the
train/test split indices (joblib dict: `{"pipeline": ..., "split": {...}}`) so
`evaluate_model`/`tune_threshold`/`feature_drivers` reload from `models/`.

### 4.4 `src/pageindex/` — vectorless, reasoning-based RAG (PageIndex style)

Concept (teach this in docstrings): instead of chunk-embed-similarity, build a
**hierarchical tree index** of each document (like a table of contents), then at query
time an LLM **reasons over the tree** to navigate to relevant nodes — like a human
expert using a ToC. No vector DB.

```python
# tree_builder.py
@dataclass class Node:  node_id: str; title: str; level: int; doc: str;
                        summary: str; text: str; children: list["Node"]
build_index(contracts_dir: Path | None = None, out_path: Path | None = None,
            use_llm: bool | None = None) -> dict
    # parses markdown headings (#/##/###) into a tree per document;
    # summary = LLM-generated (when not offline) else heading + first 240 chars;
    # persists JSON to data/contracts_index.json; returns the index dict
load_index(path: Path | None = None) -> dict

# retriever.py
@dataclass class RetrievedSection: node_id: str; doc: str; title: str; path: str;
                                   text: str; score: float
class PageIndexRetriever:
    def __init__(self, index_path: Path | None = None): ...
    def search(self, query: str, top_k: int = 3) -> list[RetrievedSection]
        # online: LLM sees the tree outline (ids/titles/summaries), selects node ids
        #   via structured output, optionally drills one level deeper
        # offline: keyword/token-overlap scoring over title+summary+text
    def outline(self) -> str   # human-readable ToC of all docs (for prompts/labs)
```

Index JSON shape: `{"docs": [{"doc": "<filename>", "title": ..., "nodes": [<Node dict>...]}]}`
(nodes nested via `children`).

### 4.5 `src/metadata/catalog.py` — OpenMetadata-style catalog client

```python
class MetadataCatalog:
    def __init__(self, catalog_path: Path | None = None): ...   # default data/metadata_catalog.json
    def list_tables(self) -> list[str]
    def get_table_schema(self, table: str) -> str       # DDL-ish text incl. column descriptions
    def get_full_schema(self) -> str                    # all tables, prompt-ready
    def search(self, query: str) -> list[str]           # table names ranked by keyword relevance
    def get_glossary(self) -> str                       # prompt-ready glossary text
```

Docstring documents the production path: same interface backed by
`openmetadata-ingestion` SDK (`OpenMetadata(OpenMetadataConnection(hostPort=..., ...))`)
— include a commented sketch, do not import the SDK.

### 4.6 `src/agents/` tools — all use `@tool` from `langchain_core.tools`,
return **strings** (markdown tables / JSON dumps), never raw DataFrames.

```python
# tools_ml.py
get_data_overview() -> str          # shape, columns, class_balance
analyze_late_rate(dimension: str) -> str        # late_rate_by as markdown table
analyze_monthly_trend() -> str
train_late_delivery_model(model_name: str = "hist_gb") -> str
evaluate_late_delivery_model(model_name: str = "hist_gb", threshold: float = 0.5) -> str
tune_decision_threshold(model_name: str = "hist_gb", min_precision: float = 0.5) -> str
explain_model_drivers(model_name: str = "hist_gb", top_k: int = 10) -> str
predict_order_risk(shipping_mode: str, market: str, category_name: str,
                   order_month: int, order_item_quantity: int = 1, ...) -> str
    # builds a one-row frame with sensible defaults, returns P(late) + risk band

# tools_rag.py
search_contracts(query: str) -> str             # top sections w/ doc+path+text
get_contract_outline() -> str                   # retriever.outline()

# tools_sql.py
list_warehouse_tables() -> str
get_table_schema(table_name: str) -> str
search_metadata(query: str) -> str
run_sql_query(sql: str) -> str
    # GUARDRAILS: single statement, SELECT-only (reject INSERT/UPDATE/DELETE/DROP/
    # ATTACH/PRAGMA), auto-append LIMIT 50 if missing, errors returned as
    # "SQL_ERROR: <msg>" strings so the agent can self-repair
```

### 4.7 `src/agents/workers.py` + `supervisor.py` + `src/graph.py`

```python
# workers.py — LangChain v1 create_agent (ReAct) per worker
make_data_analyst() / make_ml_engineer() / make_contracts_analyst() / make_sql_analyst()
    # from langchain.agents import create_agent
    # create_agent(model=get_llm(), tools=[...], system_prompt=<role prompt>)
    # sql_analyst prompt encodes the NL2SQL loop: inspect schema → draft SQL →
    # run → on SQL_ERROR repair (max 2 retries) → summarize rows in plain English

# supervisor.py
class Router(BaseModel): next: Literal[...ROUTE_OPTIONS...]; reason: str
supervisor_node(state: SupplyChainState) -> dict   # llm.with_structured_output(Router)
make_worker_node(name: str, agent) -> callable
    # invokes the ReAct agent on state messages, returns last AI message wrapped
    # as AIMessage(name=name); routes back to supervisor

# graph.py
build_graph(llm=None, workers: dict[str, Callable] | None = None,
            checkpointer=None) -> CompiledStateGraph
    # StateGraph(SupplyChainState); supervisor + 4 worker nodes;
    # conditional edges supervisor → workers/END on state["next_agent"];
    # each worker → supervisor; entry = supervisor.
    # `llm` and `workers` are dependency-injection points: tests pass a fake
    # structured-output llm and plain-callable worker nodes so the full graph
    # compiles and runs OFFLINE. Defaults (None) build the real ReAct agents.
ascii_graph() -> str   # for labs: app.get_graph().draw_ascii() fallback to mermaid text
```

`main.py`: argparse; `--ask "question"` (one-shot), `--demo` (runs 5 canned scenarios
incl. the flagship 3-agent question), `--offline-check` (verifies offline fallbacks).
Pretty-prints each agent hop (`[supervisor → ml_engineer]` etc.).

### 4.8 `src/state_v2.py` + `src/agents/supervisor_v2.py` (v2 routing)

v1 modules are **frozen**; v2 extends additively. `SupplyChainStateV2` inherits
`SupplyChainState`, so every v1 node runs unchanged on v2 state.

```python
# state_v2.py
WORKERS_V2: tuple[str, ...]        # WORKERS + ("logistics_coordinator",)
ROUTE_OPTIONS_V2: tuple[str, ...]  # WORKERS_V2 + ("FINISH",)
WORKER_DESCRIPTIONS_V2: dict[str, str]
class SupplyChainStateV2(SupplyChainState):
    verdict: str          # "" | "complete" | "needs_human"  (written by validator)
    verdict_reason: str

# supervisor_v2.py — same mechanics as supervisor.py, extended Literal
RouteNameV2 = Literal[...ROUTE_OPTIONS_V2...]   # lock-step guard at import time
class RouterV2(BaseModel): next: RouteNameV2; reason: str
make_supervisor_node_v2(llm: BaseChatModel | None = None) -> Callable[[SupplyChainStateV2], dict]
```

### 4.9 `src/mcp_servers/` + `src/agents/tools_mcp.py` (MCP layer)

Servers define tools as **plain module-level functions** registered via
`mcp.tool()(fn)` on a `FastMCP(<name>)` — testable without subprocesses; `__main__`
runs stdio transport. `sql_server` contains zero query logic: it delegates to
`tools_sql.run_sql_query` / `list_warehouse_tables` (guardrails inherited).

```python
# sap_server.py
get_purchase_orders(vendor: str | None = None, status: str | None = None) -> str  # markdown table
get_po(po_number: str) -> str
list_vendors() -> str
# servicenow_server.py
get_incidents(carrier: str | None = None, state: str | None = None) -> str
get_incident(number: str) -> str
# sql_server.py
query_warehouse(sql: str) -> str          # → tools_sql.run_sql_query
list_tables() -> str                      # → tools_sql.list_warehouse_tables

# client.py — sync stdio client (asyncio.run per call; one subprocess per call;
# production path = langchain-mcp-adapters, documented in the docstring)
SERVERS: dict[str, str]   # {"sap": "src.mcp_servers.sap_server", "servicenow": ..., "sql": ...}
call_mcp_tool(server: str, tool: str, arguments: dict | None = None, timeout: float = 30.0) -> str
list_mcp_tools(server: str, timeout: float = 30.0) -> str
    # NEVER raises — all failures return "MCP_ERROR: <msg>" strings
    # spawn = sys.executable -m <module>, cwd=PROJECT_ROOT, PYTHONPATH=PROJECT_ROOT

# tools_mcp.py — @tool wrappers, strings only
sap_purchase_orders(vendor: str = "", status: str = "") -> str
servicenow_incidents(carrier: str = "", state: str = "") -> str
warehouse_query_via_mcp(sql: str) -> str
```

### 4.10 `src/a2a/external_agent.py` + `src/agents/tools_a2a.py` (A2A layer)

A separate FastAPI service (never imported by the graph), rule-based, key-free.
A2A-style surface: agent card + JSON-RPC 2.0 `message/send`.

```python
# external_agent.py — run: python -m src.a2a.external_agent  (127.0.0.1:8001)
AGENT_CARD: dict; CARRIER_OPS: dict[str, str]
answer_carrier_question(text: str) -> str      # keyword rules over CARRIER_OPS
GET  /.well-known/agent.json                   # discovery
POST /a2a                                      # {"method": "message/send", "params": {"message": {"parts": [{"text": ...}]}}}

# tools_a2a.py
ask_carrier_agent(question: str) -> str        # @tool; httpx POST to $A2A_BASE_URL
    # (default http://127.0.0.1:8001), 5 s timeout; failures → "A2A_ERROR: ..." strings
fetch_agent_card(base_url: str | None = None) -> str   # not an LLM tool; discovery helper
```

### 4.11 `src/agents/validator.py` (validator / judge node)

```python
class Verdict(BaseModel): verdict: Literal["complete", "needs_human"]; reason: str
heuristic_verdict(question: str, answer: str) -> Verdict
    # deterministic: needs_human if answer empty, < 40 chars, or contains
    # SQL_ERROR / MCP_ERROR / A2A_ERROR; else complete
make_validator_node(llm: BaseChatModel | None = None) -> Callable[[SupplyChainStateV2], dict]
    # llm injected → LLM-as-judge via with_structured_output(Verdict)
    # llm None + offline → heuristic (DEGRADES, never raises — pageindex pattern)
    # returns {"verdict": ..., "verdict_reason": ...}
```

### 4.12 `src/graph_v2.py` (v2 graph: validator branch + HITL)

```python
build_graph_v2(llm=None, workers: dict[str, Callable] | None = None,
               validator: Callable | None = None, checkpointer=None) -> CompiledStateGraph
    # StateGraph(SupplyChainStateV2); supervisor_v2 + 5 workers + validator + human_review
    # conditional edges: supervisor → workers / "FINISH" → validator (v1 sent FINISH → END)
    #                    validator → END ("complete") / human_review ("needs_human")
    # human_review → END; same DI points as v1 + injectable validator
make_human_review_node() -> Callable
    # interrupt({question, draft_answer, validator_reason, options});
    # resume {"action": "approve"} → ship as-is; {"action": "edit", "text": ...} →
    # append AIMessage(name="human_reviewer"); both set verdict="complete"
build_demo_graph_v2(checkpointer=None) -> CompiledStateGraph
    # zero-LLM demo: keyword router + tool-backed workers (real MCP/A2A/RAG/SQL calls)
ascii_graph_v2() -> str
```

### 4.13 `src/gateway.py` (API gateway)

```python
create_app(graph_factory: Callable[[], Any] | None = None) -> FastAPI   # DI for tests
app = create_app()        # run: python -m uvicorn src.gateway:app --port 8000
GET  /health                       # never builds the graph; works keyless
POST /ask     {question, thread_id?}   # → {"status": "complete", answer, hops, ...}
                                       #   or {"status": "needs_human", review, thread_id, ...}
POST /resume  {thread_id, action: "approve"|"edit", text}   # 404 unknown / 409 not paused
GET  /threads/{thread_id}/state    # pending, paused_at, message_count, last_answer, verdict
```

Invariants: handlers are **sync `def`** (threadpool — the MCP tools call
`asyncio.run()`, which would crash in an async handler's loop); graph builds lazily on
first request (offline → `build_demo_graph_v2`, responses tagged `"mode":
"offline-demo"`); checkpointer is `MemorySaver` (threads are volatile across restarts).

---

## 5. Offline Mode (critical for classrooms without API keys)

`is_offline()` is True when `OFFLINE=1` or no provider key is set. Behavior:
- `pageindex`: heading-based summaries; keyword retrieval. Works fully.
- `ml_pipeline`, `metadata`, tools_sql guardrails: no LLM involved. Work fully.
- `graph`/`workers`: constructing them requires an LLM → labs/tests must mock.
  Tests use `langchain_core.language_models.fake_chat_models.GenericFakeChatModel`
  or monkeypatched `get_llm`.
- v2 layers: `mcp_servers` (servers + client) and `a2a/external_agent` involve **no
  LLM** — work fully offline. `validator` degrades to `heuristic_verdict`.
  `gateway` serves `build_demo_graph_v2` (keyword router, real tools) offline and tags
  responses `"mode": "offline-demo"`; `build_graph_v2` itself requires an LLM or fakes.
- Every lab MUST start with a banner cell stating which parts need a key.

## 6. Testing Contract

- `tests/conftest.py`: session fixtures that run the three data generators into a
  tmp dir OR the real `data/` dir if already generated (prefer: generate into real
  `data/` once — generators must be idempotent and fast).
- All tests pass **offline** (no API keys, no network): mock LLM for routing test.
- `test_graph_routing.py`: monkeypatch supervisor's LLM with a fake structured-output
  model; assert supervisor → worker → supervisor → FINISH flow works on the
  compiled graph with a scripted route sequence.
- v2 tests (same offline rule): `test_mcp_servers.py` (direct functions + inherited SQL
  guardrails + `MCP_ERROR` contract; one real stdio round-trip marked `slow`),
  `test_a2a.py` (TestClient on the external agent + `A2A_ERROR` contract),
  `test_validator.py` (heuristic + scripted `Verdict` judge), `test_graph_v2_hitl.py`
  (interrupt/resume on the real compiled v2 graph with `MemorySaver`),
  `test_gateway.py` (TestClient over `create_app` with an injected fake-graph factory).
- Run with: `python -m pytest tests/ -q` from the `week3&4` directory.

## 7. Conventions

- Python ≥ 3.11 syntax OK (project runs on 3.14). Type hints everywhere.
- Docstrings are teaching material: every module starts with a `"""WHY this exists"""`
  block written for a consultant learning the stack.
- No `xgboost`/`shap`/`imblearn` hard imports (guarded `try/except ImportError` only).
- Seeds: `RANDOM_STATE = 42` everywhere.
- Generators idempotent: re-running overwrites deterministically.
- Keep individual files < ~350 lines; split if larger.
- **v1 is frozen, v2 is additive**: never modify v1 signatures (§4.1–4.7) — new
  capability lands as `*_v2` modules or appended factories (§4.8–4.13). Errors cross
  layer boundaries as strings (`SQL_ERROR:` / `MCP_ERROR:` / `A2A_ERROR:`), never as
  exceptions.
