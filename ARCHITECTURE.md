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
├── requirements.txt
├── .env.example
├── main.py                      ← CLI demo: python main.py --demo | --ask "..."
├── data/
│   ├── generate_supply_chain_data.py   # → data/raw/supply_chain_orders.csv
│   ├── generate_contracts.py           # → data/contracts/*.md  (6 contracts)
│   ├── build_database.py               # → data/warehouse.db + data/metadata_catalog.json
│   ├── raw/                            # generated (gitignore-able)
│   └── contracts/                      # generated
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
│   │   ├── workers.py
│   │   └── supervisor.py
│   └── graph.py                 # build_graph() → compiled LangGraph app
├── labs/                        # 9 teaching labs, .py with `# %%` cell markers
│   ├── lab01_eda.py
│   ├── lab02_preprocessing_baseline.py
│   ├── lab03_imbalance_boosting_threshold.py
│   ├── lab04_langgraph_fundamentals.py
│   ├── lab05_supervisor_two_agents.py
│   ├── lab06_pageindex_build.py
│   ├── lab07_contracts_rag_agent.py
│   ├── lab08_nl2sql_agent.py
│   └── lab09_full_system_integration.py
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
    └── test_graph_routing.py
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

---

## 5. Offline Mode (critical for classrooms without API keys)

`is_offline()` is True when `OFFLINE=1` or no provider key is set. Behavior:
- `pageindex`: heading-based summaries; keyword retrieval. Works fully.
- `ml_pipeline`, `metadata`, tools_sql guardrails: no LLM involved. Work fully.
- `graph`/`workers`: constructing them requires an LLM → labs/tests must mock.
  Tests use `langchain_core.language_models.fake_chat_models.GenericFakeChatModel`
  or monkeypatched `get_llm`.
- Every lab MUST start with a banner cell stating which parts need a key.

## 6. Testing Contract

- `tests/conftest.py`: session fixtures that run the three data generators into a
  tmp dir OR the real `data/` dir if already generated (prefer: generate into real
  `data/` once — generators must be idempotent and fast).
- All tests pass **offline** (no API keys, no network): mock LLM for routing test.
- `test_graph_routing.py`: monkeypatch supervisor's LLM with a fake structured-output
  model; assert supervisor → worker → supervisor → FINISH flow works on the
  compiled graph with a scripted route sequence.
- Run with: `python -m pytest tests/ -q` from the `week3&4` directory.

## 7. Conventions

- Python ≥ 3.11 syntax OK (project runs on 3.14). Type hints everywhere.
- Docstrings are teaching material: every module starts with a `"""WHY this exists"""`
  block written for a consultant learning the stack.
- No `xgboost`/`shap`/`imblearn` hard imports (guarded `try/except ImportError` only).
- Seeds: `RANDOM_STATE = 42` everywhere.
- Generators idempotent: re-running overwrites deterministically.
- Keep individual files < ~350 lines; split if larger.
