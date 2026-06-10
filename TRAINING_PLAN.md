# Training Plan — Weeks 3 & 4: The Agent Forge & The Orchestrator's Citadel

A 10-day, day-by-day curriculum for trainers delivering Weeks 3 & 4 of the
**Zero to Hero** consultant program. Everything here maps onto the code contract in
[ARCHITECTURE.md](ARCHITECTURE.md); setup and quickstart live in [README.md](README.md).

---

## How these weeks fit the program

**What Weeks 1–2 gave us.** Trainees arrive with ML foundations (supervised learning,
train/test discipline, metrics), data-engineering basics (pandas, SQL literacy), and
prompt engineering. They have *not* yet built an agent, a graph, or a RAG system.

**What these weeks do.** Weeks 3 & 4 are the program's pivot: instead of isolated
exercises, trainees build **one coherent system** — a LangGraph multi-agent
"supply chain ops copilot" — combining a classical ML classifier, PageIndex-style
vectorless RAG over procurement contracts, and metadata-grounded NL2SQL, all under a
supervisor router. Week 3 (*The Agent Forge*) forges the parts: the ML core and
LangGraph fundamentals. Week 4 (*The Orchestrator's Citadel*) assembles the whole:
RAG, NL2SQL, and full integration.

**What Weeks 5+ assume.** Week 6 (Azure AI Agent Service & MCP) assumes trainees can
already reason about agents, tools, and routing — they map LangGraph concepts onto
managed Azure services. Week 6.5 (RAG infrastructure) contrasts the vectorless
PageIndex approach built here with embedding/vector-store pipelines. Week 7
(multi-agent security) attacks the very system built in these two weeks (prompt
injection via tools, SQL exfiltration, over-permissive agents) — so the guardrails
trainees write in lab08 become Week 7's case study.

**Gamification.** Each week ends in a 300 XP trial plus a quiz. Daily checkpoints are
oral/whiteboard, not graded. Stretch goals are optional flex for fast finishers.

---

## Learning outcomes

By the end of Week 4, every trainee can — measurably:

1. **Diagnose data leakage** in a tabular dataset and name the leaking columns and the mechanism (lab02; quiz).
2. **Build a leakage-free sklearn Pipeline** (ColumnTransformer + model) with a stratified split, seeded at 42 (lab02–03; trial).
3. **Choose and defend metrics for an imbalanced target** (~30% positive): explain why accuracy misleads and optimize recall under a precision floor via `tune_threshold` (lab03; checkpoint).
4. **Explain model drivers to a non-technical stakeholder** using permutation importance (lab03; capstone).
5. **Define a LangGraph state machine from scratch**: TypedDict state, reducer semantics of `add_messages`, nodes, conditional edges, compile, invoke (lab04; trial).
6. **Wrap business logic as LangChain `@tool`s** that return strings, and assemble a ReAct agent with `create_agent` (lab04–05; trial).
7. **Implement the supervisor pattern**: structured-output routing over `ROUTE_OPTIONS`, workers reporting back, `FINISH` termination (lab05, lab09; trial).
8. **Build and query a vectorless PageIndex tree** over markdown contracts, and articulate when reasoning-RAG beats chunk-embed-cosine (lab06–07; quiz).
9. **Design a grounded, citing RAG agent** whose answers carry document + section paths and that says "not covered" when retrieval is empty (lab07; capstone).
10. **Ship guard-railed NL2SQL**: metadata-catalog grounding, SELECT-only enforcement, auto-LIMIT, and the `SQL_ERROR` self-repair loop (lab08; trial).
11. **Run and narrate the full multi-agent system** end to end, including the flagship 3-agent question, with per-hop tracing (lab09; capstone demo).
12. **Operate offline-first**: state which components degrade without an API key and prove it with `python main.py --offline-check` and a green test suite (every lab; logistics).

---

## Trainer logistics box

> **Read this before Day 1.**
>
> - **Install time:** ~10–15 min on decent Wi-Fi (`pip install -r requirements.txt`); the three data generators run in well under a minute. Budget 30 min on Day 1 morning for environment triage.
> - **Keys:** the curriculum default is `LLM_PROVIDER=azure_openai` (one shared classroom resource + per-trainee `.env` works fine; temperature is 0 so costs are predictable). `openai` and `anthropic` are drop-in alternatives.
> - **No-key contingency:** the room can run **labs 01–03 and 06 in full, plus the offline portions of every other lab**, and the entire test suite, with `OFFLINE=1`. Every lab's first cell is a banner stating which cells need a key. If keys die mid-class, pivot to `python main.py --offline-check`, the offline cells, and whiteboard walkthroughs of the agent cells.
> - **Windows gotcha #1 — the `&` in the path:** the project directory is `week3&4`. In PowerShell, `&` is the call operator — **always quote**: `cd "C:\...\zeroToHero_Training\week3&4"`. Unquoted paths fail confusingly in both PowerShell and cmd.
> - **Windows gotcha #2 — backslashes:** quickstart commands use `python data\generate_supply_chain_data.py`; macOS/Linux trainees use `/`.
> - **Windows gotcha #3 — OneDrive:** if the repo lives under OneDrive, pause syncing during labs (file locks on `warehouse.db` and `__pycache__` churn cause flaky errors).
> - **Run labs cell-by-cell:** labs are `.py` files with `# %%` markers — VS Code's Python extension renders them as interactive cells. No Jupyter required.
> - **Generators are idempotent and seeded (42):** re-running them is always safe; "regenerate the data" is the first fix for any data-shaped error.
> - **Sanity ritual each morning:** `python -m pytest tests/ -q` (all green offline) then `python main.py --offline-check`.

---

# WEEK 3 — The Agent Forge

*Forge the blade before you raise the citadel: an honest ML core, then the LangGraph machinery to wield it.*

---

## W3·D1 — Scouting the Supply Lines (EDA)

**Morning concept block (60–90 min) — trainer talking points**
- The client story: late shipments → penalties + churn; our job is decisions, not dashboards.
- The canonical dataset (`data/raw/supply_chain_orders.csv`, ~12k rows, seed 42): walk the schema in ARCHITECTURE.md §3.1; flag `shipping_date` / `actual_shipping_days` as "we'll come back to these" (don't spoil leakage yet).
- Target definition: `late_delivery = actual_shipping_days > scheduled_shipping_days`; overall late rate ≈ 30% — plant the imbalance seed for D3.
- Why EDA functions return DataFrames/dicts (not prints): in Week 4 the `data_analyst` agent calls these exact functions as tools — the API surface *is* the lesson.
- Slicing instinct: late rate by `shipping_mode`, `market`, `carrier_name`, month; tight schedules (Same Day = 1 day) fail more.
- Generators as a teaching device: synthetic but with planted signal, so models genuinely learn — and `DATACO_COLUMN_MAP` swaps in the real Kaggle file later.

**Lab:** `labs/lab01_eda.py` — ~3 h.
*Success looks like:* trainee reproduces late rate by mode/market/carrier and the monthly trend with `late_rate_by`, `class_balance`, `monthly_trend`, `summary_stats`; saves the three plots to `labs/outputs/`; can state the worst mode and worst carrier from their own output.

**Afternoon checkpoint (oral, 3 questions)**
1. *What is the overall late rate, and why should that number change how we evaluate models later?*
   <details><summary>Expected answer</summary>About 30% late. With imbalance, accuracy flatters useless models (predicting "on time" for everything scores ~70%) — we'll need recall/precision and a tuned threshold instead.</details>
2. *Which shipping modes are worst and why does that make business sense?*
   <details><summary>Expected answer</summary>Same Day and First Class — the tightest schedules (1 and 2 days) have the least slack, so any hiccup makes them late. Signal also concentrates in Africa/LATAM markets, Q4, high quantities, and SwiftShip as carrier.</details>
3. *Why do `eda.py` functions return DataFrames/dicts instead of printing nice output?*
   <details><summary>Expected answer</summary>They're contracted functions that the `data_analyst` agent will call as tools in Week 4; tools need structured, serializable returns the agent layer can format — printing would make them human-only.</details>

**Stretch:** add a `late_rate_by(df, "order_country")` drill-down and find one country whose late rate contradicts its market average; explain why small `n_orders` makes it untrustworthy.

---

## W3·D2 — The Leakage Trap (preprocessing & honest baseline)

**Morning concept block (60–90 min)**
- The fastest way to lose a client: demo 99% accuracy, collapse in production. Name the crime — **data leakage**: training on columns unavailable at prediction time.
- Our planted trap: `shipping_date` and `actual_shipping_days` are known only *after* delivery; the target is literally derived from `actual_shipping_days`. `load_orders(drop_leakage=True)` is the antidote.
- Feature engineering that survives prediction time: `engineer_features` adds `order_month`, `order_weekday`, `is_weekend`, `is_q4`, `quantity_bucket`, `mode_schedule_tightness` — all derivable at order time.
- Honest evaluation scaffolding: stratified `split_data` (preserves the 30/70 ratio), `build_preprocessor` (OneHotEncoder with `handle_unknown="ignore"` + StandardScaler), Pipeline so preprocessing is fit on train only.
- Baseline philosophy: a `class_weight="balanced"` logistic regression first — every later model must beat it to earn complexity.

**Lab:** `labs/lab02_preprocessing_baseline.py` — ~3 h.
*Success looks like:* trainee first trains the *leaky* model and watches it score near-perfect, then drops leakage and lands a believable logreg baseline; can articulate the gap; pipeline saved via `train_model("logreg")` → `models/logreg.joblib`.

**Afternoon checkpoint**
1. *Why exactly are `shipping_date` and `actual_shipping_days` leakage?*
   <details><summary>Expected answer</summary>Both are only known after the shipment completes — at prediction time (order placed) they don't exist. Worse, the target is computed from `actual_shipping_days`, so including it lets the model read the answer key.</details>
2. *Why stratify the train/test split?*
   <details><summary>Expected answer</summary>With a ~30% positive class, a random split can skew class ratios between train and test; stratifying on the target keeps both representative so metrics are comparable.</details>
3. *Why `OneHotEncoder(handle_unknown="ignore")`?*
   <details><summary>Expected answer</summary>Unseen categories at inference (new country, new product) would otherwise crash the pipeline; "ignore" encodes them as all-zeros so the model degrades gracefully — production thinking.</details>
4. *Why wrap preprocessing and model in one sklearn Pipeline?*
   <details><summary>Expected answer</summary>The preprocessor is fit on training data only (no test contamination), and the whole thing serializes as one joblib artifact — exactly what `train_model` persists and what the agent tools reload.</details>

**Stretch:** prove the leak quantitatively — train logreg twice (with/without leakage columns) and report the ROC-AUC delta.

---

## W3·D3 — Forging the Blade (imbalance, boosting, thresholds, drivers)

**Morning concept block (60–90 min)**
- A model is not a deliverable; a **decision rule** is. The ops team acts on "flag this order", not on a probability.
- Metrics for 30% positives: recall = late shipments caught; precision = false alarms the team tolerates; PR-AUC vs ROC-AUC; the confusion matrix as a cost conversation.
- `MODEL_REGISTRY`: `logreg`, `random_forest`, `hist_gb` (sklearn's `HistGradientBoostingClassifier` — the LightGBM-equivalent), all `class_weight="balanced"` where supported; `xgboost` joins automatically if installed (guarded import — it is *not* required).
- Threshold tuning as the business lever: `tune_threshold(model_name, min_precision=0.5)` sweeps thresholds and returns the best-recall point subject to a precision floor — let trainees argue about the right floor.
- Explainability without shap: `feature_drivers` uses permutation importance on the held-out split; shap is an optional guarded upgrade.

**Lab:** `labs/lab03_imbalance_boosting_threshold.py` — ~3.5 h.
*Success looks like:* trainee compares the three registry models on recall/PR-AUC, tunes the `hist_gb` threshold under a chosen precision floor, and turns `feature_drivers(top_k=10)` into three plain-English sentences a VP would accept.

**Afternoon checkpoint**
1. *Your model is 78% accurate. Is that good?*
   <details><summary>Expected answer</summary>Unanswerable as asked — with 70% on-time, a do-nothing classifier gets ~70%. Need recall/precision (or PR-AUC) on the late class, plus the threshold and cost context.</details>
2. *What does `tune_threshold(min_precision=0.5)` actually optimize?*
   <details><summary>Expected answer</summary>It sweeps candidate thresholds and returns the one maximizing recall among those whose precision stays ≥ 0.5 — maximize catches subject to an acceptable false-alarm rate.</details>
3. *Why `hist_gb` rather than xgboost in the core curriculum?*
   <details><summary>Expected answer</summary>`HistGradientBoostingClassifier` is sklearn-native (no extra wheel, works on every classroom Python including 3.14) and is the same algorithm family; xgboost is a guarded optional extra in `MODEL_REGISTRY`.</details>
4. *How do `evaluate_model` and `feature_drivers` know which test split to use?*
   <details><summary>Expected answer</summary>`train_model` persists the fitted pipeline AND the split indices in one joblib dict (`{"pipeline": ..., "split": {...}}`) under `models/`; the others reload it — reproducible evaluation, no re-splitting.</details>

**Stretch:** install `imbalanced-learn` (optional) and compare SMOTE against `class_weight="balanced"`; or plot the full precision-recall trade-off curve and mark the chosen threshold.

---

## W3·D4 — First Sparks (LangGraph fundamentals)

**Morning concept block (60–90 min)**
- Demystify: a LangGraph app is a **state machine** — a typed dict passed between plain Python functions, edges deciding who runs next. No magic.
- Read `src/state.py` together (it's 60 lines and it's the whole contract): `SupplyChainState`, the `add_messages` **reducer** (nodes *append* to `messages` rather than overwrite — that's how four agents share one conversation), `next_agent` as the routing slot, `WORKERS` / `ROUTE_OPTIONS`.
- Graph anatomy: `StateGraph` → `add_node` → `add_edge` / `add_conditional_edges` → `compile()` → `invoke`/`stream`.
- Tools: `@tool` from `langchain_core.tools` turns a typed, docstringed function into something an LLM can call; our tools return **strings** (markdown tables / JSON), never DataFrames — the docstring is the agent's manual.
- ReAct in one slide: the agent loops *reason → pick tool → observe → reason…* until it answers; LangChain v1's `create_agent(model, tools, system_prompt=...)` returns a compiled graph invoked with `{"messages": [...]}`.
- Offline note: the toy graph and tool inspection run with **no LLM**; only the final ReAct cell needs a key.

**Lab:** `labs/lab04_langgraph_fundamentals.py` — ~3 h.
*Success looks like:* trainee builds and runs the toy 3-node graph fully offline, explains its state transitions, invokes a project `@tool` directly (e.g. `analyze_late_rate`), and — with a key — watches a single ReAct agent choose tools unprompted.

**Afternoon checkpoint**
1. *What does the `add_messages` reducer do, and what would break without it?*
   <details><summary>Expected answer</summary>It merges each node's returned messages into the running list (append semantics) instead of replacing the field. Without it, every node would overwrite history and agents would lose all context from previous hops.</details>
2. *What's the difference between `add_edge` and `add_conditional_edges`?*
   <details><summary>Expected answer</summary>`add_edge` is unconditional (A always → B); `add_conditional_edges` reads the state (here `next_agent`) through a routing function and picks the destination at runtime — that's the supervisor's dispatch mechanism.</details>
3. *Why must tools return strings rather than DataFrames?*
   <details><summary>Expected answer</summary>Tool outputs become message content the LLM reads; it consumes text, not Python objects. Markdown tables / JSON dumps keep results model-readable and serializable (per ARCHITECTURE.md §4.6).</details>
4. *When does a ReAct agent stop looping?*
   <details><summary>Expected answer</summary>When the model responds without a tool call (a final answer) — or when a recursion/iteration limit cuts it off.</details>

**Stretch:** add a fourth node to the toy graph with a conditional edge that short-circuits to END when a condition in the state is met.

---

## W3·D5 — The Supervisor & The Agent Forge Trial

**Morning concept block (60 min)**
- Why not one agent with 20 tools? Confusion, cost, untestability. The **supervisor pattern**: small specialists (few tools, sharp prompts) + a router deciding who acts next and when to stop.
- Routing as structured output: `class Router(BaseModel): next: Literal[...ROUTE_OPTIONS...]; reason: str` + `llm.with_structured_output(Router)` — the route is a validated enum, never free text to parse.
- The hub-and-spoke rhythm: supervisor → worker → supervisor → … → `FINISH`; workers' answers come back as `AIMessage(name=worker_name)` so the transcript shows who said what.
- Lab 05 builds a two-worker (data_analyst + ml_engineer) version *by hand* — the exact wiring of `src/graph.py` minus two workers.

**Lab:** `labs/lab05_supervisor_two_agents.py` — ~2 h (leave the afternoon for the trial).
*Success looks like:* the two-worker graph answers an EDA question and an ML question, the trainee can trace `[supervisor → worker]` hops, and the offline cells (graph wiring with injected fakes) run without a key.

**🏆 THE AGENT FORGE TRIAL — `exercises/week3_starter.py` · 300 XP**
Afternoon assessment (~2.5 h, individual). Trainees complete the starter's TODOs spanning
the week: leakage-safe loading, pipeline training, threshold tuning, tool definition,
and graph wiring. Trainer marks against `exercises/week3_solutions_TRAINER_ONLY.py`.
**Quiz:** `exercises/week3_quiz.py` (~20 min, closed-book) — imbalance metrics, leakage,
reducers, ReAct, supervisor routing.

**Exit checkpoint**
1. *How does the supervisor "decide", mechanically?*
   <details><summary>Expected answer</summary>An LLM call with structured output bound to the `Router` Pydantic model — `next` is constrained to a `Literal` over `ROUTE_OPTIONS` (the four workers + FINISH), `reason` documents the choice; the value lands in `state["next_agent"]` and the conditional edge dispatches on it.</details>
2. *Why does every worker edge back to the supervisor instead of to END?*
   <details><summary>Expected answer</summary>Multi-hop questions need re-routing: after a worker reports, the supervisor reads the updated conversation and either dispatches another worker or emits FINISH. Worker→END would hardcode single-hop behavior.</details>
3. *Why does `build_graph` accept injectable `llm` and `workers` arguments?*
   <details><summary>Expected answer</summary>Dependency injection for offline testability: tests pass a fake structured-output LLM and plain-callable workers so the real compiled graph runs without keys (ARCHITECTURE.md §4.7, §6).</details>

**Stretch:** make the supervisor's `reason` field print on every hop and observe how prompt wording changes routing quality.

---

# WEEK 4 — The Orchestrator's Citadel

*The blade is forged. Now raise the walls: contracts intelligence, a talking warehouse, and a citadel that commands them all.*

---

## W4·D1 — Mapping the Archives (vectorless RAG: PageIndex)

**Morning concept block (60–90 min)**
- The client question that opens every RAG engagement: *"Why did the bot quote the wrong contract?"* Chunk-embed-cosine loses what contracts have most: **structure**.
- The PageIndex idea: build a hierarchical **tree index** per document (a machine-readable table of contents: node ids, titles, summaries), then have an LLM *reason over the tree* to navigate to sections — like a human expert flipping to "§6 Late Delivery Penalties". No vector DB, no embeddings.
- Tour the corpus: 6 contracts in `data/contracts/` (4 carrier MSAs + 2 supplier procurement agreements), each with distinct, quotable numbers — SwiftShip 2%/late day capped 15%, Atlas flat $250 per >48 h late shipment — so retrieval is *verifiable*.
- The contract surface: `build_index()` parses markdown headings into `Node` trees and persists `data/contracts_index.json`; `PageIndexRetriever.search(query, top_k)` returns `RetrievedSection`s with doc + section path + score; `outline()` prints the full ToC.
- Offline duality (a design lesson, not a workaround): online = LLM selects node ids via structured output; offline = keyword/token-overlap scoring over title+summary+text. Same interface, swappable brain.
- Auditability as the selling point: a reasoning trace of *why* each section was chosen is what legal/compliance stakeholders demand.

**Lab:** `labs/lab06_pageindex_build.py` — ~3 h, **fully offline-capable**.
*Success looks like:* trainee builds the index, prints the outline, runs lexical retrieval for "late delivery penalty SwiftShip" and gets §6 of the SwiftShip MSA top-ranked; can sketch the tree for one contract from memory.

**Afternoon checkpoint**
1. *Contrast PageIndex retrieval with chunk-embed-cosine. When does each win?*
   <details><summary>Expected answer</summary>PageIndex: preserves document structure, navigates by reasoning, gives an auditable section path, no vector infra — shines on long structured docs (contracts, policies, filings). Embeddings: better for unstructured corpora, fuzzy semantic matches, and very large collections where tree-reasoning per query is too slow/costly.</details>
2. *What exactly is stored in `data/contracts_index.json`?*
   <details><summary>Expected answer</summary>Per document: title and a nested tree of Node dicts (node_id, title, level, doc, summary, text, children) mirroring the markdown heading hierarchy — shape `{"docs": [{"doc", "title", "nodes": [...]}]}`.</details>
3. *How do node summaries differ online vs offline, and why is that acceptable?*
   <details><summary>Expected answer</summary>Online they're LLM-generated; offline they're heading + first 240 chars. Acceptable because summaries only guide navigation — the returned section text is identical either way.</details>

**Stretch:** write a 7th mini-contract with deliberately odd heading levels, rebuild the index, and verify the tree handles it.

---

## W4·D2 — The Contracts Analyst Takes the Stand (grounded RAG agent)

**Morning concept block (60–90 min)**
- Lab 06 built retrieval; today builds the **agent** — and the commercial jump: clients don't buy "top-3 sections", they buy *answers*, and the moment an LLM phrases an answer, hallucination risk appears.
- The defense is a **grounding-and-citation contract** in the system prompt: every claim from a retrieved section; every number carries document + section path; "the contracts do not cover this" is a first-class answer.
- Tools of the worker: `search_contracts(query)` and `get_contract_outline()` from `src/agents/tools_rag.py` — two tools, sharp prompt, that's a specialist.
- Test the negative case deliberately: ask something the corpus genuinely doesn't cover and verify the agent declines instead of inventing.
- Tie-back: this is the `contracts_analyst` from `WORKER_DESCRIPTIONS` — the same agent the supervisor will command on D4.

**Lab:** `labs/lab07_contracts_rag_agent.py` — ~3 h (tool cells offline; agent cells need a key).
*Success looks like:* the agent answers "What penalty does Atlas Freight specify?" with the flat-$250 schedule *and* a doc+section citation; and on an uncovered question it answers "not covered" rather than hallucinating.

**Afternoon checkpoint**
1. *What are the three clauses of the grounding contract?*
   <details><summary>Expected answer</summary>(1) Every claim must come from a retrieved section; (2) every number must carry its document + section path citation; (3) "the contracts do not cover this" is a legitimate, expected answer when retrieval comes back empty/irrelevant.</details>
2. *Why is the "I don't know" test as important as the happy path?*
   <details><summary>Expected answer</summary>Hallucinated contract terms are a liability event, not just a wrong answer. An agent that fabricates a penalty clause is worse than no agent — the negative test proves the grounding prompt actually binds.</details>
3. *Why give this worker only two tools?*
   <details><summary>Expected answer</summary>Small tool sets + sharp prompts are the point of the supervisor pattern: less confusion, cheaper loops, easier testing. Breadth comes from the team, not the individual.</details>

**Stretch:** add a "quote verbatim" instruction and compare answer fidelity; or have the agent cross-compare SwiftShip vs NordHaul penalty structures in one answer.

---

## W4·D3 — Speaking to the Warehouse (NL2SQL + metadata catalog)

**Morning concept block (60–90 min)**
- "Just let the LLM write SQL" is how demos die. Three missing pieces, all built today.
- **(1) Metadata grounding:** the model can't guess your column names or what "late" means *in this business*. `data/metadata_catalog.json` is an OpenMetadata-style catalog (table/column descriptions, tags, sample queries, business glossary: "Late Delivery" = `actual_shipping_days > scheduled_shipping_days`). `MetadataCatalog` in `src/metadata/catalog.py` serves it prompt-ready; in production the same interface points at a real OpenMetadata server (`OPENMETADATA_HOST_PORT` + JWT — see the docstring's commented sketch).
- **(2) Guardrails** in `run_sql_query`: single statement only; SELECT-only (reject INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA); auto-append `LIMIT 50`; errors returned as `"SQL_ERROR: <msg>"` strings.
- **(3) Self-repair:** because errors are strings (not exceptions), the agent *sees* them and retries — the `sql_analyst` prompt encodes the loop: inspect schema → draft SQL → run → on SQL_ERROR repair (max 2 retries) → summarize rows in plain English.
- Warehouse tour: `data/warehouse.db` star-ish schema — `orders` fact + `customers`/`products`/`carriers` dims (note `carriers.contract_file` quietly links the SQL world to the RAG world).
- Security foreshadow: these guardrails are exactly what Week 7 attacks.

**Lab:** `labs/lab08_nl2sql_agent.py` — ~3.5 h (guardrails + catalog cells fully offline).
*Success looks like:* trainee demonstrates the guardrails rejecting `DROP TABLE orders` while a join over carriers/orders runs; with a key, the agent answers "late rate per carrier" with correct SQL after consulting the schema — and visibly self-repairs a planted error.

**Afternoon checkpoint**
1. *List the guardrails in `run_sql_query` and the attack each blocks.*
   <details><summary>Expected answer</summary>Single statement (blocks piggybacked `; DROP ...`); SELECT-only rejecting INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA (blocks mutation/exfiltration/config tampering); auto `LIMIT 50` (blocks runaway result dumps); SQL_ERROR-as-string (prevents crashes becoming denial of service, enables repair).</details>
2. *Why return `SQL_ERROR: ...` as a string instead of raising an exception?*
   <details><summary>Expected answer</summary>An exception kills the agent loop; a string becomes a tool observation the LLM reads, so it can diagnose and rewrite the query — the self-repair loop (max 2 retries) depends on it.</details>
3. *What does the metadata catalog add over just dumping `CREATE TABLE` statements into the prompt?*
   <details><summary>Expected answer</summary>Business semantics: column descriptions, tags, sample queries, and a glossary mapping terms like "Late Delivery" and "On-Time SLA" to concrete columns. DDL says what exists; the catalog says what it *means* — that's what makes NL2SQL answer business questions correctly.</details>
4. *How would you point this at a real OpenMetadata server?*
   <details><summary>Expected answer</summary>Same `MetadataCatalog` public interface, backed by the `openmetadata-ingestion` SDK with `OPENMETADATA_HOST_PORT` + JWT token (documented as a commented sketch in `catalog.py`); the JSON mirrors OpenMetadata's Table/Glossary entities.</details>

**Stretch:** try to defeat your own guardrails (comment tricks, CTEs wrapping a delete, `select` in mixed case) and patch any hole you find — bring findings to Week 7.

---

## W4·D4 — Raising the Citadel (full integration & flagship demo)

**Morning concept block (60–90 min)**
- Assembly day: labs 01–08 built inventory; **orchestration is the product**.
- Walk `src/graph.py: build_graph()` line by line: `StateGraph(SupplyChainState)`, supervisor + 4 worker nodes, conditional edges on `state["next_agent"]`, every worker → supervisor, entry = supervisor. Show `ascii_graph()`.
- Walk `main.py`'s streaming loop: with `stream_mode="values"` each yield is the full state after a node — message-list growth distinguishes a supervisor decision (`[supervisor -> sql_analyst]`) from a worker answer. Checkpointing: `MemorySaver` + `thread_id` gives each conversation persistent state; `recursion_limit=25` is the runaway-loop fuse.
- The flagship question dissected *before* running it: risk score → `ml_engineer` (`predict_order_risk`); penalty → `contracts_analyst` (SwiftShip §6: 2%/late day, cap 15%); last-quarter late rate → `sql_analyst` (orders ⋈ carriers). Three systems, one sentence.
- Dependency injection recap: tests run this exact graph offline with fake LLMs — that's why the architecture is testable at all.

**Lab:** `labs/lab09_full_system_integration.py` — ~3.5 h (wiring/injection cells offline; live runs need a key).
*Success looks like:* `python main.py --demo` completes all 5 scenarios; the trainee narrates the flagship run hop by hop and verifies each agent's contribution against ground truth (the contract file, the warehouse, the model).

**Afternoon checkpoint**
1. *Trace the flagship question: which workers, which tools, what order?*
   <details><summary>Expected answer</summary>Supervisor → `ml_engineer` (`predict_order_risk` for Same Day/Western Europe → P(late) + risk band) → supervisor → `contracts_analyst` (`search_contracts` → SwiftShip MSA §6: 2% of shipment value per late business day, capped at 15%) → supervisor → `sql_analyst` (schema lookup, then SELECT over orders joined to carriers filtered to last quarter) → supervisor → FINISH with synthesis. (Order of the middle hops may vary — the supervisor decides.)</details>
2. *What do checkpointer + `thread_id` buy us?*
   <details><summary>Expected answer</summary>Persistent per-conversation state: the graph can resume a thread with full message history (multi-turn memory), and distinct `thread_id`s isolate conversations — `main.py` uses a fresh thread per demo scenario.</details>
3. *Why does `--demo` print `[supervisor -> worker]` hops instead of just the final answer?*
   <details><summary>Expected answer</summary>Observability is the demo: clients (and debuggers) need to see routing decisions and which specialist produced which fact. Mechanically it falls out of streaming full state values and watching the message list grow.</details>

**Stretch:** add a 6th demo scenario requiring a worker *pair* that no canned scenario uses together (e.g. contracts_analyst + data_analyst) and check the supervisor finds the route.

---

## W4·D5 — The Orchestrator's Trial & the Client Briefing

**Morning (~2.5 h): 🏆 THE ORCHESTRATOR'S TRIAL — `exercises/week4_starter.py` · 300 XP**
Individual assessment spanning Week 4: PageIndex retrieval, grounding prompts, SQL
guardrail reasoning, and graph wiring with dependency injection (offline-gradable).
Trainer marks against `exercises/week4_solutions_TRAINER_ONLY.py`.
**Quiz:** `exercises/week4_quiz.py` (~20 min) — vectorless RAG, metadata grounding,
self-repair, checkpointing, supervisor mechanics.

**Afternoon: Capstone presentation — the 10-minute client briefing**
Each trainee (or pair) briefs "the client" (trainer + peers) as if closing a discovery
sprint: the problem, why this architecture, a live demo, and what it takes to harden
for production. **100 points, 4 dimensions × 25:**

| Dimension (25 pts each) | 25 = excellent | 15 = adequate | 5 = weak |
|---|---|---|---|
| **Problem framing & business value** | Quantifies the late-shipment pain, ties every capability to a decision the ops team makes | States the problem, generic value claims | Recites features with no business anchor |
| **Architecture rationale** | Defends supervisor-vs-single-agent, vectorless-vs-embeddings, catalog-grounded NL2SQL — with trade-offs and when *not* to use each | Describes the architecture correctly but can't argue alternatives | Cannot explain why the pieces exist |
| **Live demo execution** | Flagship question runs; narrates each hop; recovers gracefully from any hiccup (offline fallback ready) | Demo runs but narration is thin | Demo fails with no fallback plan |
| **Hardening roadmap** | Concrete next steps: HITL approval before SQL, persistence, tracing, eval suite, security review — prioritized and costed in effort terms | Lists generic improvements | "Add more features" |

> **Hardening prep — read before the briefing.** [CHALLENGES_GUIDE.md](CHALLENGES_GUIDE.md)
> is the production overlay for this dimension: it maps every Week 1–2 whiteboarding
> challenge (agent loops, hallucinations, prompt/SQL injection, cost, drift, governance,
> observability, evaluation, memory) to the exact mechanism in this repo and the named
> production upgrade (Langfuse/OTel, RBAC, PII masking, semantic caching, HITL). Each
> challenge ships a "whiteboard moment" talking-point script and a one-line "Try it" —
> arm trainees with it so the hardening roadmap is concrete and costed, not generic.

**Exit checkpoint (program hand-off)**
1. *Which Week 7 attack surface did you ship this week?*
   <details><summary>Expected answer</summary>The tool layer: `run_sql_query` (injection/exfiltration target despite guardrails) and `search_contracts` (retrieved text is untrusted input that can carry prompt injection into the agent's context).</details>
2. *What's the first thing you'd change before a real client pilot?*
   <details><summary>Expected answer</summary>Any defensible pick from the extensions appendix — most common: human-in-the-loop interrupt before `run_sql_query`, durable checkpointing (SqliteSaver), and tracing/evals — with a reason.</details>

---

## Appendix A — Concepts glossary

| Concept | Working definition (as used in this repo) |
|---|---|
| **Supervisor pattern** | A router node (LLM with structured output) that dispatches each turn to one specialist worker — or FINISH — with every worker reporting back to the router. Hub-and-spoke, not pipeline. See `src/agents/supervisor.py`, `src/graph.py`. |
| **Reducer** | A merge function attached to a state field telling LangGraph how node returns combine with existing state. `messages: Annotated[list, add_messages]` appends instead of overwriting — the mechanism behind shared conversation memory. |
| **ReAct agent** | An LLM loop of reason → call tool → observe → repeat, ending when the model answers without a tool call. Built here via `from langchain.agents import create_agent` (LangChain v1), invoked with `{"messages": [...]}`. |
| **Vectorless RAG / PageIndex** | Retrieval without embeddings: parse documents into hierarchical tree indexes (a machine ToC) and let an LLM reason over node titles/summaries to navigate to sections — auditable paths, no vector DB. Offline fallback: lexical scoring. See `src/pageindex/`. |
| **Metadata catalog / OpenMetadata** | A registry of what data *means*: table/column descriptions, tags, sample queries, and a business glossary mapped to assets. Ours is a JSON mirror of OpenMetadata's entities served by `MetadataCatalog`; production points the same interface at a live server. |
| **NL2SQL self-repair** | Tool errors returned as `SQL_ERROR: ...` strings so the agent observes failures and rewrites its query (max 2 retries) instead of crashing — turning exceptions into conversation. |
| **Checkpointing** | Persisting graph state per `thread_id` via a checkpointer (`MemorySaver` here) so conversations are resumable and isolated; the foundation for memory, replay, and HITL. |
| **HITL (human-in-the-loop)** | Pausing the graph at a sensitive node (e.g. before executing SQL) for human approval, via LangGraph interrupts on a checkpointed thread — the production posture for write-adjacent tools. |

## Appendix B — Extensions (fast finishers & post-course)

1. **Real data at scale:** drop the Kaggle DataCo CSV (~180k rows) onto `load_orders` via `DATACO_COLUMN_MAP` (`src/ml_pipeline/data_loader.py`); re-run lab03 and compare drivers against the synthetic world.
2. **Real OpenMetadata:** run the OpenMetadata server via Docker, ingest `warehouse.db`, and back `MetadataCatalog` with the `openmetadata-ingestion` SDK per the commented sketch in `src/metadata/catalog.py` (`OPENMETADATA_HOST_PORT` + JWT).
3. **Tracing:** enable LangSmith and study the supervisor's routing decisions and per-worker token spend across the 5 demo scenarios.
4. **HITL guard:** add a LangGraph interrupt before `run_sql_query` so a human approves every SQL statement — the natural hardening-roadmap item from the capstone.
5. **Durable memory:** swap `MemorySaver` → `SqliteSaver` in `main.py`'s `_build_app()` and demonstrate a conversation surviving a process restart.
6. **Serve it:** wrap `build_graph()` in a FastAPI endpoint (per the Week 3 deployment pattern from the earlier program) — `POST /ask {"question", "thread_id"}` streaming hop events.
