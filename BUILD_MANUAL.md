# BUILD MANUAL — The Full Architecture, Step by Step

**Who this is for:** anyone — including someone who has never written Python — who wants to
build and run a complete multi-agent AI system: agents, tools, MCP servers, an external
partner agent, a quality gate, human review, and an HTTP API, end to end.

**What you need:** a Windows PC, an internet connection for setup, and about a day.
An LLM API key is optional — every chapter has an offline path that works without one.

**The rule of this manual:** every chapter ends with something you can *run* and a
✅ checkpoint that proves it worked. If a checkpoint fails, the ⚠️ box right under it
lists the usual causes. Never move on past a red checkpoint.

---

## The big picture — what you are building

This is the target architecture. The circled numbers are the chapters of this manual:

```
                         User / Frontend          (12)
                               |
                               v
                          API Gateway             (11)
                               |
                               v
                        LangGraph Runtime         (4)
                               |
                               v
                     Supervisor / Planner Node    (4)+(8)
                               |
        ------------------------------------------------
        |                     |                        |
        v                     v                        v
   Internal Agent        Tool Node               External Agent
      (3)                  (2)                       (7)
        |                     |                        |
        |                     v                        |
        |                  MCP Client             (6)  |
        |                     |                        |
        |                     v                        |
        |                MCP Servers              (5)  |
        |            SQL / SAP / ServiceNow            |
        |                                              |
        |---------------- A2A ------------------- (7)--|
                               |
                               v
                     Validator / Judge Node       (9)
                               |
                    ------------------------
                    |                      |
                    v                      v
              Auto Complete          Human Review (10)
                    |                      |
                    v                      v
                   End              Resume Graph  (10)
```

**Why the chapters run inside-out instead of top-down:** every chapter ends with something
runnable, because each layer's dependencies were built in an earlier chapter. Top-down
would force you to stub layers that don't exist yet and stare at error messages for half
the book. The diagram above is your map; every chapter starts with a "You are here".

**The story the system tells:** a logistics company (GlobalTrade) ships orders through
carriers (SwiftShip Express, Atlas Freight, …). Shipments arrive late; contracts contain
penalty clauses; the warehouse holds 12,000 historical orders. The agent team answers
questions like *"Order 104872 ships Same Day via SwiftShip — how likely is it late, what
penalty does the contract say, and how did SwiftShip do last quarter?"*

---

## Chapter 0 — Setup (30 min)

> **You are here:** before the diagram — getting a machine that can run it.

### 0.1 Install Python

You need Python **3.11 to 3.14**. Check what you have — open **PowerShell** (press the
Windows key, type `powershell`, press Enter) and run:

```powershell
python --version
```

**Expected output** (any 3.11/3.12/3.13/3.14 is fine):

```
Python 3.14.3
```

If you get an error or an older version, install from <https://www.python.org/downloads/>
— during installation **tick the "Add python.exe to PATH" checkbox**.

### 0.2 Get the code

```powershell
git clone https://github.com/mahi80/training26052016.git
cd training26052016
```

(No git? Download the ZIP from the GitHub page → "Code" → "Download ZIP", unzip it, then
`cd` into the unzipped folder.)

> ⚠️ **If your folder path contains `&`** (like `...\week3&4`), PowerShell treats `&` as
> a special character. **Always put the path in quotes**: `cd "C:\...\week3&4"`. Every
> command in this manual already assumes you are *inside* the project folder.

### 0.3 Create a virtual environment and install dependencies

A virtual environment is a private box of Python packages for this project — nothing you
install here touches the rest of your PC.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Expected:** the prompt now starts with `(.venv)`, and pip ends with a line like
`Successfully installed ... langgraph-1.x ... mcp-1.x ...` (a few warnings are normal).

> ⚠️ `Activate.ps1 cannot be loaded because running scripts is disabled` → run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again.

### 0.4 Generate the data

The project ships *generators*, not data files — running them builds identical data on
every machine (everything is seeded, so your numbers will match this manual's):

```powershell
python data\generate_supply_chain_data.py
python data\generate_contracts.py
python data\build_database.py
```

**Expected:** each script prints what it wrote — a ~12,000-row CSV, six markdown
contracts, and `data\warehouse.db` + `data\metadata_catalog.json`.

### 0.5 Choose: with or without an API key

The *reasoning* parts (agents deciding things) need an LLM. Everything else runs offline.

**Option A — no key (works for the whole manual):**

```powershell
$env:OFFLINE="1"
```

(This lasts for the current PowerShell window only — rerun it in each new window.)

**Option B — with a key (agents actually reason):**

```powershell
copy .env.example .env
notepad .env
```

In Notepad: set `LLM_PROVIDER` to `azure_openai`, `openai`, or `anthropic` and fill in
the matching API key line. Save and close.

### ✅ Checkpoint 0

```powershell
python main.py --offline-check
```

**Expected output** ends with all checks passing:

```
[PASS] contracts RAG ...
[PASS] SQL tool ...
[PASS] SQL guardrail ...
[PASS] ML pipeline ...
```

> ⚠️ `ModuleNotFoundError` → your venv isn't active (`.\.venv\Scripts\Activate.ps1`).
> ⚠️ A red `UserWarning: Core Pydantic V1 functionality...` on Python 3.14 is **harmless**
> — you will see it on almost every command; ignore it everywhere.

---

# PART I — Tour the engine you already have

The repo already contains a working multi-agent core (the middle of the diagram). Part I
is a guided tour — you run things and read short code excerpts, you don't write anything
yet. Understanding these four ideas makes Part II easy.

---

## Chapter 1 — The State: the team's shared clipboard (15 min)

> **You are here:** `LangGraph Runtime` — the thing that passes state between nodes.

**What is this box?** A LangGraph system is a *state machine*: one dictionary (the
"state") is passed from node to node, and each node returns a small update to it. Think
of a clipboard passed around a meeting room — everyone reads it, everyone appends to it,
nobody starts a private side-conversation.

Open [src/state.py](src/state.py). The entire contract is ~10 lines:

```python
class SupplyChainState(TypedDict):
    """The single state object every node in the graph reads and writes."""

    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
```

Two fields. That's the whole system's memory:

- `messages` — the conversation so far. The `add_messages` annotation is a *reducer*: it
  means node updates are **appended**, never overwritten, so every agent always sees the
  full history.
- `next_agent` — where the supervisor wrote its routing decision.

### ✅ Checkpoint 1

```powershell
python -c "from src.state import WORKERS, ROUTE_OPTIONS; print(WORKERS); print(ROUTE_OPTIONS)"
```

**Expected output:**

```
('data_analyst', 'ml_engineer', 'contracts_analyst', 'sql_analyst')
('data_analyst', 'ml_engineer', 'contracts_analyst', 'sql_analyst', 'FINISH')
```

Four specialists plus FINISH. Remember these names — they are graph node names AND the
labels the supervisor routes by.

---

## Chapter 2 — Tools: the hands (20 min)

> **You are here:** `Tool Node` — what agents use to touch the real world.

**What is this box?** An LLM can only produce text — it cannot open a file or query a
database. A *tool* is a Python function with a descriptive docstring; the agent reads
the docstring (that's literally its only view of your code) and asks for the tool to be
run. Tools are where AI meets reality, so tools are where the **guardrails** live.

Run a tool yourself — no agent, no LLM, just the function:

```powershell
python -c "from src.agents.tools_sql import run_sql_query; print(run_sql_query.invoke({'sql': 'SELECT COUNT(*) AS n_orders FROM orders'}))"
```

**Expected output:**

```
1 row(s):
| n_orders |
| --- |
| 12000 |
```

Now try to do damage:

```powershell
python -c "from src.agents.tools_sql import run_sql_query; print(run_sql_query.invoke({'sql': 'DROP TABLE orders'}))"
```

**Expected output:**

```
SQL_ERROR: statement starts with 'DROP' but this tool is read-only. Only SELECT (or WITH ... SELECT) is allowed.
```

Two things to notice — they repeat through this whole manual:

1. **The guardrail refused, the program didn't crash.** Failures come back as *strings*
   starting with `SQL_ERROR:`. An agent reads that string mid-conversation, fixes its
   SQL, and retries. An exception would have killed the whole run instead.
2. **The tool is safe no matter who calls it** — an agent, you, or (in Chapter 5) a
   protocol client. Guardrails live in the tool, not in the caller.

### ✅ Checkpoint 2

Both commands above behaved as shown: a count, then a refusal — no crash.

---

## Chapter 3 — Workers: the specialists (15 min)

> **You are here:** `Internal Agent` — one brain + a small toolbox + a job description.

**What is this box?** A worker is a *ReAct agent*: an LLM that loops "think → pick a tool
→ read the result → think again" until it can answer. Each worker gets a **small**
toolbox (2–5 tools) and a role prompt that encodes the discipline of the job, not just a
persona. Small toolboxes beat mega-agents — that's the core design choice of this repo.

Open [src/agents/workers.py](src/agents/workers.py). Every worker is a factory with the
same shape:

```python
def make_sql_analyst(model: BaseChatModel | None = None) -> Any:
    """ReAct agent for schema-grounded NL2SQL over the warehouse."""
    from src.agents.tools_sql import (
        get_table_schema, list_warehouse_tables, run_sql_query, search_metadata,
    )
    return create_agent(
        model=model or get_llm(),
        tools=[list_warehouse_tables, get_table_schema, search_metadata, run_sql_query],
        system_prompt=SQL_ANALYST_PROMPT,
    )
```

And look at one line of `SQL_ANALYST_PROMPT` — the *discipline*, in plain English:

> "If run_sql_query returns a string starting with SQL_ERROR, read the message, correct
> your SQL, and retry — at most 2 retries, then report honestly that the query failed."

That's the `SQL_ERROR` string from Chapter 2 closing its loop: tool guardrail + prompt
discipline = an agent that self-repairs instead of crashing or lying.

### ✅ Checkpoint 3

```powershell
python -c "from src.agents.workers import make_data_analyst, make_ml_engineer, make_contracts_analyst, make_sql_analyst; print('4 worker factories import OK')"
```

**Expected output:** `4 worker factories import OK`

---

## Chapter 4 — Supervisor + Graph: the boss and the floor plan (30 min)

> **You are here:** `Supervisor / Planner Node` + `LangGraph Runtime` — who decides, and
> the wiring that enforces it.

**What is this box?** With four specialists, someone must decide who acts next. The
supervisor is a *router*: it reads the conversation and picks ONE worker — or FINISH.
Two tricks make it reliable (open [src/agents/supervisor.py](src/agents/supervisor.py)):

**Trick 1 — structured output.** The supervisor cannot answer in prose; it is forced to
return a validated object whose `next` field can ONLY be one of the legal route names:

```python
class Router(BaseModel):
    next: RouteName    # Literal: one of the 4 workers, or "FINISH" — nothing else
    reason: str
```

**Trick 2 — the graph enforces the floor plan.** Open [src/graph.py](src/graph.py):

```python
builder.add_edge(START, "supervisor")
for name, node in workers.items():
    builder.add_node(name, node)
    builder.add_edge(name, "supervisor")          # every worker reports back
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],            # read the routing decision
    {name: name for name in workers} | {"FINISH": END},
)
```

Hub and spoke: workers never talk to each other; everything flows through the supervisor
and the shared state. See it run:

**Offline** (proves the wiring with a scripted fake supervisor — this is exactly what the
test suite does):

```powershell
python -m pytest tests/test_graph_routing.py -q
```

**Expected:** `6 passed`.

**With a key** (real agents, real routing — the flagship demo):

```powershell
python main.py --demo
```

**Expected (shape, not exact text):**

```
=== Scenario 5/5 — Multi-agent ===
Q: Order 104872 ships Same Day via SwiftShip Express ...
  [supervisor -> ml_engineer]
  [ml_engineer] answered (812 chars)
  [supervisor -> contracts_analyst]
  [contracts_analyst] answered (645 chars)
  [supervisor -> sql_analyst]
  [sql_analyst] answered (590 chars)
  [supervisor -> FINISH]
Final answer: ...
```

### ✅ Checkpoint 4

The routing tests pass (offline), or the demo routes through multiple workers (online).

> ⚠️ `--demo` errors with "No LLM available" → that's Option A (offline) working as
> designed; the pytest line above is your offline proof instead.

**End of the tour.** You now have the middle of the diagram. Everything from here on you
build — and notice what's still missing: tools are *Python imports* (no protocol), there
is no external agent, nothing checks answers before they ship, and there's no API. Part
II fixes each, one chapter per box.

---

# PART II — Build the new layers

> **A note on "build":** every file in Part II already exists in this repo, fully
> working, so you can't get stuck. The chapter shows you the file, explains every moving
> part, and you run it. Want the full builder experience? Rename the repo's copy (e.g.
> `sap_server.py` → `sap_server.py.bak`), create the file yourself from the listings,
> and compare afterwards. Either way, **run every command** — the checkpoints are the
> point.

---

## Chapter 5 — MCP Servers: giving company systems a phone number (40 min)

> **You are here:** `MCP Servers — SQL / SAP / ServiceNow` (bottom middle of the diagram).

**What is this box?** Your agents' tools are currently hard-wired Python imports — like
a colleague who can only help if they sit at your desk. **MCP (Model Context Protocol)**
gives each company system its own *phone number*: a small server program that answers a
standard set of questions over a standard wire format:

- "What tools do you have?" → a list of names + descriptions
- "Run tool X with arguments Y" → the result, as text

Any agent that speaks MCP can call any MCP server — yours, a vendor's, anyone's. You
build three: a mock **SAP** (purchase orders), a mock **ServiceNow** (incident tickets),
and a **SQL** server with a twist you'll see in 5.3.

### 5.1 The SAP server

Open [src/mcp_servers/sap_server.py](src/mcp_servers/sap_server.py). The anatomy (90
lines total) is three plain functions + four registration lines:

```python
PURCHASE_ORDERS: list[dict] = [
    {"po_number": "PO-2025-0041", "vendor": "SwiftShip Express",
     "material": "Same Day freight lane — Western Europe", "quantity": 120,
     "value_usd": 48500.00, "promised_date": "2025-11-14", "status": "Delayed"},
    # ... 9 more rows; SwiftShip has the most "Delayed" — consistent with the data story
]

def get_purchase_orders(vendor: str | None = None, status: str | None = None) -> str:
    """List purchase orders, optionally filtered by vendor and/or status. ..."""
    # filter PURCHASE_ORDERS, return a markdown table (or a helpful no-match message)

mcp = FastMCP("mock-sap")
mcp.tool()(get_purchase_orders)     # docstring becomes the tool description
mcp.tool()(get_po)
mcp.tool()(list_vendors)

if __name__ == "__main__":
    mcp.run()                       # serve over stdio — stdout belongs to the protocol
```

The one design decision that matters: **the tools are plain functions first**, registered
on the server afterwards. So you can call them without any protocol at all:

```powershell
python -c "from src.mcp_servers.sap_server import get_purchase_orders; print(get_purchase_orders('SwiftShip', 'Delayed'))"
```

**Expected output:**

```
3 purchase order(s):
| po_number | vendor | material | quantity | value_usd | promised_date | status |
| --- | --- | --- | --- | --- | --- | --- |
| PO-2025-0041 | SwiftShip Express | Same Day freight lane — Western Europe | 120 | 48500.0 | 2025-11-14 | Delayed |
| PO-2025-0057 | SwiftShip Express | First Class freight lane — LATAM | 80 | 31200.0 | 2025-12-02 | Delayed |
| PO-2026-0003 | SwiftShip Express | Same Day freight lane — USCA | 150 | 61750.0 | 2026-01-20 | Delayed |
```

### 5.2 The ServiceNow server

[src/mcp_servers/servicenow_server.py](src/mcp_servers/servicenow_server.py) is the same
pattern over incident tickets (`INC0010001`…): `get_incidents(carrier, state)` and
`get_incident(number)` over a `FastMCP("mock-servicenow")`. SwiftShip has a P1 ticket
open — "Missed Same Day pickup window — Western Europe hub". The story is consistent
across every system on purpose: that's what lets a multi-agent answer *converge*.

### 5.3 The SQL server — the punchline

Open [src/mcp_servers/sql_server.py](src/mcp_servers/sql_server.py) — it's ~40 lines and
contains **zero query logic**:

```python
def query_warehouse(sql: str) -> str:
    """Run one read-only SELECT against the supply-chain SQLite warehouse. ..."""
    from src.agents.tools_sql import run_sql_query
    return run_sql_query.invoke({"sql": sql})    # the EXACT tool from Chapter 2

mcp = FastMCP("supply-chain-sql")
mcp.tool()(query_warehouse)
```

It delegates to the *exact same guard-railed function* you ran in Chapter 2. The
`DROP TABLE` refusal, the LIMIT 50, the `SQL_ERROR:` strings — all inherited for free.
**That is the whole point of MCP:** you don't rewrite capabilities per consumer, you put
a standard wire format in front of the ones you already trust.

### ✅ Checkpoint 5

```powershell
python -m pytest tests/test_mcp_servers.py -q
```

**Expected:** `14 passed` — table formats, filters, the inherited DROP guardrail, and
one genuine protocol round-trip.

> ⚠️ `ModuleNotFoundError: No module named 'mcp'` → rerun `pip install -r requirements.txt`
> inside the venv.
> ⚠️ `ModuleNotFoundError: No module named 'src'` → you're not in the project folder
> (`cd "<project path>"` — quotes if it contains `&`).

---

## Chapter 6 — MCP Client: teaching the team to dial (30 min)

> **You are here:** `MCP Client` — between Tool Node and MCP Servers.

**What is this box?** The servers have phone numbers; the client is the dialer. When an
agent tool calls `call_mcp_tool("sap", "get_purchase_orders", {...})`, the client:

1. spawns the server as a subprocess (`python -m src.mcp_servers.sap_server`),
2. performs the MCP handshake and the tool call over stdin/stdout (JSON-RPC),
3. returns the text result and lets the subprocess exit.

Open [src/mcp_servers/client.py](src/mcp_servers/client.py). The registry is the part
you'd edit in real life — swapping a mock for a real vendor server is one line:

```python
SERVERS: dict[str, str] = {
    "sap": "src.mcp_servers.sap_server",
    "servicenow": "src.mcp_servers.servicenow_server",
    "sql": "src.mcp_servers.sql_server",
}
```

And the error contract (read the module docstring for the full design notes): the client
**never raises** — unknown server, spawn failure, timeout, tool error all come back as
`MCP_ERROR: ...` strings. Same convention as `SQL_ERROR`, one layer further out.

Dial a server — this is a real protocol round-trip:

```powershell
python -c "from src.mcp_servers.client import call_mcp_tool; print(call_mcp_tool('servicenow', 'get_incidents', {'carrier': 'SwiftShip', 'state': 'In Progress'}))"
```

**Expected output** (the `INFO Processing request...` lines are the server's own logging
— normal):

```
2 incident(s):
| number | carrier | short_description | priority | state | opened_at |
| --- | --- | --- | --- | --- | --- | 
| INC0010001 | SwiftShip Express | Missed Same Day pickup window — Western Europe hub | P1 | In Progress | 2026-06-08 |
| INC0010002 | SwiftShip Express | Tracking feed outage: no scan events for 6 hours | P2 | In Progress | 2026-06-09 |
```

Now the agent-facing side: [src/agents/tools_mcp.py](src/agents/tools_mcp.py) wraps the
dialer in three `@tool` functions (`sap_purchase_orders`, `servicenow_incidents`,
`warehouse_query_via_mcp`) — each a one-liner around `call_mcp_tool`. To a worker agent,
"call SAP over a protocol" looks exactly like "query the warehouse": just another tool.

### ✅ Checkpoint 6

```powershell
python -c "from src.mcp_servers.client import call_mcp_tool; print(call_mcp_tool('jira', 'anything'))"
```

**Expected output** (failure as a string — no crash, and it names the known servers):

```
MCP_ERROR: unknown server 'jira'. Known servers: sap, servicenow, sql.
```

> ⚠️ The round-trip takes 1–2 seconds — that's the subprocess spawning. Production
> keeps sessions open (see the client docstring: `langchain-mcp-adapters`); per-call
> spawning is the zero-session-management teaching trade.

---

## Chapter 7 — The External Agent + A2A: talking to another company (40 min)

> **You are here:** `External Agent` and the `A2A` link (right side of the diagram).

**What is this box?** MCP connects your agent to *systems*. **A2A (agent-to-agent)**
connects your agent to *other agents* — systems with their own brain, owned by someone
else. The difference matters: SwiftShip will never give you database access (MCP), but
they'll happily expose an agent you can *ask questions* (A2A). Each side keeps its own
tools and data; only conversation crosses the boundary.

Open [src/a2a/external_agent.py](src/a2a/external_agent.py) — the "SwiftShip Carrier
Partner Agent", a completely separate web service with two endpoints:

```python
@app.get("/.well-known/agent.json")     # the AGENT CARD: who am I, what can I do
def agent_card() -> dict: ...

@app.post("/a2a")                       # JSON-RPC "message/send": question in, answer out
def a2a_endpoint(request: JsonRpcRequest) -> dict: ...
```

Its "brain" is keyword rules over mock operations data (`CARRIER_OPS`) — deliberately,
so it needs **zero API keys**. The lesson here is the wire, not the wit.

### 7.1 Run it — your first second terminal

Open a **new** PowerShell window (keep your current one), then:

```powershell
cd "<your project folder>"        # quotes! e.g. cd "C:\...\week3&4"
.\.venv\Scripts\Activate.ps1
python -m src.a2a.external_agent
```

**Expected output** (this window is now *busy being SwiftShip* — leave it running):

```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

> ⚠️ Windows Firewall may pop up a dialog → click **Allow**. It's a local-only server.

### 7.2 Talk to it from your first terminal

Discovery first — fetch the agent card:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/.well-known/agent.json | ConvertTo-Json -Depth 5
```

**Expected:** JSON with `"name": "SwiftShip Carrier Partner Agent"` and three skills
(fleet status, pickup capacity, hub congestion).

Now ask it a question, exactly the way our agent will — JSON-RPC `message/send`:

```powershell
$body = '{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"role":"user","parts":[{"text":"Any same-day pickup capacity left in Western Europe?"}]}}}'
Invoke-RestMethod -Method Post http://127.0.0.1:8001/a2a -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 6
```

**Expected:** a `result.message.parts[0].text` mentioning pickup slots ("Western Europe
0 same-day slots (book next-day)…").

### 7.3 Our side of the call

Open [src/agents/tools_a2a.py](src/agents/tools_a2a.py): the `ask_carrier_agent` `@tool`
posts that same JSON-RPC envelope with `httpx` and returns the text. Connection refused?
You get — say it with me — a string: `A2A_ERROR: carrier agent not reachable at ... Start
it with: python -m src.a2a.external_agent`. The error message *teaches the fix*.

```powershell
python -c "from src.agents.tools_a2a import ask_carrier_agent; print(ask_carrier_agent.invoke({'question': 'Is the fleet running late anywhere?'}))"
```

**Expected output** (with the second terminal still running):

```
SwiftShip Partner Agent: Fleet status: 87% of line-haul capacity operating normally. Known disruption: Western Europe Same Day fleet running ~4h behind due to a hub outage in Rotterdam (since 2026-06-08).
```

### ✅ Checkpoint 7

```powershell
python -m pytest tests/test_a2a.py -q
```

**Expected:** `8 passed` (these use an in-process client, so they pass even with the
second terminal closed).

---

## Chapter 8 — The 5th Worker + Supervisor v2: hiring the coordinator (30 min)

> **You are here:** back at `Supervisor / Planner Node` — extending the team.

**What is this box?** You now have live-systems tools (MCP, Chapter 6) and a partner
agent (A2A, Chapter 7) — but no worker owns them, and the supervisor doesn't know such a
worker could exist. This chapter hires the **logistics_coordinator** and teaches the
supervisor to route to it. Adding a worker to a supervisor system is *exactly three
moves* — watch for them:

**Move 1 — extend the state registry.** [src/state_v2.py](src/state_v2.py):

```python
WORKERS_V2: tuple[str, ...] = WORKERS + ("logistics_coordinator",)

class SupplyChainStateV2(SupplyChainState):     # inherits v1 — additive, never rewrite
    verdict: str          # written by Chapter 9's validator
    verdict_reason: str
```

(The two `verdict` fields are Chapter 9's parking spot — defined now so the state schema
changes once, not twice.)

**Move 2 — extend the route Literal + the prompt.**
[src/agents/supervisor_v2.py](src/agents/supervisor_v2.py) is a near-copy of v1's
supervisor with `logistics_coordinator` added to `RouteNameV2` and one new routing rule:

> "Questions about LIVE operations (open POs, current incidents, today's carrier
> capacity) go to logistics_coordinator; historical analysis stays with the others."

That near-copy is deliberate. The v1 files are this course's frozen contract; v2
*extends* — the lock-step guard at the top of the file crashes loudly at import if the
Literal and the registry ever drift apart.

**Move 3 — give the worker its toolbox.** At the bottom of
[src/agents/workers.py](src/agents/workers.py):

```python
def make_logistics_coordinator(model: BaseChatModel | None = None) -> Any:
    from src.agents.tools_a2a import ask_carrier_agent
    from src.agents.tools_mcp import (
        sap_purchase_orders, servicenow_incidents, warehouse_query_via_mcp,
    )
    return create_agent(
        model=model or get_llm(),
        tools=[sap_purchase_orders, servicenow_incidents, warehouse_query_via_mcp, ask_carrier_agent],
        system_prompt=LOGISTICS_COORDINATOR_PROMPT,
    )
```

Every tool in that toolbox crosses a protocol boundary — MCP subprocesses or the
external agent's HTTP endpoint. And its prompt closes the error loop one more time: *"If
a tool returns MCP_ERROR or A2A_ERROR, report that the system is unreachable in plain
language … never invent live data."*

### ✅ Checkpoint 8

```powershell
python -c "from src.state_v2 import WORKERS_V2; from src.agents.supervisor_v2 import RouterV2; from src.agents.workers import make_logistics_coordinator; print(WORKERS_V2)"
```

**Expected output:**

```
('data_analyst', 'ml_engineer', 'contracts_analyst', 'sql_analyst', 'logistics_coordinator')
```

---

## Chapter 9 — The Validator / Judge: quality control before anything ships (30 min)

> **You are here:** `Validator / Judge Node` — the gate below the agents.

**What is this box?** Until now, when the supervisor said FINISH the answer went
straight out the door. Nobody checked it. The validator is one more node between FINISH
and the user that rules on the draft answer:

- **`complete`** → auto-complete; the answer ships.
- **`needs_human`** → a person must look first (Chapter 10).

Open [src/agents/validator.py](src/agents/validator.py). There are *two judges with one
contract*, and choosing between them is the lesson:

```python
class Verdict(BaseModel):
    verdict: Literal["complete", "needs_human"]    # structured output again —
    reason: str                                    # a graph edge can act on this
```

**Judge 1 — the heuristic (offline, deterministic, explainable):**

```python
def heuristic_verdict(question: str, answer: str) -> Verdict:
    # needs_human if: no answer at all, under 40 chars, or a raw tool failure
    # (SQL_ERROR / MCP_ERROR / A2A_ERROR) leaked into the final text
```

**Judge 2 — LLM-as-judge (online):** a structured-output call grading the answer against
the question — smarter, but it costs a call and *can itself be wrong*.

Production systems run **both**: cheap deterministic checks first, model judges on what
passes. And note the factory's offline behavior: no key → it *degrades* to the heuristic
instead of raising, because a quality gate that crashes is worse than a dumb one.

Try the heuristic on three answers:

```powershell
python -c "from src.agents.validator import heuristic_verdict as h; print(h('worst carrier?', 'SwiftShip Express has the highest late rate at 34.2% across 2,841 orders.').verdict); print(h('worst carrier?', 'SwiftShip.').verdict); print(h('status?', 'I checked but got MCP_ERROR: timeout under the hood.').verdict)"
```

**Expected output:**

```
complete
needs_human
needs_human
```

### ✅ Checkpoint 9

```powershell
python -m pytest tests/test_validator.py -q
```

**Expected:** `7 passed`.

---

## Chapter 10 — Human Review: the pause button (40 min)

> **You are here:** `Human Review` → `Resume Graph` — the bottom-right of the diagram.

**What is this box?** When the validator says `needs_human`, the graph must *stop* —
not crash, not poll in a loop, not time out. LangGraph's `interrupt()` is a pause
button: execution freezes mid-node, the checkpointer saves everything under the
conversation's `thread_id`, and the process is free to do other work. Minutes (or days)
later, a `Command(resume=...)` with the same `thread_id` continues the graph from the
exact line it stopped on.

Open [src/graph_v2.py](src/graph_v2.py). The human-review node:

```python
def human_review_node(state: SupplyChainStateV2) -> dict:
    decision = interrupt({                      # graph FREEZES here...
        "question": ..., "draft_answer": ..., "validator_reason": ...,
        "options": "resume with {'action': 'approve'} or {'action': 'edit', 'text': '...'}",
    })
    # ...and on resume, interrupt() RETURNS the human's decision:
    if decision.get("action") == "edit":
        return {"verdict": "complete",
                "messages": [AIMessage(content=decision["text"], name="human_reviewer")]}
    return {"verdict": "complete", "verdict_reason": "Human reviewer approved..."}
```

And the wiring — compare this to Chapter 4's graph; **two edges are the entire diff**:

```python
builder.add_conditional_edges(
    "supervisor", lambda state: state["next_agent"],
    {name: name for name in workers} | {"FINISH": "validator"},   # was: END
)
builder.add_conditional_edges(
    "validator", lambda state: state["verdict"],
    {"complete": END, "needs_human": "human_review"},             # new branch
)
builder.add_edge("human_review", END)
```

In v1, FINISH meant "ship it". In v2, FINISH means "the team is done — now the gate
decides if it ships."

See the pause/resume cycle live — run the lab's HITL section (or the whole lab):

```powershell
python labs\lab10_production_layers.py
```

**Expected output includes:**

```
Graph paused?       True
Paused at node:     ('human_review',)
Payload for human:  sql_analyst: draft answer (unreviewed).
Verdict after resume: complete — Human reviewer approved the draft answer.
Final answer by 'human_reviewer': Reviewed answer: 2,841 orders across 4 carriers.
```

One run paused and approved; a second run paused and the "human" replaced the answer —
tagged `name='human_reviewer'` so the audit trail shows who really wrote it.

### ✅ Checkpoint 10

```powershell
python -m pytest tests/test_graph_v2_hitl.py -q
```

**Expected:** `5 passed` — auto-complete path, pause, resume-approve, resume-edit, and
routing to the new worker, all on the real compiled graph.

> ⚠️ **Interrupts REQUIRE a checkpointer.** `interrupt()` without one raises at runtime
> — there'd be nowhere to save the paused state. If you ever wire your own graph and
> hit that error, you forgot `checkpointer=MemorySaver()`.

---

## Chapter 11 — The API Gateway: the front door (40 min)

> **You are here:** `API Gateway` — second box from the top.

**What is this box?** Everything so far runs inside one Python process you start by
hand. Real users don't run Python — they (or their apps) speak HTTP. The gateway wraps
the v2 graph in four REST endpoints and owns the runtime concerns: building the graph
once, minting `thread_id`s, and translating interrupts into a REST-shaped conversation:

| Endpoint | What it does |
|---|---|
| `GET /health` | "Is the service up?" — never builds the graph, works keyless |
| `POST /ask` | Run a question; returns `complete` (answer) or `needs_human` (review payload) |
| `POST /resume` | Continue a paused thread with `approve` or `edit` |
| `GET /threads/{id}/state` | Where is my conversation? Paused? Answered by whom? |

Open [src/gateway.py](src/gateway.py) and read the module docstring's two load-bearing
design notes — both are the kind of thing that breaks mysteriously if "fixed":

1. **Handlers are sync `def`, not `async def`** — the MCP tools call `asyncio.run()`
   underneath, which would crash inside an async handler's event loop. Sync handlers run
   in FastAPI's threadpool, where that's safe.
2. **The graph builds lazily on the first request** — so the server starts (and
   `/health` answers) before any API key exists. Offline, it serves the **demo graph**:
   same topology, keyword routing instead of an LLM supervisor, real MCP/A2A/SQL tools
   underneath, every response tagged `"mode": "offline-demo"`.

### 11.1 Start it

In your main terminal (the SwiftShip terminal from Chapter 7 still running):

```powershell
python -m uvicorn src.gateway:app --port 8000
```

**Expected:** `Uvicorn running on http://127.0.0.1:8000`. This terminal is now busy
being the gateway — open a **third** PowerShell window for the next commands (no venv
needed; `Invoke-RestMethod` is plain PowerShell).

### 11.2 Call it

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

**Expected output (offline):**

```
status offline mode
------ ------- ----
ok        True offline-demo
```

Ask a question — offline this routes by keyword to the MCP-backed demo worker:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question": "Any open incident tickets for SwiftShip?"}'
```

**Expected:** `status: complete`, `mode: offline-demo`, `hops: {live_incidents}`, and an
`answer` containing the ServiceNow ticket table — fetched **over MCP** by the gateway's
demo worker. (With a key: `mode: live`, and `hops` shows real supervisor routing like
`{logistics_coordinator}`.)

Now trigger the human-review path. Ask about pickup capacity **with the SwiftShip
terminal stopped** (go to the Chapter 7 window, press `Ctrl+C`):

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question": "How much pickup capacity does the carrier have today?"}'
```

**Expected:** `status: needs_human` — the demo worker got `A2A_ERROR` (partner agent
down), the heuristic validator caught the leaked error, and the graph paused. The
response includes the `review` payload and a `thread_id`. Resume it as the human:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/resume -ContentType "application/json" -Body '{"thread_id": "<PASTE THE thread_id HERE>", "action": "edit", "text": "Carrier agent is down; capacity unknown. Ops team notified."}'
```

**Expected:** `status: complete`, `answer:` your text, `hops` ending in `human_reviewer`.
You just performed the diagram's entire bottom half over HTTP.

### ✅ Checkpoint 11

```powershell
python -m pytest tests/test_gateway.py -q
```

**Expected:** `8 passed`.

> ⚠️ `[Errno 10048] ... address already in use` → something already owns port 8000; use
> `--port 8002` (and aim the `Invoke-RestMethod` calls there too).
> ⚠️ Restarting the gateway **forgets all threads** — MemorySaver is in-process memory.
> The production fix is swapping in a SqliteSaver/PostgresSaver (one line, see
> CHALLENGES_GUIDE.md §3.10).

---

## Chapter 12 — The Frontend: a UI for free (20 min)

> **You are here:** `User / Frontend` — the top of the diagram. Last box.

**What is this box?** Anything that speaks HTTP can be the frontend. You already have
one, built into FastAPI: with the gateway running, open this in your **browser**:

```
http://127.0.0.1:8000/docs
```

That's **Swagger UI** — interactive documentation generated from the gateway's code:

1. Click `POST /ask` → **Try it out**.
2. Edit the JSON body: `{"question": "Any open incident tickets for SwiftShip?"}`.
3. Click **Execute** — the response appears below, same JSON you saw in PowerShell.

For a *real* chat page (HTML + a `fetch()` call, ~40 lines), see **Appendix B** — it
needs nothing but a text editor and a browser. The architectural point either way: the
frontend knows **four URLs and zero Python**. Swap the entire agent system behind the
gateway and the frontend never notices.

### ✅ Checkpoint 12

You executed `/ask` from the browser and got a JSON response with `status` and `answer`.

---

## Chapter 13 — Grand Finale: every box at once (30 min)

Three terminals, the full diagram live:

| Terminal | Runs | Diagram boxes |
|---|---|---|
| A | `python -m src.a2a.external_agent` | External Agent |
| B | `python -m uvicorn src.gateway:app --port 8000` | API Gateway → LangGraph Runtime → everything |
| C | your `Invoke-RestMethod` commands (or the browser at `/docs`) | User / Frontend |

(Terminals A and B need the venv active and the project folder as the working directory;
quotes around the `cd` path.)

**The flagship question** — in terminal C:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask -ContentType "application/json" -Body '{"question": "Are there open incidents for SwiftShip, and what does their partner agent say about pickup capacity and hub congestion?"}'
```

Trace what just happened against the diagram, box by box:

1. **User/Frontend** (C) sent HTTP to the **API Gateway** (B).
2. The gateway invoked the **LangGraph Runtime**; the **Supervisor** (or the keyword
   router, offline) picked a worker.
3. The worker's **Tool Node** dialed the **MCP Client** → spawned the **ServiceNow MCP
   server** → got the ticket table.
4. The **A2A** tool called the **External Agent** (A) over HTTP → got live capacity.
5. The **Validator** judged the draft; `complete` → **Auto Complete → End** — or, if
   something was down, `needs_human` → **Human Review**, and your `/resume` was the
   **Resume Graph** arrow.

Then the full proof — in a terminal with the venv active:

```powershell
python -m pytest tests/ -q
```

**Expected:** **106 passed** — 64 from the original system, 42 from the five layers you
just built (run `python -m pytest tests/ -q -m "not slow"` if you want to skip the
subprocess round-trip, and add `python main.py --offline-check` for the v1 sanity pass).

**You're done.** You built: three MCP servers and a protocol client, an external A2A
agent and its tool, a fifth worker, a v2 supervisor, a two-judge validator gate, a
pausable/resumable human-review node, an HTTP gateway, and a frontend — every box on
the diagram, every one verified by a test you can rerun anytime.

### Where to go next

- `labs/lab10_production_layers.py` — the same layers, cell by cell, with exercises.
- `CHALLENGES_GUIDE.md` — what breaks in production and how each pattern here mitigates it.
- `ARCHITECTURE.md` §4.8–4.13 — the formal contracts for everything you just built.

---

## Appendix A — Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `The ampersand (&) character is not allowed` | Unquoted path. `cd "C:\...\week3&4"` — quotes, always. |
| `ModuleNotFoundError: No module named 'src'` | Wrong working directory. `cd` into the project folder first. |
| `ModuleNotFoundError: No module named 'mcp'` (or fastapi/httpx) | Dependencies missing → `pip install -r requirements.txt` with the venv active. |
| `UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14` | Harmless on 3.14. Ignore it everywhere. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`. |
| `[Errno 10048] address already in use` | Port taken → `--port 8002`, or close the other process. |
| Windows Firewall dialog when starting uvicorn | Local-only server → **Allow**. |
| `A2A_ERROR: carrier agent not reachable` | The Chapter 7 terminal isn't running → `python -m src.a2a.external_agent`. |
| `MCP_ERROR: TimeoutError` | First spawn on a cold machine can exceed the timeout — rerun; if persistent, check antivirus isn't scanning every Python spawn. |
| Gateway answers `needs_human` unexpectedly (offline) | Working as designed: a tool error leaked into the draft and the heuristic caught it. Read `review.validator_reason`. |
| `No LLM available: OFFLINE=1 or no API key set` | You ran an online-only path in offline mode. Set a key in `.env`, or stay on the chapter's offline track. |
| Garbled table characters (`â”€`, `�`) in console | `chcp 65001`, or use Windows Terminal instead of the legacy console. |
| Gateway forgot a `thread_id` after restart | MemorySaver is in-process. Expected; durable checkpointers are the production upgrade. |

## Appendix B — A real chat page in 40 lines (optional)

Save as `chat.html` anywhere, double-click it (gateway must be running):

```html
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Ops Copilot</title>
<style>
  body { font-family: system-ui; max-width: 720px; margin: 2rem auto; }
  #log { white-space: pre-wrap; border: 1px solid #ccc; padding: 1rem; min-height: 240px; }
  input { width: 78%; padding: .5rem; } button { padding: .5rem 1rem; }
</style></head>
<body>
<h2>Supply Chain Ops Copilot</h2>
<div id="log">Ask me about incidents, purchase orders, contracts, or carrier capacity.</div>
<p><input id="q" placeholder="Your question..."><button onclick="ask()">Ask</button></p>
<script>
async function ask() {
  const q = document.getElementById("q").value;
  const log = document.getElementById("log");
  log.textContent += "\n\nYOU: " + q + "\n...thinking...";
  const r = await fetch("http://127.0.0.1:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  });
  const data = await r.json();
  log.textContent += "\rCOPILOT (" + data.status + ", via " + (data.hops || []).join(" -> ") + "):\n"
    + (data.answer || JSON.stringify(data.review, null, 2));
}
</script>
</body>
</html>
```

(If the browser blocks the request with a CORS error, add FastAPI's `CORSMiddleware` to
`create_app` — five lines, search "fastapi cors" — or just use `/docs`.)

## Appendix C — Glossary

| Term | Plain English |
|---|---|
| **State** | The shared dictionary every node reads and updates — the team's clipboard. |
| **Reducer** | The merge rule for one state field. `add_messages` = append, don't overwrite. |
| **Node** | A plain function: state in, partial update out. |
| **Conditional edge** | "Read this state field, jump to the node it names." |
| **ReAct agent** | An LLM looping think → call tool → read result → think, until it can answer. |
| **Tool** | A Python function an agent may invoke; the docstring is the agent's only manual for it. |
| **Structured output** | Forcing the LLM to return a validated object (e.g. `Router`, `Verdict`) instead of prose. |
| **Supervisor** | The router LLM that picks which worker acts next, or FINISH. |
| **MCP** | Model Context Protocol — a standard wire format for "list your tools / run this tool". Connects agents to *systems*. |
| **A2A** | Agent-to-agent — asking another party's agent a question over HTTP. Connects agents to *agents*. |
| **Agent card** | An A2A agent's self-description at `/.well-known/agent.json` — who am I, what skills. |
| **Validator / judge** | The node that grades the draft answer: ship it, or send to a human. |
| **HITL** | Human-in-the-loop — `interrupt()` pauses the graph; `Command(resume=...)` continues it. |
| **Checkpointer** | Saves graph state per `thread_id`, enabling memory and pause/resume. MemorySaver = volatile. |
| **thread_id** | The conversation's key in the checkpointer — same id, same memory. |
| **Errors-as-strings** | `SQL_ERROR:` / `MCP_ERROR:` / `A2A_ERROR:` — failures agents can read and react to. |

## Appendix D — Diagram box → file map

| Diagram box | File(s) | Built in |
|---|---|---|
| User / Frontend | Swagger `/docs`, Appendix B `chat.html` | Ch 12 |
| API Gateway | [src/gateway.py](src/gateway.py) | Ch 11 |
| LangGraph Runtime | [src/graph.py](src/graph.py), [src/graph_v2.py](src/graph_v2.py) | Ch 4, 10 |
| Supervisor / Planner | [src/agents/supervisor.py](src/agents/supervisor.py), [src/agents/supervisor_v2.py](src/agents/supervisor_v2.py) | Ch 4, 8 |
| Internal Agents | [src/agents/workers.py](src/agents/workers.py) | Ch 3, 8 |
| Tool Node | [src/agents/tools_sql.py](src/agents/tools_sql.py), tools_ml, tools_rag, [src/agents/tools_mcp.py](src/agents/tools_mcp.py) | Ch 2, 6 |
| MCP Client | [src/mcp_servers/client.py](src/mcp_servers/client.py) | Ch 6 |
| MCP Servers (SQL/SAP/ServiceNow) | [src/mcp_servers/](src/mcp_servers/) `sql_server.py`, `sap_server.py`, `servicenow_server.py` | Ch 5 |
| External Agent | [src/a2a/external_agent.py](src/a2a/external_agent.py) | Ch 7 |
| A2A link | [src/agents/tools_a2a.py](src/agents/tools_a2a.py) | Ch 7 |
| Validator / Judge | [src/agents/validator.py](src/agents/validator.py) | Ch 9 |
| Human Review / Resume | `make_human_review_node` in [src/graph_v2.py](src/graph_v2.py) | Ch 10 |
| Shared state | [src/state.py](src/state.py), [src/state_v2.py](src/state_v2.py) | Ch 1, 8 |
