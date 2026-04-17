# Technical Architecture Research

> **Purpose:** evaluate technology and architecture options for each layer of the agent framework defined in `01-agent-framework-blueprint.md` and staged in `02-implementation-roadmap.md`
>
> **Scope:** technology comparisons, architecture comparisons, open questions, and final recommendations
>
> **Based on:** blueprint §3–19, roadmap Stage 0–7, research files `01`–`05`
>
> **Date:** 2026-04

---

## Table of Contents

1. [Runtime Engine](#1-runtime-engine)
2. [Language and Async Runtime](#2-language-and-async-runtime)
3. [Schema Validation and Typing](#3-schema-validation-and-typing)
4. [Persistence and Checkpointing](#4-persistence-and-checkpointing)
5. [Event Streaming](#5-event-streaming)
6. [LLM Abstraction Layer](#6-llm-abstraction-layer)
7. [Tool Execution and Sandboxing](#7-tool-execution-and-sandboxing)
8. [Vector Store and Memory](#8-vector-store-and-memory)
9. [Guardrails and Policy](#9-guardrails-and-policy)
10. [Observability and Tracing](#10-observability-and-tracing)
11. [Evaluation Harness](#11-evaluation-harness)
12. [Async Job Queue](#12-async-job-queue)
13. [Serving Layer](#13-serving-layer)
14. [Protocol Adapters: MCP and A2A](#14-protocol-adapters-mcp-and-a2a)
15. [Open Questions](#15-open-questions)
16. [Proposed Architecture and Technology Choices](#16-proposed-architecture-and-technology-choices)

---

## 1. Runtime Engine

The runtime engine is the most architecturally consequential decision. Everything else builds on top of it.

### 1.1 Options

#### **Option A: Build on LangGraph**

LangGraph is a graph-based state machine runtime from LangChain. Nodes are Python functions; edges are static or conditional. State is a `TypedDict` shared across all nodes.

| Dimension | Assessment |
|-----------|------------|
| Control granularity | ★★★★★ — full control of every node function |
| Checkpointing | Built-in: `SqliteSaver`, `PostgresSaver`, `RedisSaver` |
| Streaming | Built-in: token-level and state-transition level |
| Human-in-the-loop | First-class `interrupt()` mechanism |
| Observability | LangSmith integration is seamless |
| Durability | Production-tested at scale |
| Vendor coupling | LangChain ecosystem; LangSmith for observability |
| Boilerplate | Medium — graph primitives add ceremony |
| Community | Large, active, well-documented |

```python
# Canonical LangGraph pattern
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_conditional_edges("planner", route_fn, {"execute": "executor", "done": END})
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("./checkpoints.db"))

# Resume after crash: pass existing thread_id
result = graph.invoke(input, config={"configurable": {"thread_id": run_id}})
```

**Strengths:** battle-tested, streaming + interrupt built-in, persistent checkpoints, no code needed for durability  
**Weaknesses:** LangChain dependency chain, `TypedDict` state model is less type-safe than Pydantic, LangSmith creates soft SaaS coupling

---

#### **Option B: Build a custom graph engine**

Design a minimal graph/state machine explicitly matched to the blueprint's `Run`, `Step`, and `Event` contracts.

| Dimension | Assessment |
|-----------|------------|
| Control granularity | ★★★★★ — total control |
| Schema alignment | Perfect: contracts are your own Pydantic models |
| Vendor coupling | None |
| Checkpointing | Must build: serialize `Run` state to DB after each transition |
| Streaming | Must build: `asyncio.Queue` or SSE |
| Boilerplate | High initially; pays off at Stage 3+ |
| Risk | Implementation risk, especially for interrupt/resume |

Core building block: a `GraphEngine` that steps through a `Dict[str, Callable]` of node functions, persisting `RunState` after each transition.

```python
class GraphEngine:
    def __init__(self, nodes: dict[str, Callable], checkpointer: Checkpointer):
        self.nodes = nodes
        self.checkpointer = checkpointer

    async def run(self, run: Run) -> Run:
        while run.current_node not in TERMINAL_NODES:
            node_fn = self.nodes[run.current_node]
            run = await node_fn(run)
            run = await self.checkpointer.save(run)   # persist every transition
            yield RunEvent.from_run(run)               # emit to event bus
        return run
```

**Strengths:** zero external dependencies, contracts are pure Pydantic, full alignment with blueprint  
**Weaknesses:** 2–4 weeks of engineering before reaching Stage 1 exit criteria

---

#### **Option C: Temporal**

Temporal is a durable workflow orchestration platform. Workflows are Python code; Temporal handles durability, retries, and fault tolerance at the infrastructure level.

| Dimension | Assessment |
|-----------|------------|
| Durability | ★★★★★ — enterprise grade |
| Control | ★★★ — workflow code is constrained |
| Checkpointing | Implicit — no code required |
| Streaming | Not native; requires workarounds |
| Setup complexity | High — separate Temporal server required |
| LLM ecosystem | No native LLM integration |
| Best for | Long-running workflows (days), financial-grade durability |

**Verdict:** overkill for Stage 1–4; consider at Stage 6+ for queue-backed multi-agent runs if scale demands it.

---

### 1.2 Comparison Summary

| Criterion | LangGraph | Custom Engine | Temporal |
|-----------|-----------|---------------|----------|
| Time-to-Stage-1 | Fastest | Moderate | Slowest |
| Durability (built-in) | ✅ | ❌ (build) | ✅ |
| Blueprint alignment | Partial | Full | Partial |
| Vendor coupling | Medium (LangChain) | None | Medium (Temporal) |
| Streaming | ✅ | ❌ (build) | ❌ (build) |
| Long-term flexibility | Medium | High | Medium |
| Production maturity | High | Unknown | Very High |

---

## 2. Language and Async Runtime

### 2.1 Options

#### **Python 3.11+ with asyncio**

- Dominant language in AI/ML; every LLM SDK has a first-class Python client
- `asyncio` enables non-blocking LLM and tool calls (critical for parallelism)
- Pydantic v2 (Rust-backed) gives fast, rich schema validation
- `uv` for fast dependency management; `pyproject.toml` for packaging
- Type checking via `mypy` or `pyright`
- Weak points: GIL (mitigated in Python 3.13 free-threaded mode), deployment packaging

#### **TypeScript / Node.js**

- Strong static typing; excellent developer experience
- Rich LLM SDK ecosystem (OpenAI, Anthropic have first-class TS SDKs)
- Better for web-first UIs, CLI tools, and SDK distribution
- Lacks Python's ML/scientific ecosystem
- LangGraph.js exists but is less mature than the Python version

#### **Both (polyglot)**

LangChain maintains Python and TypeScript SDK parity. Adding TypeScript later for an SDK/CLI layer is viable. Core runtime stays Python.

### 2.2 Verdict

**Python 3.11+ (primary).** TypeScript client SDK is a Stage 7 or post-M4 concern.

Key toolchain:
- **Package manager:** `uv` (10–100× faster than pip)
- **Types:** `mypy` in strict mode from Stage 0
- **Linting:** `ruff` (replaces flake8 + isort + black)
- **Testing:** `pytest` + `pytest-asyncio`

---

## 3. Schema Validation and Typing

The roadmap requires all core contracts (`Run`, `Step`, `ToolResult`, `Event`, `GuardrailResult`) to be schema-validated. This is the foundation of Stage 0.

### 3.1 Options

#### **Pydantic v2**

- De-facto standard for Python data validation
- Rust-backed core: ~5–50× faster than Pydantic v1
- `.model_json_schema()` exports JSON Schema for tool definitions
- First-class support in FastAPI, LangChain, and most LLM tooling
- `model_validator`, `field_validator` for custom business rules
- Discriminated unions for `Step.kind` variants

```python
from pydantic import BaseModel, Field
from typing import Literal

class ToolStep(BaseModel):
    kind: Literal["tool"] = "tool"
    tool_name: str
    arguments: dict
    risk_tier: int = Field(ge=0, le=4)

class AnswerStep(BaseModel):
    kind: Literal["answer"] = "answer"
    content: str

Step = ToolStep | AnswerStep  # discriminated union
```

#### **dataclasses + JSON Schema**

- Lighter weight; no external dependency
- No runtime validation by default
- Less ergonomic for nested models and custom validators

#### **attrs + cattrs**

- Faster serialization than Pydantic v1
- Smaller community in the LLM space
- Less integration with FastAPI and LLM toolkits

### 3.2 Verdict

**Pydantic v2.** No meaningful competition for this use case.

---

## 4. Persistence and Checkpointing

The roadmap requires checkpoints after every state transition (§11, Stage 1.5).

### 4.1 Options

#### **SQLite (via `aiosqlite`)**

- Zero infrastructure — single file on disk
- `aiosqlite` for async access
- LangGraph's default `SqliteSaver` uses it
- Suitable for single-node deployments and local dev
- Not suited for multi-worker setups or horizontal scaling
- `runs.db` file can be mounted as a Docker volume for persistence

#### **PostgreSQL (via `asyncpg` or `psycopg[asyncio]`)**

- Production-grade, multi-worker safe
- `asyncpg` is the fastest async PostgreSQL driver (2–5× faster than psycopg2)
- JSON/JSONB columns for `Run.context` and `Step.arguments`
- Full-text and vector search available via extensions
- Suitable for production multi-tenant deployments
- LangGraph's `PostgresSaver` provides drop-in support

#### **Redis (via `redis-py` async)**

- Sub-millisecond reads/writes — ideal for checkpoint hot path
- Also serves as event streaming bus and job queue (multipurpose)
- Data is in-memory: needs `appendonly yes` for durability
- Not a relational DB; structured queries require Redis Stack (JSON, Search modules)
- Best as a cache layer in front of PostgreSQL, not as primary store

#### **Hybrid: SQLite (dev) + PostgreSQL (prod)**

LangGraph's saver interface abstracts the backend. Switching from `SqliteSaver` to `PostgresSaver` is a config change. This is the standard pattern.

### 4.2 Schema for checkpoint store

```sql
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    state       JSONB NOT NULL,          -- full Run model serialized
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE events (
    event_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_events_run_id ON events(run_id);
CREATE INDEX idx_events_type ON events(event_type);
```

### 4.3 Verdict

- **Stage 0–2 (dev/local):** SQLite via `aiosqlite`
- **Stage 3+ (production path):** PostgreSQL via `asyncpg`
- **Same interface** via an abstract `Checkpointer` base class

---

## 5. Event Streaming

The roadmap requires typed events for every state transition, consumable in real-time (Stage 1.7, 1.8).

### 5.1 Options

#### **In-process `asyncio.Queue` + SSE**

- Zero infrastructure
- `asyncio.Queue` collects events; FastAPI SSE endpoint consumes them
- Suitable for single-process deployments
- Does not cross process boundaries (breaks in multi-worker mode)

```python
async def event_generator(run_id: str, queue: asyncio.Queue):
    while True:
        event = await queue.get()
        yield f"data: {event.model_dump_json()}\n\n"
        if event.type == "run_completed":
            break
```

#### **Redis Streams**

- Persistent, multi-consumer event log
- Survives process restarts
- `XADD` to append; `XREAD` / `XREADGROUP` to consume
- Works across multiple workers
- Adds Redis infrastructure dependency
- Same Redis instance as job queue (multipurpose)

#### **Kafka / Redpanda**

- Enterprise-grade, high-throughput event log
- Significant operational overhead for the problem at hand
- Overkill until Stage 6+ multi-agent fan-out at scale

#### **NATS JetStream**

- Lightweight, extremely fast
- Good for microservice event buses
- Less common in the Python AI ecosystem

### 5.2 Verdict

- **Stages 0–4:** In-process `asyncio.Queue` + Server-Sent Events (SSE)
- **Stage 6+ (multi-worker):** Redis Streams (same Redis as job queue)

The `EventBus` interface should be abstract from the start so the backend can be swapped without changing emitters.

---

## 6. LLM Abstraction Layer

The framework needs a provider abstraction that supports OpenAI, Anthropic, Google Gemini, and local models (Stage 7.3).

### 6.1 Options

#### **LiteLLM**

LiteLLM provides a unified OpenAI-compatible API across 100+ LLM providers.

```python
from litellm import acompletion

response = await acompletion(
    model="anthropic/claude-sonnet-4-5",    # or "openai/gpt-4o", "ollama/llama3"
    messages=[{"role": "user", "content": "..."}],
    tools=tool_schemas,
    stream=True,
)
```

| Feature | LiteLLM |
|---------|---------|
| Provider coverage | 100+ (OpenAI, Anthropic, Gemini, Mistral, Ollama, Bedrock, etc.) |
| Unified streaming | ✅ |
| Tool calling | ✅ (normalized across providers) |
| Structured outputs | ✅ |
| Async support | ✅ (`acompletion`) |
| Cost tracking | ✅ (built-in `response.usage`) |
| Fallback chains | ✅ (`fallbacks=[...]`) |
| Proxy mode | ✅ (self-hosted gateway) |
| Caching | ✅ (in-memory, Redis, s3) |

**Weakness:** LiteLLM can lag provider releases by days; advanced provider-specific features may not be exposed.

#### **Custom thin wrapper**

Write a `ModelRunner` that directly calls provider SDKs (openai, anthropic, google-genai).

- Full control over every provider feature
- No proxy overhead
- ~200–400 lines of code per provider
- Must maintain yourself when APIs change

#### **LangChain ChatModel**

- Well-tested, broad provider coverage
- Pulls in the LangChain dependency tree
- Best choice if using LangGraph as the runtime engine

#### **OpenAI Agents SDK + LiteLLM**

OpenAI Agents SDK uses OpenAI-format messages. LiteLLM provides an OpenAI-compatible endpoint for any provider.

```python
from agents import set_default_openai_client
from openai import AsyncOpenAI
import litellm

# Route OpenAI SDK calls through LiteLLM proxy
client = AsyncOpenAI(base_url="http://0.0.0.0:4000", api_key="...")
set_default_openai_client(client)
```

### 6.2 Verdict

**LiteLLM** for provider abstraction. Rationale:
- Provider swap is config-only (satisfies Stage 7.3 acceptance criterion)
- Built-in cost tracking feeds Stage 4.3
- Fallback chains support graceful degradation (blueprint §3.12)
- Proxy mode enables centralized rate-limiting and caching

**Model routing strategy** (per research `05` §3.4):
| Role | Model Tier | Example |
|------|------------|---------|
| Planning / complex reasoning | Reasoning model | `claude-sonnet-4-5`, `o4-mini` |
| Tool selection / routing | Fast model | `gpt-4.1-mini`, `claude-haiku-4-5` |
| Summarization / extraction | Fast model | same |
| Final user response | Standard model | `claude-sonnet-4-5`, `gpt-4o` |

---

## 7. Tool Execution and Sandboxing

The roadmap requires sandbox execution for `run_command` (risk tier 3) and code execution (Stage 7.5).

### 7.1 Sandboxing Options

| Technology | Isolation | Startup | Self-hostable | Best for |
|------------|-----------|---------|---------------|----------|
| **E2B** | VM-level | ~150ms | No (SaaS) | Production managed sandboxes |
| **Docker** | Container | 500ms–2s | Yes | Self-hosted, full Linux env |
| **WASM (Pyodide)** | Process | ~10ms | Yes | Pure Python, no subprocess |
| **RestrictedPython** | Language | ~0ms | Yes | Internal tools, low-risk code |
| **subprocess + seccomp** | Process + syscall filter | ~5ms | Yes | Simple command execution |

#### **E2B** (recommended for production code execution)

- MicroVM per sandbox, spins up in ~150ms
- Full Linux environment; install any package
- Streaming stdout/stderr
- Python and JavaScript SDKs
- Per-use pricing (cost tracked per invocation)

```python
from e2b_code_interpreter import Sandbox

async def run_code_safely(code: str) -> ToolResult:
    async with Sandbox() as sbx:
        execution = await sbx.run_code(code)
        return ToolResult(
            tool_name="run_code",
            ok=execution.error is None,
            output={"stdout": execution.logs.stdout, "stderr": execution.logs.stderr},
            duration_ms=execution.execution_time_ms,
        )
```

#### **Docker** (recommended for self-hosted shell execution)

- Use `docker-py` SDK for container lifecycle
- Pre-built executor image with required tools
- Bind-mount only the run workspace directory (no host filesystem access)
- Apply network restrictions via `--network none` or a restricted bridge network

```python
import docker

async def run_command_in_sandbox(cmd: str, workdir: str) -> ToolResult:
    client = docker.from_env()
    container = client.containers.run(
        "agent-executor:latest",
        command=cmd,
        volumes={workdir: {"bind": "/workspace", "mode": "rw"}},
        network_mode="none",           # no network access
        mem_limit="512m",
        cpu_period=100000,
        cpu_quota=50000,               # 50% CPU
        remove=True,
        stdout=True,
        stderr=True,
    )
    return ToolResult(tool_name="run_command", ok=True, output={"stdout": container.decode()})
```

### 7.2 Tool Registry Design

Tool definitions should be stored as Pydantic models and exported to JSON Schema for LLM tool-calling:

```python
class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict           # JSON Schema
    output_schema: dict          # JSON Schema
    side_effect_level: Literal["read", "transform", "write", "execute", "destructive"]
    risk_tier: int = Field(ge=0, le=4)
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = RetryPolicy()
    idempotent: bool = False
    approval_required: bool = False
    capability_tags: list[str] = []
    examples: list[ToolExample] = []

    def to_llm_schema(self) -> dict:
        """Export as OpenAI-compatible tool schema."""
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "strict": True,
        }}
```

### 7.3 Verdict

- **Stage 2 built-in tools:** subprocess with `asyncio.create_subprocess_exec` + timeout enforcement
- **Stage 3+ risky execution (tier 3):** Docker container sandbox
- **Stage 7 production code execution:** E2B (managed) or Docker (self-hosted)
- **Filesystem boundaries:** restrict via Docker volume mounts; no host `/etc`, `/root`, or env vars accessible

---

## 8. Vector Store and Memory

The roadmap separates memory into working, episodic, semantic, and artifact stores (Stage 5). Each has different access patterns.

### 8.1 Vector Database Options

| DB | License | Startup | Scale | Best for |
|----|---------|---------|-------|---------|
| **Chroma** | Apache 2.0 | Embedded or server | Medium (100M vectors) | Dev speed, small/medium |
| **Qdrant** | Apache 2.0 | Self-hosted (Rust) | Large | On-prem, payload filtering |
| **pgvector** | PostgreSQL ext | Co-located with Postgres | Medium | PostgreSQL shops |
| **Weaviate** | Open + Cloud | Self-hosted or cloud | Large | Multi-tenancy, hybrid search |
| **Pinecone** | SaaS only | Managed | Very large | Managed at scale |

**For this framework:**

```
Local development / Stage 5 → Chroma (embedded mode, zero infra)
Production (Stage 7)        → Qdrant (on-prem) or pgvector (if Postgres already running)
Multi-tenant SaaS           → Weaviate Cloud or Pinecone
```

### 8.2 Memory Architecture

| Tier | Storage | Technology | Access Pattern |
|------|---------|-----------|----------------|
| **Working memory** | LLM context (in-state) | `Run.context` dict | Direct read/write in state reducer |
| **Episodic memory** | Recent run summaries | PostgreSQL + pgvector | Semantic retrieval at planner start |
| **Semantic memory** | Durable facts | Chroma / Qdrant | Top-k semantic search |
| **Artifact store** | Files, patches, reports | Object store (S3/MinIO) + metadata in Postgres | Retrieve by run_id or tag |

### 8.3 Embedding Model

| Model | Dims | Context | Notes |
|-------|------|---------|-------|
| `text-embedding-3-small` | 1536 | 8K tokens | Good balance cost/quality |
| `text-embedding-3-large` | 3072 | 8K tokens | Best quality, 3× cost |
| `nomic-embed-text-v1.5` | 768 | 8K tokens | Open source, Matryoshka, self-hostable |
| `bge-m3` | 1024 | 8K tokens | Multi-lingual, open source |

**Recommendation:** `text-embedding-3-small` for cloud deployments; `nomic-embed-text-v1.5` for air-gapped / cost-sensitive.

### 8.4 Verdict

- **Stage 5 (dev):** Chroma (embedded) + `text-embedding-3-small`
- **Stage 7 (production):** Qdrant (server mode) or pgvector
- **Abstract behind a `MemoryStore` interface** from Stage 5 day one

---

## 9. Guardrails and Policy

The roadmap requires guardrails at 5 points: input, planning, tool-input, tool-output, response (Stage 3).

### 9.1 Options

#### **Custom Pydantic-based validators**

The simplest and most predictable approach. Each guardrail is a `GuardrailCheck` function that takes an input and returns a `GuardrailResult`.

```python
class GuardrailResult(BaseModel):
    guardrail_id: str
    type: Literal["input", "output", "planning", "tool"]
    passed: bool
    violations: list[str]
    action_taken: Literal["allow", "block", "modify", "escalate"]

async def check_input(user_input: str) -> GuardrailResult:
    violations = []
    if contains_injection_pattern(user_input):
        violations.append("prompt_injection_detected")
    return GuardrailResult(
        guardrail_id="input_safety",
        type="input",
        passed=len(violations) == 0,
        violations=violations,
        action_taken="block" if violations else "allow",
    )
```

**Strength:** zero dependencies, fully predictable, easy to test  
**Weakness:** requires manual implementation of each check

#### **Guardrails.ai**

Open-source framework with a library of pre-built validators (NSFW, secrets detection, JSON format, competitor mentions, etc.).

```python
from guardrails import Guard
from guardrails.hub import DetectPII, ValidJSON

guard = Guard().use(DetectPII).use(ValidJSON)
result = guard.validate(agent_output)
```

| Feature | Assessment |
|---------|------------|
| Pre-built validators | ✅ 100+ via Guardrails Hub |
| Custom validators | ✅ |
| Async support | ✅ (v0.5+) |
| LLM-based validators | ✅ |
| Dependency weight | Medium |
| Production maturity | Medium |

**Strength:** large library of ready-made validators  
**Weakness:** added dependency; some validators have LLM calls (latency + cost)

#### **NeMo Guardrails (NVIDIA)**

Uses Colang DSL to define conversational guardrails.

- Powerful for conversational safety
- High learning curve (Colang is a new language)
- Over-engineered for structured agent pipelines
- Best for chat-style agents, not structured workflow agents

#### **LlamaGuard / ShieldGemma (model-based)**

LlamaGuard 3 is a fine-tuned Llama model trained specifically to classify harmful content categories.

```python
# Use as an output guardrail
harm_categories = ["violence", "hate", "self-harm", "illegal_activity"]
classification = guard_model.classify(agent_output, categories=harm_categories)
```

| Feature | Assessment |
|---------|------------|
| Coverage | Broad (14 harm categories) |
| Latency | ~100–300ms (LLM inference) |
| Cost | Per token |
| Self-hostable | ✅ (open weights) |
| Bypass resistance | Medium |

**Best for:** high-stakes content safety as one layer among many, not as primary guardrail mechanism

### 9.2 Verdict

**Custom Pydantic validators + Guardrails.ai selectively + LlamaGuard for content safety** (layered approach):

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Schema validation | Pydantic v2 | Zero cost, deterministic |
| Pattern-based checks | Custom regex + allow/deny lists | Fast, transparent |
| Structured output guardrails | Guardrails.ai validators | Ready-made, composable |
| Content safety (tier 3+ actions) | LlamaGuard 3 | Final barrier for high-risk outputs |
| Policy engine (risk tiers) | Custom rule engine | Domain-specific, config-driven |

---

## 10. Observability and Tracing

The roadmap requires full trace persistence, state snapshots, cost tracking, and replay (Stage 4).

### 10.1 Options

#### **Langfuse** (recommended)

Open-source LLM observability platform.

| Feature | Assessment |
|---------|------------|
| Self-hostable | ✅ Docker Compose in 5 minutes |
| Framework agnostic | ✅ (not tied to LangChain) |
| OpenTelemetry input | ✅ |
| Cost tracking | ✅ (per model, per run) |
| Prompt versioning | ✅ |
| Dataset-based evals | ✅ |
| Multi-modal traces | ✅ |
| Python SDK | ✅ |

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Wrap a run
trace = langfuse.trace(name="agent_run", id=run_id, user_id=user_id)
span = trace.span(name="planner", input={"goal": goal})
span.end(output={"action": planned_action}, usage={"input": 500, "output": 200})
```

#### **LangSmith** (alternative if using LangGraph)

- Deeply integrated with LangGraph — zero-config tracing
- Best-in-class UX for graph visualization
- SaaS (no full self-hosting without contract)
- Best choice if the runtime decision is LangGraph

#### **Arize Phoenix**

- Open-source, OTEL-native
- Strong for RAG evaluation and retrieval quality
- Good for teams with existing OTEL infrastructure

#### **Custom OpenTelemetry**

- Use `opentelemetry-sdk` to emit spans
- Ship to Grafana Tempo, Jaeger, or Datadog
- Maximum control, maximum effort
- Best for organizations with existing OTEL infrastructure

### 10.2 Structured Logging

Use `structlog` for structured JSON logs:

```python
import structlog

log = structlog.get_logger()
log.info("tool_call", run_id=run_id, tool_name="read_file", duration_ms=45, status="success")
```

### 10.3 Verdict

- **Primary:** Langfuse (self-hosted) — framework-agnostic, full data ownership
- **If LangGraph runtime chosen:** LangSmith
- **Logging:** `structlog` with JSON output
- **Metrics:** emit to Prometheus via `prometheus-client`; dashboard in Grafana

---

## 11. Evaluation Harness

The roadmap requires eval fixtures in CI from Stage 1, with scoring, regression tracking, and benchmark categories (Stage 4.8, 4.9).

### 11.1 Options

#### **Custom eval runner**

A simple fixture-runner is ~100 lines:

```python
class EvalFixture(BaseModel):
    id: str
    category: str           # tool_correctness | plan_quality | resume | guardrail | cost
    input: dict
    expected: dict
    scorer: str             # exact_match | llm_judge | schema_check | cost_threshold

class EvalResult(BaseModel):
    fixture_id: str
    passed: bool
    score: float            # 0.0–1.0
    details: dict

async def run_eval_suite(fixtures: list[EvalFixture]) -> list[EvalResult]:
    ...
```

**Strength:** total control over scoring logic, no external dependency  
**Weakness:** must build and maintain LLM-judge scorer

#### **DeepEval**

Open-source evaluation framework with pre-built metrics.

| Metric | Purpose |
|--------|---------|
| `AnswerRelevancyMetric` | Does the answer address the question? |
| `FaithfulnessMetric` | Is the answer grounded in retrieved context? |
| `ToolCorrectnessMetric` | Did the agent use the right tools correctly? |
| `HallucinationMetric` | Does the answer contain fabrications? |
| `TaskCompletionMetric` | Was the high-level task completed? |

```python
from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric, TaskCompletionMetric

evaluate(test_cases, metrics=[ToolCorrectnessMetric(), TaskCompletionMetric()])
```

- CI integration: `pytest-deepeval` plugin
- LLM-as-a-judge for qualitative metrics
- Dataset management

#### **RAGAS**

- Focused on RAG evaluation (retrieval quality, faithfulness, relevance)
- Less suited for general agent task evaluation
- Best as a supplementary scorer for Stage 5 memory quality

#### **Promptfoo**

- Config-based (YAML) regression testing
- Good for prompt regression testing
- Less suited for complex agent workflow evaluation

### 11.2 Verdict

**Custom runner + DeepEval metrics**:
- Custom runner for full control of fixture format and CI integration
- DeepEval metrics as scoring functions (import as a library, not a test runner)
- LLM-as-a-judge for qualitative assessment using a cheap fast model (Haiku, GPT-4.1-mini)
- Fixtures stored as JSONL files in `evals/fixtures/`

---

## 12. Async Job Queue

Required for Stage 6.6 (async queue mode for multi-agent runs).

### 12.1 Options

| Queue | Backend | Async-native | Python | Notes |
|-------|---------|-------------|--------|-------|
| **ARQ** | Redis | ✅ | ✅ | Lightweight, asyncio-native, simple |
| **Celery** | Redis/RabbitMQ | Partial | ✅ | Mature, rich monitoring, heavier |
| **Dramatiq** | Redis/RabbitMQ | Partial | ✅ | Cleaner API than Celery |
| **Temporal** | Temporal server | ✅ | ✅ | Durable workflows, enterprise grade |
| **Redis Queue (RQ)** | Redis | ❌ | ✅ | Sync workers, simpler |

**ARQ** example:
```python
from arq import create_pool

async def run_agent_task(ctx, run_id: str, goal: str):
    run = await resume_or_create_run(run_id, goal)
    return await graph_engine.run(run)

# Enqueue
redis = await create_pool(RedisSettings())
job = await redis.enqueue_job("run_agent_task", run_id, goal)
```

### 12.2 Verdict

**ARQ** (Stage 6) — asyncio-native, uses Redis (already in the stack), minimal setup. Evaluate Temporal if Stage 7 multi-agent workloads require durable distributed workflows.

---

## 13. Serving Layer

### 13.1 Options

| Framework | Async | SSE | WebSocket | Type hints | Notes |
|-----------|-------|-----|-----------|------------|-------|
| **FastAPI** | ✅ | ✅ | ✅ | ✅ | Standard choice |
| **Litestar** | ✅ | ✅ | ✅ | ✅ | Faster, stricter typing |
| **aiohttp** | ✅ | ✅ | ✅ | Partial | Lower-level |
| **Flask** | No | Partial | Partial | Partial | Sync-first |

### 13.2 API Surface (minimal)

```
POST   /runs                  # create a run
GET    /runs/{run_id}         # get run status
GET    /runs/{run_id}/events  # SSE stream of events
POST   /runs/{run_id}/approve # respond to approval interrupt
DELETE /runs/{run_id}         # cancel a run
GET    /health                # readiness probe
```

### 13.3 Verdict

**FastAPI + uvicorn** — battle-tested, Pydantic-native, SSE and WebSocket support, `BackgroundTasks` for Stage 1–5, ARQ integration for Stage 6.

---

## 14. Protocol Adapters: MCP and A2A

### 14.1 MCP Adapter

The `mcp` Python SDK (`pip install mcp`) provides:
- `StdioServerParameters` for local MCP servers (stdio transport)
- `SSEServerParameters` for remote MCP servers (HTTP + SSE)
- `ClientSession.list_tools()` / `ClientSession.call_tool()` for capability discovery and invocation

**Adapter design:**

```python
class MCPToolAdapter:
    """Wraps an MCP server as a ToolDefinition in the native registry."""

    async def discover(self, server: MCPServer) -> list[ToolDefinition]:
        async with ClientSession(server.params) as session:
            tools = await session.list_tools()
            return [self._convert(t) for t in tools.tools]

    async def invoke(self, tool_name: str, args: dict) -> ToolResult:
        result = await self.session.call_tool(tool_name, args)
        return ToolResult(
            tool_name=tool_name,
            ok=result.isError is not True,
            output=result.content,
            metadata={"provider": "mcp"},
        )
```

### 14.2 A2A Adapter

The A2A Python SDK provides:
- `AgentCard` discovery from `/.well-known/agent.json`
- `TaskSendParams` for task creation
- SSE-based task update streaming

```python
class A2AAgentAdapter:
    """Wraps a remote A2A agent as a tool in the native registry."""

    async def discover(self, agent_url: str) -> ToolDefinition:
        card = await fetch_agent_card(agent_url)
        return ToolDefinition(name=card.name, description=card.description, ...)

    async def invoke(self, goal: str, context: dict) -> ToolResult:
        task = await self.client.send_task(TaskSendParams(message=goal, metadata=context))
        async for update in task.stream():
            if update.final:
                return ToolResult(tool_name=self.card.name, ok=True, output=update.artifact)
```

### 14.3 Verdict

Both adapters should implement the same `ToolAdapter` interface so the policy engine, risk tiers, and approval gates apply uniformly regardless of tool source.

---

## 15. Open Questions

The following questions are unresolved and require a decision before or during implementation.

### 15.1 Runtime engine: LangGraph or custom?

**Tension:** LangGraph gives durability + streaming for free and cuts Stage 1 time in half. Custom engine gives full blueprint alignment and zero vendor coupling.

**Decision needed by:** Stage 0 / day one

**Leaning:** If the primary goal is shipping a working system quickly, **LangGraph** is the pragmatic choice. If the goal is a framework others will build on (SDK/platform play), **custom engine** pays off by Stage 4.

---

### 15.2 How to handle context window overflow in the state reducer?

As a run progresses, `Run.context.messages` will eventually exceed the model's context window. The options:

1. **Sliding window** — keep last N messages + system prompt
2. **Summarize on overflow** — use a cheap model call to compress history when `len(tokens) > threshold`
3. **Selective retention via embeddings** — retrieve only relevant prior messages by semantic similarity

No option is free of trade-offs. Summarization loses detail; retrieval adds latency; sliding window loses early context.

**Decision needed by:** Stage 1

---

### 15.3 Granularity of guardrail checks

Checking every tool call (including tier-0 reads) adds measurable latency. Only checking tier 2+ reduces overhead but creates blind spots for read-based information exfiltration.

**Options:**
- A) Check all tool calls (uniform safety, higher latency)
- B) Check only tier 2+ (lower latency, acceptable for most cases)
- C) Check probabilistically (1 in N for tier 0/1, always for tier 2+)

**Decision needed by:** Stage 3

---

### 15.4 Agent identity and auth across A2A boundaries

When Agent A delegates to Agent B via A2A, what identity does Agent B see? Options:
1. **Caller's identity propagated** — Agent B sees the end user's auth context
2. **Framework service identity** — Agent B sees the framework's service account
3. **Scoped delegation tokens** — A generates a JWT scoped to specific capabilities for B

Option 3 is most secure but adds complexity. Options 1 and 2 are simpler but have security trade-offs.

**Decision needed by:** Stage 7

---

### 15.5 Cost attribution in multi-agent runs

When a supervisor delegates to a specialist agent, how is token cost attributed?
- To the top-level `run_id`?
- To a child `run_id` per agent?
- To the delegating agent's budget?

This affects both billing and budget enforcement (blueprint §13.1).

**Decision needed by:** Stage 6

---

### 15.6 How opinionated should the eval harness be?

**Minimal:** the harness is a fixture runner with pluggable scorers. Teams supply their own rubrics.  
**Opinionated:** pre-defined metrics aligned to the roadmap's benchmark categories (tool_correctness, plan_quality, resume_accuracy, guardrail_compliance, cost).

The opinionated approach ships faster and ensures consistent measurement across stages.

**Decision needed by:** Stage 0 (eval harness scaffolding)

---

### 15.7 Checkpoint format: full state or delta?

**Full snapshot per step:** simple to implement, easy to replay; storage grows linearly with steps.  
**Delta encoding:** only stores changes per step; cheaper storage, more complex replay.

For most agent runs (≤50 steps, ~10–100KB state), full snapshots are fine. Delta is premature optimization.

**Leaning:** full snapshots. Revisit at Stage 4 if storage is a measurable concern.

---

### 15.8 Should tool examples be part of the tool definition or injected dynamically?

Static examples in `ToolDefinition.examples` are simple but can increase token usage for tools that are rarely called. Dynamic injection (select relevant examples per context) is more efficient but requires a retrieval step.

**Leaning:** static examples at Stage 2; dynamic few-shot at Stage 5 (when retrieval is available).

---

## 16. Proposed Architecture and Technology Choices

### 16.1 Summary Table

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Language** | Python 3.11+ | asyncio throughout |
| **Package manager** | `uv` | Replaces pip/poetry |
| **Linting/formatting** | `ruff` | Replaces black + flake8 + isort |
| **Type checking** | `mypy --strict` | From Stage 0 |
| **Schema validation** | Pydantic v2 | All core contracts |
| **Web framework** | FastAPI + uvicorn | SSE, WebSocket, Pydantic-native |
| **Runtime engine** | **LangGraph** (Stage 1–2) → evaluate custom engine at M2 | Pragmatic start; reassess |
| **Checkpoint store (dev)** | SQLite via `aiosqlite` | `SqliteSaver` |
| **Checkpoint store (prod)** | PostgreSQL via `asyncpg` | `PostgresSaver` |
| **Event streaming (local)** | `asyncio.Queue` + SSE | Zero infra |
| **Event streaming (multi-worker)** | Redis Streams | Upgrade at Stage 6 |
| **LLM abstraction** | LiteLLM | 100+ providers, built-in cost tracking |
| **Guardrails (schema)** | Custom Pydantic validators | Fast, deterministic |
| **Guardrails (content)** | Guardrails.ai + LlamaGuard 3 | For tier 3+ actions |
| **Tool sandboxing (dev)** | subprocess + timeout | Stage 2 |
| **Tool sandboxing (prod)** | Docker via `docker-py` | Stage 3 |
| **Tool sandboxing (code exec)** | E2B | Stage 7 |
| **Vector store (dev)** | Chroma (embedded) | Stage 5 |
| **Vector store (prod)** | Qdrant or pgvector | Stage 7 |
| **Embedding model** | `text-embedding-3-small` | Balanced cost/quality |
| **Observability** | Langfuse (self-hosted) + structlog | Full data ownership |
| **Metrics** | `prometheus-client` → Grafana | Standard stack |
| **Eval harness** | Custom runner + DeepEval metrics | CI from Stage 1 |
| **Job queue** | ARQ (Redis-backed) | Stage 6, asyncio-native |
| **MCP adapter** | Official `mcp` Python SDK | Stage 7 |
| **A2A adapter** | Google A2A Python SDK | Stage 7 |

---

### 16.2 Proposed Layered Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│ API Layer                                                           │
│  FastAPI (REST + SSE)  ·  CLI (click/typer)                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ Agent Composition Layer                                             │
│  Agent definitions (Pydantic)  ·  Handoff manager                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ Orchestration Runtime                                               │
│  LangGraph (Stage 1–2) / custom graph engine (Stage 3+)            │
│  RunManager  ·  GraphEngine  ·  StateReducer  ·  InterruptHandler  │
└──────────┬────────────────────────────────────────────┬────────────┘
           │                                            │
┌──────────▼──────────┐                    ┌───────────▼────────────┐
│ Guardrails Layer    │                    │ Policy Layer            │
│  Input validators   │                    │  RiskTier engine        │
│  Planning validators│                    │  Approval interrupts    │
│  Output filters     │                    │  Budget enforcement     │
│  LlamaGuard (tier3+)│                    │  AuthZ rules            │
└──────────┬──────────┘                    └───────────┬────────────┘
           │                                            │
┌──────────▼────────────────────────────────────────────▼────────────┐
│ Execution Layer                                                     │
│  ModelRunner (LiteLLM)  ·  ToolRunner  ·  StreamEvents             │
│  Planner node  ·  Executor node  ·  Termination logic              │
└──────────┬────────────────────────────────────────────┬────────────┘
           │                                            │
┌──────────▼──────────┐                    ┌───────────▼────────────┐
│ Tool Registry       │                    │ State + Memory Layer    │
│  Native tools       │                    │  Working memory (state) │
│  MCP adapter        │                    │  Episodic (Postgres)    │
│  A2A adapter        │                    │  Semantic (Chroma/Qdrant│
│  Docker/E2B sandbox │                    │  Artifacts (S3/MinIO)   │
└─────────────────────┘                    └────────────────────────┘
           │                                            │
┌──────────▼────────────────────────────────────────────▼────────────┐
│ Infrastructure Adapters                                             │
│  LiteLLM (OpenAI/Anthropic/Gemini/Ollama)                          │
│  PostgreSQL (asyncpg)  ·  SQLite (aiosqlite)                       │
│  Redis (ARQ queue + Streams)  ·  Chroma/Qdrant                     │
│  Langfuse (traces)  ·  Prometheus (metrics)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 16.3 Stage-by-Stage Technology Introduction

| Stage | Technologies Added |
|-------|--------------------|
| **0 – Foundation** | Python 3.11, uv, ruff, mypy, Pydantic v2, pytest, FastAPI skeleton |
| **1 – Runtime Kernel** | LangGraph, aiosqlite/SqliteSaver, asyncio.Queue + SSE, LiteLLM |
| **2 – Tool Execution** | Tool registry (Pydantic), subprocess sandbox, 5 built-in tools |
| **3 – Guardrails & Policy** | Custom validators, Guardrails.ai, Docker sandbox, LlamaGuard |
| **4 – Observability** | Langfuse (self-hosted), structlog, prometheus-client |
| **5 – Memory** | Chroma (embedded), text-embedding-3-small, episodic Postgres store |
| **6 – Composition** | ARQ + Redis, Redis Streams, subgraph/handoff engine |
| **7 – Ecosystem** | MCP Python SDK, A2A Python SDK, E2B, Qdrant/pgvector |

---

### 16.4 Key Architecture Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Runtime engine | LangGraph (short-term) | Fastest path to Stage 1–2 exit; reassess at M2 |
| Schema layer | Pydantic v2 | De-facto standard; fast; JSON Schema export for tools |
| Persistence | SQLite → PostgreSQL | Zero-infra dev; production-grade promotion via same interface |
| LLM routing | LiteLLM | Provider-agnostic; built-in cost; fallback chains |
| Guardrails approach | Layered (schema → pattern → model) | Defense in depth; low-cost checks first |
| Observability | Langfuse self-hosted | Open source; framework-agnostic; full data ownership |
| Eval strategy | Custom runner + DeepEval metrics | Control + ready-made scoring functions |
| Sandboxing path | subprocess → Docker → E2B | Escalate as risk tier requires |
| Memory (dev) | Chroma embedded | Zero infra at Stage 5 |
| Memory (prod) | Qdrant or pgvector | Depends on existing Postgres presence |
| Async queue | ARQ on Redis | asyncio-native; Redis already in stack |

---

### 16.5 Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LangGraph version lock-in | Medium | High | Wrap behind `GraphEngine` interface; custom engine ready at M2 |
| LiteLLM provider lag on new features | Low | Medium | Pin versions; fall back to direct SDK for critical features |
| SQLite contention in multi-worker | High | Medium | Postgres migration at Stage 3+ (pre-planned) |
| E2B cost at high volume | Medium | Medium | Self-hosted Docker fallback available |
| Context window overflow causes missed context | High | High | Implement summarization fallback at Stage 1 |
| Prompt injection despite guardrails | Medium | High | Defense-in-depth + LlamaGuard + privilege separation |
| Eval harness flakiness blocking CI | Medium | Medium | Separate fast unit evals (deterministic) from LLM-judge evals (nightly) |

---

## 17. Key Empirical Data from Research (April 2026)

This section consolidates benchmarks, cost data, and production reliability figures from the research corpus to ground architectural decisions.

### 17.1 Agent Reliability at Scale

Single-step LLM accuracy of 95% compounds poorly across a run:

| Steps | P(success) at 95%/step | P(success) at 99%/step |
|-------|------------------------|------------------------|
| 1 | 95% | 99% |
| 5 | 77% | 95% |
| 10 | 60% | 90% |
| 20 | 36% | 82% |

**Implication:** keeping runs short (≤8 steps, per roadmap target) is not just a performance goal — it is a reliability requirement. Every added step multiplies failure probability.

**Production benchmarks (structured tasks, April 2026):**
- Single-agent success rate: 60–85%
- Multi-agent supervisor pattern: 75–92%
- SWE-bench Verified (Claude 3.7 Sonnet + scaffolding): >70%
- HumanEval with agentic loop vs. zero-shot: 95.1% vs. 48.1%

---

### 17.2 Inference Cost Collapse

Inference cost has dropped 100–600× since GPT-4 launch. Cost decisions that seemed expensive in 2023 are often negligible today.

| Model / Year | Cost (output tokens) |
|---|---|
| GPT-4 (Mar 2023) | $30 / 1M tokens |
| GPT-4o (May 2024) | $15 / 1M tokens |
| GPT-4o-mini (2024) | $0.60 / 1M tokens |
| Llama 3 (self-hosted) | $0.20 / 1M tokens |
| Llama 4 (2026) | ~$0.05 / 1M tokens |

**Implication for model routing:** routing by model tier (planning → reasoning model, execution → fast model) is the highest-leverage cost optimization — 10–50× reduction with careful assignment.

---

### 17.3 Context Window Landscape (April 2026)

| Model | Context Window | Effective Utilization |
|-------|---------------|-----------------------|
| GPT-5 / o4 | 1M tokens | ~70% |
| Claude 4 Opus/Sonnet | 200–500K tokens | ~85% (best retrieval quality) |
| Gemini 2.5 Ultra | 2M tokens | ~80% |
| Llama 4 Scout | 10M tokens | ~60% |
| Llama 4 Maverick | 1M tokens | ~65% |

**Implication:** large context windows do not eliminate the need for structured memory. Claude 4's 85% effective utilization at 200–500K tokens outperforms models with larger but less effective windows.

---

### 17.4 Three-Level Evaluation Pyramid

Derived from production teams at scale (research `04`):

```
Level 3 — A/B Testing (days, highest cost)
├── Randomized production traffic split
├── Compare new vs. old agent version on real users
└── Measure business KPIs (task completion, escalation rate)

Level 2 — LLM-as-Judge + Human Eval (hours, medium cost)
├── Golden dataset (50+ examples)
├── Structured rubric: correctness, efficiency, safety
├── Human review of low-confidence outputs
└── Track score over time for regressions

Level 1 — Unit Tests (seconds, cheap, deterministic)
├── Schema validation (every contract)
├── Tool correctness (happy path + error + timeout)
├── Guardrail correctness (injection patterns blocked)
└── Checkpoint resume (state identity after crash)
```

**Rule:** Level 1 runs on every commit; Level 2 runs nightly or on model/prompt changes; Level 3 runs for major releases.

---

### 17.5 Approval Tier Model

From production deployments (research `04`):

| Tier | Category | Examples | Gate |
|------|----------|----------|------|
| **0** | Read-only, fully reversible | file read, search, fetch URL | None — execute immediately |
| **1** | Local writes, reversible | write file, generate patch | Log + async notification |
| **2** | Execution with local scope | run test, lint, build | Hard gate — block until approved |
| **3** | Irreversible / external scope | deploy, publish, delete, network mutation | Always human-initiated |

This maps directly to blueprint risk tiers 0–4 with a small consolidation (tier 1+2 = soft approval, tier 3–4 = hard approval / human-only).

---

### 17.6 MCP v2 Capabilities (March 2026)

MCP v2 added capabilities that affect the adapter design (Stage 7):

| Feature | v1 | v2 |
|---------|----|----|
| Streaming responses | ❌ | ✅ |
| Server-to-client sampling | ❌ | ✅ (servers can call back to host LLM) |
| Authentication | None | OAuth 2.1 |
| Transports | stdio, HTTP+SSE | + WebSocket |
| Resource subscriptions | ❌ | ✅ (push on change) |

**Key implication:** v2's server-to-client sampling enables compound tools — an MCP server that orchestrates its own LLM calls. The A2A adapter should be designed with the same consideration.

---

### 17.7 Temporal vs. LangGraph Checkpointing Decision Rule

From production patterns (research `04`):

| Scenario | Use LangGraph + Postgres | Use Temporal |
|----------|--------------------------|--------------|
| Task duration | < 30 minutes | Hours to days |
| Failure recovery | Acceptable to manual-restart | Must be automatic |
| Team familiarity | Python-native | OK to learn new system |
| Compliance audit trail | Via trace store | Built-in event history |
| Infrastructure budget | Low (Postgres already running) | Separate Temporal server |

**For this roadmap:** LangGraph + Postgres covers Stage 1–6. Temporal becomes relevant only if Stage 6 async queue mode produces runs exceeding 30 minutes, or compliance requirements demand Temporal's native audit history.

---

### 17.8 Model Routing Recommendation (April 2026)

Based on test-time compute scaling research and production cost data:

| Agent Role | Model Class | Example | Why |
|------------|-------------|---------|-----|
| Complex planning / multi-constraint | Reasoning model | `o4-mini`, `claude-sonnet-4-5` with extended thinking | Best multi-step accuracy |
| Routing / classification | Fast model | `gpt-4.1-mini`, `claude-haiku-4-5` | Low latency, 50× cheaper |
| Tool selection (routine) | Fast model | same | Simple decision, no reasoning needed |
| Final user-facing response | Standard model | `gpt-4o`, `claude-sonnet-4-5` | Quality matters for user trust |
| Evaluation / grading | Standard model | `gpt-4o` | Rubric-following ability |
| Summarization / extraction | Fast model | `claude-haiku-4-5` | Cost-sensitive, high-volume |

**Hybrid pipeline pattern (recommended):** reasoning model generates the plan → fast model executes individual tool-selection decisions → standard model writes the final answer.

