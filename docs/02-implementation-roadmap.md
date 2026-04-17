# Agent Framework — Implementation Roadmap

> **Approach:** staged value delivery — each stage ships demonstrable, production-usable capability before the next begins
>
> **Basis:** [01-agent-framework-blueprint.md](./01-agent-framework-blueprint.md)
>
> **References:** LangChain, LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, MCP, A2A Protocol

---

## Delivery Principles

| # | Principle | Rule |
|---|-----------|------|
| 1 | Ship at every stage | Each stage ends with a runnable, demonstrable system — not design artifacts |
| 2 | Quantify exits | Every stage has measurable acceptance criteria before moving forward |
| 3 | Defer complexity | Only build what the current stage demands — resist premature generalization |
| 4 | Validate before expanding | Prove value in benchmarks before adding the next capability |
| 5 | Eval-driven | Define success criteria before building; run evals in CI from Stage 1 onward |

---

## Stage Overview

| Stage | Name | Primary Capability Delivered | Exit Gate |
|-------|------|------------------------------|-----------|
| **0** | Foundation | Core schemas, CI, repo scaffold | Schemas compile + validate; CI green |
| **1** | Runtime Kernel | Durable agent loop with streaming | 3-step run survives crash and resumes; events stream |
| **2** | Tool Execution | Grounded, schema-validated tool use | Agent uses 5 tools to edit a file correctly |
| **3** | Guardrails & Policy | Governed, safe execution | Malicious input blocked; tier-3 action requires approval |
| **4** | Observability & Evals | Debuggable, measurable runs | Full replay from trace; baseline benchmark passes |
| **5** | Memory | Cross-run context retrieval | Agent uses prior run context to improve current task |
| **6** | Composition & Handoffs | Workflows, multi-agent delegation | Workflow with 3 nodes + agent handoff completes correctly |
| **7** | Ecosystem | External tool and agent interop | MCP tool + A2A agent invoked through adapter layer |

**Milestones:** M1 = Stages 0–2 · M2 = Stages 3–4 · M3 = Stage 5 · M4 = Stages 6–7

---

## Stage 0 — Foundation

**Goal:** establish project scaffold, typed contracts, and repeatable development workflow

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 0.1 | Repository structure | Matches blueprint module layout (`runtime/`, `tools/`, `state/`, `policy/`, `guardrails/`, `memory/`, `observability/`, `evals/`) |
| 0.2 | Core schema definitions | `Run`, `Step`, `ToolResult`, `Event`, `GuardrailResult` schemas defined, typed, and schema-validated |
| 0.3 | CI pipeline | Lint + type-check + unit tests run on every commit; failures block merge |
| 0.4 | Dev environment | Single command (`make dev` or equivalent) installs deps and runs tests |
| 0.5 | Baseline eval harness | Empty eval runner exists; structure ready for task fixtures |

### Exit Criteria

- [ ] All 5 schemas compile with zero type errors
- [ ] `pytest` / `npm test` passes with ≥1 test per schema
- [ ] CI pipeline green on `main`
- [ ] `README` documents local setup in ≤5 steps

---

## Stage 1 — Runtime Kernel

**Goal:** a single-agent graph loop that persists state, streams events, and resumes after process failure

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 1.1 | Run manager | Create, load, persist, and cancel runs; enforce max_steps and budget limits |
| 1.2 | Graph engine | Execute named nodes with typed transitions; support branching |
| 1.3 | Planner node | Return structured next-action (tool / answer) from goal + message history |
| 1.4 | State reducer | Merge observations into run state after each step |
| 1.5 | Checkpoint persistence | Run state written to durable store after every transition |
| 1.6 | Termination logic | Stop on: final answer, max steps, cost budget exceeded, repeated failures |
| 1.7 | Event stream | Typed events emitted for every state transition; consumable in real-time |
| 1.8 | Streaming output | Model token stream surfaced to caller during generation |

### Exit Criteria

- [ ] Demo: agent accepts goal → executes 3 mock steps → emits structured final answer
- [ ] Demo: kill process mid-run → restart → run resumes from last checkpoint with correct state
- [ ] Demo: token stream printed incrementally to CLI (not buffered)
- [ ] ≥80% unit test coverage on `runtime/` module
- [ ] Event log contains ≥8 typed events for a 3-step run
- [ ] ≥3 benchmark task fixtures added to eval harness

---

## Stage 2 — Tool Execution

**Goal:** agent invokes real tools with schema-validated inputs/outputs and structured error handling

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 2.1 | Tool registry | Register tools with: `name`, `description`, `input_schema`, `output_schema`, `side_effect_level`, `timeout`, `retry_policy`, `idempotency` |
| 2.2 | Tool executor | Invoke tool, normalize result into `ToolResult`, classify errors as recoverable vs fatal |
| 2.3 | 5 built-in tools | `read_file`, `write_file`, `search_code`, `run_command`, `fetch_url` |
| 2.4 | Structured results | Every tool returns schema-validated `ToolResult`; free-form text rejected |
| 2.5 | Timeout + retry | Tool terminated at timeout; retry with backoff for recoverable errors |
| 2.6 | Graceful degradation | On repeated tool failure: log error, reduce scope, surface failure state |
| 2.7 | Dynamic tool filtering | Middleware can restrict available tools per context |

### Exit Criteria

- [ ] Demo: agent reads a file → searches for a symbol → writes a corrected patch; file modified correctly
- [ ] Each tool has ≥3 tests: happy path, error response, timeout
- [ ] `ToolResult` includes `ok`, `duration_ms`, `provider`, `artifacts`
- [ ] On max retries exceeded: run enters `blocked` state with structured error, not silent failure
- [ ] Benchmark: ≥10 tool-use task fixtures; ≥80% pass rate

---

## Stage 3 — Guardrails & Policy

**Goal:** runtime enforces input/output validation, prompt injection defense, risk-tiered approvals, and execution budgets

### Why this stage precedes Observability

Guardrails and policy are runtime primitives (§3, §6.2.H, §13 of blueprint). Observable but ungoverned runs cannot be safely demonstrated. Safety gates must exist before widening any audience or scope.

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 3.1 | Input guardrails | Validate and sanitize user input before planner call; reject or escalate on violation |
| 3.2 | Output guardrails | Validate agent response schema and content before returning to caller |
| 3.3 | Tool input/output guardrails | Validate arguments before tool call; filter or flag results after |
| 3.4 | Prompt injection defense | Separate system context from user data; detect and block injection patterns |
| 3.5 | Risk tier assignment | Every tool tagged with tier 0–4; tier drives approval and sandbox policy |
| 3.6 | Policy engine | Allow / deny / transform actions based on configurable rules; log every decision |
| 3.7 | Approval interrupt | Pause run at tier ≥3 action; expose state; resume or reject after human decision |
| 3.8 | Budget enforcement | Enforce per-run limits: tokens, tool calls, wall-clock time, cost (USD) |
| 3.9 | Dynamic tool filtering | Filter tool set based on role, tenant, task stage, or active budget |

**Guardrail types (per blueprint §13.2):**

| Guardrail | When | Purpose |
|-----------|------|---------|
| Input | Before planner | Reject malicious / invalid user input |
| Planning | After plan emitted | Validate proposed action set |
| Tool input | Before tool call | Validate arguments against schema |
| Tool output | After tool call | Filter / validate results |
| Response | Before returning to user | Content safety, schema compliance |

### Exit Criteria

- [ ] Demo: injected prompt attempting privilege escalation is detected and blocked
- [ ] Demo: tier-3 action (`run_command`) triggers approval wait; run resumes correctly after approval
- [ ] Demo: token budget exceeded → run terminates with `budget_exceeded` status and cost breakdown
- [ ] `GuardrailResult` event logged for every guardrail check with `action_taken`
- [ ] ≥90% of policy rules and guardrail paths covered by unit tests
- [ ] Benchmark: 100% of simulated unsafe actions rejected or escalated

---

## Stage 4 — Observability & Evals

**Goal:** every run is fully traceable, cost-tracked, replayable, and systematically evaluated

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 4.1 | Trace persistence | All events stored with `run_id`, `step_id`, `event_type`, `timestamp`; queryable |
| 4.2 | State snapshots | Full state diff captured per step transition |
| 4.3 | Cost tracking | Token usage (input/output) and cost (USD) tracked per step, per run |
| 4.4 | Structured logs | JSON logs with consistent schema fields across all modules |
| 4.5 | Replay mode | Re-execute run from stored trace using mock tools; verify state identity |
| 4.6 | Metrics export | Latency, token usage, tool call counts, error taxonomy, cost per run |
| 4.7 | CLI trace viewer | Human-readable run timeline including tool call tree and cost breakdown |
| 4.8 | Eval harness | Task fixtures, scoring rubric, CI integration, regression tracking |
| 4.9 | Eval categories | Cover: tool correctness, plan quality, resume accuracy, guardrail compliance, cost |

### Exit Criteria

- [ ] Demo: replay a 5-step run from trace; transitions are identical; tool calls are mocked correctly
- [ ] Trace query by `run_id` returns all events in <100ms for a 20-step run
- [ ] Metrics endpoint returns JSON with ≥8 metric types including cost and token breakdown
- [ ] Eval harness runs in CI; failures block merge
- [ ] ≥20 task fixtures across 5 benchmark categories; ≥80% pass rate baseline documented

---

## Stage 5 — Memory

**Goal:** agent retrieves relevant cross-run context without polluting or overflowing the active prompt

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 5.1 | Working memory | Current run state, messages, plan, recent tool results explicitly bounded |
| 5.2 | Episodic store | Index and retrieve past run summaries by semantic similarity |
| 5.3 | Semantic store | Store and query durable facts (preferences, env metadata, domain knowledge) |
| 5.4 | Artifact store | Persist run outputs (patches, docs, reports) with run provenance |
| 5.5 | Retrieval middleware | Inject relevant memories into planner context before each decision |
| 5.6 | Memory budget | Cap injected token budget per call; evict lowest-relevance entries first |

### Exit Criteria

- [ ] Demo: agent completes similar task more efficiently using prior run outcome (≥1 fewer step)
- [ ] Retrieval latency <500ms at 1,000 stored episodes
- [ ] Memory injection adds ≤2,000 tokens to planner prompt
- [ ] Memory store entries are separated by type; no mixed-blob retrieval
- [ ] Eval: memory-augmented runs score ≥10% higher on relevant benchmark tasks vs baseline

---

## Stage 6 — Composition & Handoffs

**Goal:** directed multi-node workflows, explicit agent-to-agent handoffs, and parallel execution

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 6.1 | Subgraph support | A graph node can invoke a nested graph; context transferred correctly |
| 6.2 | Workflow templates | Declarative workflow definition (YAML/JSON); ≤50 lines for a 4-node graph |
| 6.3 | Supervisor pattern | Orchestrator agent routes tasks to ≥2 specialist agents and merges results |
| 6.4 | Explicit handoffs | Agent A delegates to Agent B with full context; handoff chain tracked in run state |
| 6.5 | Parallel branches | Graph supports fan-out / fan-in; branch results merged correctly |
| 6.6 | Async queue mode | Runs enqueued and processed by workers; results retrievable by `run_id` |

**Handoff patterns supported (per blueprint §7.4):**

| Pattern | Use Case |
|---------|---------|
| Explicit handoff | Clear task boundary delegation |
| Supervisor routing | Task classification and dispatch |
| Hierarchical | Complex multi-level decomposition |
| Parallel fan-out | Independent sub-problems |

### Exit Criteria

- [ ] Demo: 4-node workflow (intake → classify → fetch → answer) completes end-to-end
- [ ] Demo: supervisor delegates 2 parallel subtasks to specialists; merged result is correct
- [ ] Demo: Agent A hands off to Agent B mid-run; `handoff_chain` in state contains ≥2 entries
- [ ] Handoff context loss: zero fields dropped across handoff boundary (verified by schema diff)
- [ ] Eval: workflow task category achieves ≥80% success rate

---

## Stage 7 — Ecosystem

**Goal:** framework integrates with external tool ecosystems, agent protocols, provider APIs, and infrastructure

### Deliverables

| # | Item | Acceptance Criterion |
|---|------|----------------------|
| 7.1 | MCP adapter | Discover and invoke tools from ≥3 MCP servers via a single normalized interface |
| 7.2 | A2A adapter | Invoke remote agents via Agent-to-Agent Protocol; capabilities discovered via Agent Cards |
| 7.3 | Provider plugins | Swap LLM provider (OpenAI ↔ Anthropic ↔ local) via config change only — no code change |
| 7.4 | Vector store adapter | Pluggable retrieval backend; reference implementation for ≥1 vector DB |
| 7.5 | Sandbox adapter | Execute code/commands in isolated container; apply filesystem and network boundaries |
| 7.6 | Queue adapter | Pluggable async job queue; reference implementation for ≥1 queue backend |

### Exit Criteria

- [ ] Demo: agent discovers and invokes MCP filesystem server tool through adapter
- [ ] Demo: agent delegates sub-task to a remote agent via A2A; result returned through normalized `ToolResult`
- [ ] Provider swap test: same benchmark task run against 2 providers; scores within 5% of each other
- [ ] Each adapter interface has ≥1 documented reference implementation
- [ ] Sandbox: shell command execution cannot read files outside designated working directory

---

## Milestone Summary

| Milestone | Stages | Capability | Key Gate |
|-----------|--------|------------|----------|
| **M1 — Runnable Agent** | 0–2 | Durable single-agent runtime with real tools | Agent edits a file correctly; survives crash |
| **M2 — Governed Agent** | 3–4 | Safe, traceable, cost-accountable execution | Unsafe actions blocked; full replay from trace |
| **M3 — Intelligent Agent** | 5 | Context-aware across runs | Memory improves task performance by ≥10% |
| **M4 — Composable Platform** | 6–7 | Workflows, handoffs, ecosystem interop | Supervisor workflow + MCP/A2A integration live |

---

## Build Sequence

```
 Stage 0                   Stage 1                   Stage 2
 Foundation           ──►  Runtime Kernel        ──►  Tool Execution
 Schemas, CI               Loop, checkpoints,         5 tools,
 Eval scaffold             streaming events            structured results
       │                         │                         │
       └─────────────────────────┴─────────────────────────┘
                                 │
                           [M1: Runnable Agent]
                                 │
                                 ▼
 Stage 3                   Stage 4
 Guardrails & Policy  ──►  Observability & Evals
 Input/output guards,       Traces, cost tracking,
 Approvals, budgets,        Replay, eval harness,
 Prompt injection           Benchmark baseline
       │                         │
       └─────────────────────────┘
                                 │
                           [M2: Governed Agent]
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
         Stage 5                              Stage 6         Stage 7
         Memory                               Composition ──►  Ecosystem
         Episodic,                            Workflows,        MCP, A2A,
         semantic,                            Handoffs,         Providers,
         retrieval                            Multi-agent       Sandbox
              │                                    │               │
              ▼                                    └───────────────┘
      [M3: Intelligent Agent]                              │
                                                 [M4: Composable Platform]
```

Stages 5 and 6–7 can proceed in parallel once M2 is complete.

---

## End-State Success Metrics

| Metric | Target | Stage Set |
|--------|--------|-----------|
| Task success rate (benchmark suite) | ≥85% | 1–7 |
| Average steps per successful task | ≤8 | 1–2 |
| Checkpoint resume success rate | 100% | 1 |
| Unsafe action compliance (no unapproved tier-3+) | 100% | 3 |
| Guardrail effectiveness (injections blocked) | 100% | 3 |
| Trace replay fidelity | 100% | 4 |
| Tool call error rate | <5% | 2 |
| Cost tracked per run | Yes, all runs | 4 |
| Memory relevance improvement vs baseline | ≥10% | 5 |
| Handoff context loss | 0 fields | 6 |
| MCP servers integrated | ≥3 | 7 |
| Provider portability (config-only swap) | Verified | 7 |

---

## Anti-Patterns This Roadmap Guards Against

> Derived from blueprint §22

| Anti-Pattern | Addressed By |
|--------------|-------------|
| God agent — single agent doing everything | Stage 6 supervisor + handoff patterns |
| Implicit state in prompts | Stage 1 explicit state reducer |
| Unbounded loops | Stage 1 termination logic + Stage 3 budgets |
| Silent failures | Stage 2 structured errors + Stage 4 error taxonomy |
| Prompt-only safety | Stage 3 runtime guardrails + policy engine |
| Tool explosion | Stage 2 dynamic filtering + Stage 3 tool budgets |
| Memory soup | Stage 5 separated memory types with retrieval policies |
| Hope-based testing | Stage 4 eval harness in CI from Stage 1 |
| Provider lock-in | Stage 7 clean provider abstraction |

---

## Next Action

Begin **Stage 0**: initialize the repository structure, define and validate all 5 core schemas, and wire up CI.
