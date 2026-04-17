# Modern Agent Framework Blueprint

> **Purpose:** a refined blueprint for building a modern agentic framework with production-grade architecture, informed by current framework patterns
>
> **References considered:** LangChain, LangGraph, AutoGen, CrewAI, MCP, A2A (Agent-to-Agent Protocol), ReAct, Reflexion, CoALA, OpenAI Agents SDK, Anthropic Claude Computer Use, workflow/state-machine runtimes
>
> **Last updated:** 2026-04

---

## 1. Executive Direction

If you are building your own framework, the most durable architecture is:

> **A stateful orchestration runtime with typed tools, explicit policies, persistent execution, structured outputs, and a thin model layer.**

That is the consistent lesson from modern systems:

- **LangChain** provides high-level agent ergonomics, model/tool abstractions, and middleware
- **LangGraph** emphasizes durable execution, stateful orchestration, interrupts, persistence, and human-in-the-loop
- **AutoGen** separates chat-level agent composition from a lower-level event-driven core
- **CrewAI** demonstrates role-based agent coordination with task delegation patterns
- **OpenAI Agents SDK** shows the value of built-in tool calling, handoffs, and guardrails as first-class primitives
- **MCP** pushes the ecosystem toward a standard interface for tools, data sources, and external capabilities
- **A2A (Agent-to-Agent Protocol)** standardizes inter-agent communication for multi-agent orchestration across boundaries

The implication is clear:

**Your framework should not be "just an agent loop."** It should be a **runtime platform** for long-running, inspectable, policy-governed workflows with clear agent-to-agent and agent-to-tool boundaries.

---

## 2. What Modern Agent Frameworks Teach

### 2.1 LangChain pattern

LangChain's current pattern is:

- simple entry point for building agents quickly
- model abstraction across providers
- tool abstraction
- middleware for dynamic model/tool routing and policy injection
- agents that run until a final output or iteration limit

**Takeaway:** expose a high-level developer API, but keep it backed by a more explicit runtime.

### 2.2 LangGraph pattern

LangGraph's current value proposition is centered on:

- **durable execution**
- **streaming**
- **human-in-the-loop**
- **stateful orchestration**
- **memory**
- **deployment/operability**

**Takeaway:** agent frameworks become reliable when they are built on top of a graph or state-machine runtime, not when they rely on hidden prompt loops.

### 2.3 AutoGen pattern

AutoGen separates:

- a simpler conversational layer for agent prototyping
- a more serious event-driven core for scalable and multi-agent systems
- extensions for runtimes, tools, Docker execution, and distributed agents

**Takeaway:** keep your framework layered. Do not couple developer-facing chat APIs directly to the orchestration internals.

### 2.4 MCP pattern

MCP establishes a standard way to connect models and agents to:

- tools
- data sources
- external workflows

**Takeaway:** your tool layer should be interoperable. Build a native tool registry, but design an adapter layer so MCP servers can be consumed as first-class tools.

### 2.5 OpenAI Agents SDK pattern

OpenAI's Agents SDK demonstrates:

- **built-in primitives**: tool calling, handoffs, and guardrails as framework-level concepts
- **agent handoffs**: explicit mechanism for one agent to delegate to another
- **input/output guardrails**: validation and safety checks before/after agent execution
- **tracing built-in**: first-class observability without external instrumentation

**Takeaway:** handoffs, guardrails, and tracing should be framework primitives, not afterthoughts.

### 2.6 A2A (Agent-to-Agent Protocol) pattern

Google's Agent-to-Agent Protocol establishes:

- standardized agent discovery via "Agent Cards"
- task lifecycle management (create, execute, cancel)
- structured message passing between agents
- capability advertisement and negotiation

**Takeaway:** when building multi-agent systems, use standardized protocols for agent discovery and communication rather than ad-hoc integrations.

### 2.7 Anthropic Computer Use pattern

Anthropic's approach to computer use demonstrates:

- **vision-based interaction**: screenshots as primary feedback loop
- **atomic actions**: small, verifiable steps (click, type, scroll)
- **confirmation loops**: verify state before and after actions
- **graceful degradation**: fall back when actions fail

**Takeaway:** for UI/environment interaction, prefer vision+action loops with explicit verification over brittle automation scripts.

---

## 3. Design Principles

### 3.1 Runtime over prompt engineering

The framework's core value should be its execution model, not a clever prompt template.

### 3.2 Lowest effective agency

Use the least autonomous architecture that satisfies the task. High-agency systems should be intentional, not the default.

### 3.3 State first

Every meaningful agent system should have explicit state:

- run state
- step state
- tool state
- approval state
- memory state
- budget state
- handoff state (for multi-agent)

### 3.4 Schema-first contracts

Everything crossing a boundary should be typed and validated:

- tool inputs
- tool outputs
- planner outputs
- events
- checkpoints
- artifacts
- agent-to-agent messages

### 3.5 One-step execution, multi-step planning

Planning may be multi-step, but execution should usually be one step at a time with persistence after every step.

### 3.6 Middleware everywhere

Modern frameworks benefit from middleware/hooks for:

- dynamic model selection
- tool filtering
- policy enforcement
- observability
- retries
- redaction
- approval interrupts
- guardrails (input and output validation)

### 3.7 Human interruptibility

Human review should be a built-in state transition, not a bolted-on escape hatch.

### 3.8 Durable by default

A run should survive process crashes, retries, timeouts, and operator intervention.

### 3.9 Interoperable tools

Native tools are necessary, but external tool ecosystems should be adapter-friendly, especially MCP.

### 3.10 Evaluation before sophistication

Do not add memory, swarms, or reflection loops before you can measure whether they improve outcomes.

### 3.11 Structured outputs as default

Prefer structured (JSON schema-validated) outputs for all agent decisions, tool calls, and inter-agent communication. Free-form text should be reserved for final human-facing responses.

### 3.12 Graceful degradation

Agents should fail gracefully: retry with backoff, fall back to simpler strategies, and clearly surface failures rather than silently producing bad output.

---

## 4. Architectural Goal

The framework should solve this problem:

> How do we let an LLM reason over a task, use tools safely, persist progress, support interrupts, and produce auditable outcomes?

That requires five capabilities working together:

1. **orchestration**
2. **action execution**
3. **state and memory**
4. **policy and safety**
5. **observability and evaluation**

---

## 5. Reference Architecture

### 5.1 Layered stack

```text
+--------------------------------------------------------------+
| Developer Experience Layer                                   |
| SDK / CLI / API / declarative agent definitions              |
+--------------------------------------------------------------+
| Agent Composition Layer                                      |
| single-agent, supervisor-agent, workflow templates, handoffs |
+--------------------------------------------------------------+
| Orchestration Runtime                                        |
| graph engine, state machine, checkpoints, interrupts         |
+--------------------------------------------------------------+
| Guardrails Layer                                             |
| input validation, output validation, content filters         |
+--------------------------------------------------------------+
| Execution Layer                                              |
| model calls, tool calls, code execution, streaming           |
+--------------------------------------------------------------+
| State + Memory Layer                                         |
| run store, checkpoints, working memory, episodic memory      |
+--------------------------------------------------------------+
| Policy + Safety Layer                                        |
| authz, approvals, budgets, guardrails, sandbox boundaries    |
+--------------------------------------------------------------+
| Observability + Evals                                        |
| traces, metrics, replay, benchmark tasks, regressions        |
+--------------------------------------------------------------+
| Infra Adapters                                               |
| model providers, MCP, A2A, queues, DBs, vector stores        |
+--------------------------------------------------------------+
```

This is the main refinement over the earlier blueprint: the framework should be treated as a **layered platform**, not just a planner and some tools.

### 5.2 Control plane vs data plane

It helps to think in two planes.

### Control plane

Responsible for:

- run creation
- policy checks
- approvals
- scheduling
- checkpointing
- tracing
- replay

### Data plane

Responsible for:

- model inference
- tool execution
- retrieval calls
- file or command actions
- stream transport

This separation makes scaling and operations much easier later.

---

## 6. Core Runtime Model

### 6.1 State-machine orientation

A modern agent runtime should behave like a graph or state machine, even if the public API looks simple.

```text
START
  |
  v
LOAD_CONTEXT
  |
  v
PLAN_NEXT_ACTION
  |
  +--> NEED_HUMAN_APPROVAL ----> WAIT_FOR_APPROVAL
  |                                  |
  |                                  v
  |<------------- APPROVED / REJECTED
  |
  +--> EXECUTE_TOOL
  |        |
  |        v
  |    CAPTURE_OBSERVATION
  |        |
  |        v
  |    UPDATE_STATE
  |
  +--> GENERATE_FINAL_RESPONSE
  |
  v
CHECK_TERMINATION
  |
  +--> CONTINUE -> PLAN_NEXT_ACTION
  |
  +--> END
```

This is much closer to current production-grade frameworks than a hidden `while model says continue` loop.

### 6.2 Suggested runtime components

### A. Run manager

- create and resume runs
- assign run identifiers
- own lifecycle state
- enforce global budgets and limits

### B. Graph/workflow engine

- route between nodes
- persist transition results
- support branching and joins
- support interrupts and resume

### C. Planner node

- proposes next action or sub-plan
- may choose model, tool subset, and reasoning mode
- should return structured actions rather than free-form text

### D. Executor node

- invokes tools or emits final output
- normalizes errors
- records timing and resource usage

### E. Policy node

- blocks forbidden actions
- transforms available actions
- triggers human approval
- applies role/tenant/feature-based restrictions

### F. State reducer

- merges new observations into canonical run state
- updates step status
- updates memory summaries
- records artifacts

### G. Interrupt handler

- pauses execution safely
- exposes state for human inspection
- resumes from checkpoint

### H. Guardrails engine

- validates inputs before model/tool calls
- validates outputs before returning to user or next step
- applies content filters
- enforces schema compliance
- triggers rejection or retry on violation

### I. Handoff manager

- manages agent-to-agent delegation
- preserves context across handoffs
- tracks handoff chain for debugging
- enforces handoff policies (who can delegate to whom)

---

## 7. Agent Composition Model

Your framework should support three composition styles, but implement them in order.

### 7.1 Level 1: Single-agent loop

Best for:

- coding agents
- research agents
- operational assistants
- CRUD and workflow automation

This should be your first production-ready architecture.

### 7.2 Level 2: Directed workflows

A predefined graph with agentic nodes.

Best for:

- intake -> classify -> fetch -> validate -> answer
- compliance and business processes
- document processing pipelines

This is usually more reliable than free-form multi-agent systems.

### 7.3 Level 3: Multi-agent coordination

Supervisor, specialist, handoff, or swarm patterns.

Best for:

- domain decomposition
- independent sub-problems
- long-running collaborative tasks
- tasks requiring specialized expertise

Add this only after the single-agent and workflow runtime are stable.

**Modern recommendation:** prefer **workflow graphs with specialized nodes** over unrestricted agent swarms. Use explicit **handoffs** (like OpenAI Agents SDK) rather than implicit message passing when agents need to delegate.

### 7.4 Handoff patterns

When implementing multi-agent coordination, consider these handoff styles:

| Pattern | Description | Best for |
|---------|-------------|----------|
| **Explicit handoff** | Agent A explicitly delegates to Agent B with context | Clear task boundaries |
| **Supervisor routing** | Central agent routes to specialists | Task classification |
| **Round-robin** | Agents take turns processing | Iterative refinement |
| **Broadcast** | Message sent to all agents | Consensus/voting |
| **Hierarchical** | Tree of supervisors and workers | Complex decomposition |

---

## 8. Memory Architecture

Memory should be explicit and layered.

### 8.1 Working memory

Stored directly in run state:

- current messages
- current task
- current plan
- recent tool results
- active constraints

### 8.2 Episodic memory

Stores prior trajectories:

- previous runs
- failures and resolutions
- summarized outcomes
- reusable task traces

Useful for:

- reflection
- example retrieval
- operator debugging

### 8.3 Semantic memory

Stores durable facts:

- user preferences
- environment metadata
- project facts
- domain knowledge references

### 8.4 Artifact memory

Stores external artifacts:

- file patches
- command logs
- fetched documents
- generated reports

**Rule:** do not mix memory types in one opaque blob. Keep them separate and query them intentionally.

---

## 9. Tool Architecture

This is where many custom agent frameworks become fragile.

### 9.1 Tool interface contract

Each tool should expose:

- `name`
- `description`
- `input_schema` (JSON Schema)
- `output_schema` (JSON Schema)
- `side_effect_level` (read/write/destructive)
- `timeout`
- `retry_policy`
- `approval_policy`
- `idempotency`
- `capability_tags`
- `examples` (for few-shot prompting)
- `error_codes` (documented failure modes)

### 9.2 Tool categories

### Read tools

- search
- file read
- web fetch
- database query

### Transform tools

- summarize
- parse
- diff
- compile structured output

### Side-effect tools

- file write
- API mutation
- shell execution
- deployment operations

### Control tools

- request approval
- enqueue subtask
- emit checkpoint

### 9.3 Dynamic tool exposure

This is a strong lesson from LangChain's middleware-oriented model:

- do not expose every tool all the time
- filter tools by permissions, state, user role, task stage, or cost budget

Too many tools degrades planning quality.

### 9.4 MCP adapter strategy

Support multiple tool sources:

1. **native tools** (built into framework)
2. **MCP-backed tools** (via MCP protocol)
3. **A2A-backed agents** (other agents as tools via Agent-to-Agent protocol)

Recommended design:

- normalize all behind one internal tool interface
- attach provenance and transport metadata
- cache capability discovery
- apply the same policy layer to all sources

### 9.5 Tool result handling

Tool results should be:

- **structured**: prefer JSON over free-form text
- **bounded**: enforce output size limits
- **validated**: check against output_schema
- **annotated**: include metadata (timing, resource usage, warnings)
- **error-aware**: distinguish recoverable from fatal errors

---

## 10. Model Layer

Treat models as pluggable reasoning engines, not as the framework itself.

### 10.1 Model abstraction

Support:

- provider abstraction (OpenAI, Anthropic, local, etc.)
- static model selection
- dynamic model routing (based on task type, cost, latency)
- structured output mode (JSON schema enforcement)
- streaming mode
- timeout and retry policies
- fallback chains (try model A, fall back to B)

### 10.2 Reasoning modes

Separate model usage modes by intent:

- classification
- planning
- tool selection
- summarization
- final response generation
- self-critique or reflection
- structured extraction

This lets you optimize cost and reliability with the right model per stage.

### 10.3 Middleware around model calls

Middleware should be able to:

- swap models
- redact input (PII, secrets)
- inject policy hints
- limit toolset
- attach tracing metadata
- reject unsafe requests before inference
- cache responses (for deterministic queries)
- enforce rate limits

---

## 11. Persistence and Durability

Durability is a defining feature of modern agent frameworks.

### 11.1 Persist after every transition

At minimum, persist:

- run snapshot
- current node
- step status
- tool request
- tool result
- artifacts
- prompt/version metadata

### 11.2 Replayability

A run should be replayable for:

- debugging
- regression analysis
- audit
- evaluation

### 11.3 Resume semantics

You need clear rules for resuming after:

- process crash
- timeout
- operator stop
- approval wait
- network failure

---

## 12. Streaming and Interaction Model

Modern frameworks should assume interactive execution, not just batch execution.

Support:

- token streaming from model calls
- step-by-step event streaming
- tool lifecycle events
- interrupt/resume events
- partial result delivery

This matters for CLI, UI, and operator trust.

---

## 13. Safety and Governance

Safety should be a runtime concern, not just a prompt concern.

### 13.1 Policy engine responsibilities

- action allow/deny
- path and environment restrictions
- approval requirements
- tool quotas
- tenant or role restrictions
- network restrictions
- escalation policies
- data classification and handling rules

### 13.2 Guardrails (input/output validation)

Modern agent frameworks should implement guardrails at multiple points:

| Guardrail Type | When Applied | Purpose |
|----------------|--------------|---------|
| **Input guardrails** | Before agent processes user input | Reject malicious/invalid inputs |
| **Planning guardrails** | After plan generation | Validate proposed actions |
| **Tool input guardrails** | Before tool execution | Validate tool arguments |
| **Tool output guardrails** | After tool execution | Filter/validate results |
| **Response guardrails** | Before returning to user | Content safety, schema compliance |

### 13.3 Risk tiers

Assign a risk level to each action:

| Tier | Meaning                 | Examples                                  |
| ---- | ----------------------- | ----------------------------------------- |
| 0    | Safe read-only          | search, read file                         |
| 1    | Low-risk transform      | summarize, parse                          |
| 2    | Local side effects      | write file, generate patch                |
| 3    | Sensitive execution     | shell command, DB mutation                |
| 4    | External or destructive | deploy, delete, publish, network mutation |

Use these tiers to drive approval and sandbox policy.

### 13.4 Sandboxing

If your framework executes code or shell commands, assume you will eventually need:

- sandboxed execution (containers, VMs, WASM)
- filesystem boundaries (allowlists, deny patterns)
- network boundaries (egress rules, domain allowlists)
- resource quotas (CPU, memory, time limits)
- isolated credentials (per-run secrets)
- audit logging (all sandbox interactions)

### 13.5 Prompt injection defense

Defend against prompt injection at multiple layers:

- **input sanitization**: escape or reject suspicious patterns
- **context isolation**: separate system prompts from user data
- **output validation**: verify outputs match expected schemas
- **privilege separation**: untrusted data should not influence tool selection
- **monitoring**: detect anomalous agent behavior patterns

---

## 14. Observability

This is one of the biggest differences between demos and usable systems.

### 14.1 Required telemetry

- run traces (full execution path)
- state transitions
- model call metadata (tokens, latency, model version)
- tool call metadata (inputs, outputs, duration)
- latency (end-to-end and per-step)
- token usage (input, output, by model)
- cost tracking (per run, per step)
- error taxonomy (categorized failures)
- approval events
- resume events
- handoff events (for multi-agent)
- guardrail violations

### 14.2 Useful developer views

- execution timeline
- state diff by step
- tool call tree
- prompt/version comparison
- replay mode
- cost breakdown
- token usage heatmap

Frameworks like LangGraph/LangSmith are strong here for a reason: tracing is not optional once workflows become stateful.

### 14.3 Structured logging

Use structured logging (JSON) with consistent fields:

```json
{
  "run_id": "...",
  "step_id": "...",
  "event_type": "tool_call",
  "tool_name": "read_file",
  "duration_ms": 45,
  "status": "success",
  "timestamp": "2026-04-15T14:30:00Z"
}
```

---

## 15. Evaluation System

A serious framework needs an eval harness.

### 15.1 Evaluate the runtime, not only the model

You should measure:

- task success rate
- number of steps
- cost
- latency
- unsafe action rate
- approval frequency
- recovery success after failure
- loop/stall rate

### 15.2 Benchmark categories

- tool use correctness
- long-running resume
- policy compliance
- file editing tasks
- retrieval quality
- multi-step planning quality
- human-interrupt handling
- guardrail effectiveness
- handoff accuracy (for multi-agent)
- adversarial robustness (prompt injection resistance)

### 15.3 Evaluation-driven development

Adopt an eval-first mindset:

1. **Define success criteria** before building features
2. **Create benchmark tasks** that test the criteria
3. **Run evals continuously** (CI/CD integration)
4. **Track regressions** over model/framework changes
5. **A/B test** architectural decisions with evals

---

## 16. Recommended Repository Shape

```text
src/
  api/
    cli.py|ts
    http.py|ts
    stream.py|ts
  agent/
    definitions.py|ts
    factory.py|ts
    composition.py|ts
    handoffs.py|ts           # agent-to-agent delegation
  runtime/
    engine.py|ts
    graph.py|ts
    scheduler.py|ts
    interrupts.py|ts
    termination.py|ts
  planning/
    planner.py|ts
    step_builder.py|ts
    validators.py|ts
  execution/
    executor.py|ts
    tool_runner.py|ts
    model_runner.py|ts
    stream_events.py|ts
  tools/
    base.py|ts
    registry.py|ts
    adapters/
      native.py|ts
      mcp.py|ts
      a2a.py|ts              # Agent-to-Agent protocol adapter
    builtin/
      read_file.py|ts
      write_file.py|ts
      search_code.py|ts
      run_command.py|ts
      fetch_url.py|ts
  state/
    models.py|ts
    reducers.py|ts
    checkpoints.py|ts
    store.py|ts
    events.py|ts
  memory/
    working.py|ts
    episodic.py|ts
    semantic.py|ts
    retrieval.py|ts
  policy/
    engine.py|ts
    approvals.py|ts
    risk.py|ts
    authz.py|ts
  guardrails/                # input/output validation
    engine.py|ts
    input.py|ts
    output.py|ts
    content_filter.py|ts
    schema_validator.py|ts
  observability/
    tracing.py|ts
    metrics.py|ts
    replay.py|ts
    cost_tracker.py|ts       # token/cost accounting
  evals/
    fixtures/
    runner.py|ts
    scoring.py|ts
    regressions.py|ts
    benchmarks/              # standard benchmark tasks
  providers/
    llm/
    vector/
    queue/
    sandbox/
```

---

## 17. Core Runtime Contracts

### 17.1 Run

```json
{
  "run_id": "string",
  "agent_id": "string",
  "goal": "string",
  "status": "pending|running|waiting_approval|completed|failed|cancelled|blocked|handoff",
  "current_node": "string",
  "current_step_id": "string|null",
  "parent_run_id": "string|null",
  "max_steps": 20,
  "budgets": {
    "tokens": 0,
    "tool_calls": 0,
    "wall_clock_seconds": 0,
    "cost_usd": 0
  },
  "context": {},
  "artifacts": [],
  "handoff_chain": []
}
```

### 17.2 Planned action

```json
{
  "step_id": "string",
  "kind": "tool|answer|approval|subgraph|handoff",
  "title": "string",
  "tool_name": "string|null",
  "target_agent": "string|null",
  "arguments": {},
  "risk_tier": 0,
  "depends_on": [],
  "status": "pending|in_progress|done|failed|blocked",
  "guardrails_passed": true
}
```

### 17.3 Tool result

```json
{
  "tool_name": "string",
  "ok": true,
  "output": {},
  "error": null,
  "duration_ms": 0,
  "artifacts": [],
  "metadata": {
    "provider": "native|mcp|a2a|other",
    "tokens_used": 0,
    "cost_usd": 0
  }
}
```

### 17.4 Event

```json
{
  "event_id": "string",
  "run_id": "string",
  "type": "run_started|state_loaded|plan_emitted|policy_checked|guardrail_checked|tool_called|tool_completed|approval_requested|approval_resolved|handoff_initiated|handoff_completed|state_persisted|run_completed|run_failed",
  "timestamp": "iso-8601",
  "payload": {}
}
```

### 17.5 Guardrail result

```json
{
  "guardrail_id": "string",
  "type": "input|output|planning|tool",
  "passed": true,
  "violations": [],
  "action_taken": "allow|block|modify|escalate",
  "timestamp": "iso-8601"
}
```

These contracts should be schema-validated and versioned.

---

## 18. Recommended MVP

The right MVP is still intentionally narrow, but more modern than a bare loop.

### Build this first

- single-agent runtime
- graph/state-machine execution core
- persistent checkpoints
- streaming event bus
- schema-first tool registry
- dynamic tool filtering
- policy engine with approval gates
- basic guardrails (input/output validation)
- 5 core tools
- trace viewer or replay logs
- cost tracking

### Do not build first

- unrestricted multi-agent swarms
- autonomous recursive self-improvement
- unbounded long-term memory
- too many tools
- provider-specific logic scattered through the codebase
- complex A2A federation (get single-agent right first)

---

## 19. Implementation Roadmap

### Phase 1: Runtime kernel

Build:

- run model
- graph/state engine
- step transitions
- checkpoint persistence
- termination rules
- event stream

**Outcome:** runs are durable, resumable, and inspectable.

### Phase 2: Tooling and execution

Build:

- tool registry
- native tools
- executor
- structured tool results
- timeout/retry behavior

**Outcome:** the agent can safely perform grounded work.

### Phase 3: Policy and approvals

Build:

- risk tiers
- policy engine
- approval interrupts
- dynamic tool filtering
- basic guardrails

**Outcome:** the runtime can govern high-risk behavior.

### Phase 4: Observability and replay

Build:

- traces
- metrics
- cost tracking
- state diffs
- replay tooling

**Outcome:** failures become diagnosable.

### Phase 5: Memory and retrieval

Build:

- episodic summaries
- semantic store
- retrieval policies

**Outcome:** memory becomes intentional, not accidental.

### Phase 6: Composition and handoffs

Build:

- workflow templates
- subgraphs
- supervisor pattern
- explicit handoffs
- queue-backed async execution

**Outcome:** the framework can scale beyond one loop without becoming opaque.

### Phase 7: Ecosystem adapters

Build:

- MCP adapters
- A2A adapters (Agent-to-Agent protocol)
- provider plugins
- sandbox adapters
- queue/store integrations

**Outcome:** the framework becomes extensible without rewriting the core.

---

## 20. Practical Recommendation

If your goal is a serious modern agent framework, optimize for this sequence:

1. **stateful runtime**
2. **typed tools**
3. **guardrails**
4. **policy engine**
5. **durability and replay**
6. **observability + cost tracking**
7. **memory**
8. **handoffs and multi-agent composition**

That sequence matches what current successful frameworks implicitly teach: **orchestration quality matters more than autonomy theater**.

---

## 21. Final Position

The modern architecture for an agentic framework is:

> **a layered, graph-oriented, durable execution system with model and tool abstractions, middleware, guardrails, policy enforcement, memory boundaries, observability, cost tracking, and interoperable tool/agent adapters.**

If you build that foundation first, you can later support:

- LangChain-style high-level agent ergonomics
- LangGraph-style durable orchestration
- AutoGen-style event-driven multi-agent composition
- OpenAI Agents SDK-style handoffs and guardrails
- MCP-style tool interoperability
- A2A-style agent-to-agent federation

Without that foundation, you will likely end up with a prompt loop that looks impressive in demos but is difficult to govern, debug, scale, or trust.

---

## 22. Anti-Patterns to Avoid

| Anti-Pattern | Why It's Problematic | Better Alternative |
|--------------|---------------------|-------------------|
| **God agent** | Single agent doing everything | Specialized agents with clear boundaries |
| **Implicit state** | State buried in prompts | Explicit state machine |
| **Unbounded loops** | Agent runs forever | Step limits, cost budgets |
| **Silent failures** | Errors swallowed | Structured error handling + telemetry |
| **Prompt-only safety** | "Please don't do X" | Runtime policy enforcement |
| **Tool explosion** | 50+ tools available | Dynamic filtering, capability tiers |
| **Memory soup** | All context in one blob | Separated memory types |
| **Hope-based testing** | "It worked once" | Systematic evals + regression suite |
| **Synchronous everything** | Blocking on all calls | Streaming + async where appropriate |
| **Provider lock-in** | Code tied to one LLM | Clean provider abstraction |

---

## 23. Emerging Patterns to Watch

- **Agentic RAG**: Agents that iteratively refine retrieval queries
- **Tool learning**: Agents that discover/create tools at runtime
- **Collaborative agents**: Agents that negotiate and coordinate (beyond simple handoffs)
- **Self-improving agents**: Agents that update their own prompts/strategies based on outcomes
- **Multimodal agents**: Vision, audio, and text in unified workflows
- **Confidential agents**: Privacy-preserving agent execution (encrypted state, secure enclaves)
