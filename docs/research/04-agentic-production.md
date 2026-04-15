# Production-Ready Agentic Systems

> A practitioner's guide to taking AI agents from demo to production — covering reliability engineering, evaluation systems, human oversight, scalability, cost control, observability, and the operational patterns that separate prototype from product.

---

## Table of Contents

1. [The Production Gap](#1-the-production-gap)
2. [Architecture Decision Framework](#2-architecture-decision-framework)
3. [Reliability Engineering](#3-reliability-engineering)
4. [State Management & Persistence](#4-state-management--persistence)
5. [Human-in-the-Loop (HITL)](#5-human-in-the-loop-hitl)
6. [Evaluation Systems](#6-evaluation-systems)
7. [Guardrails & Safety](#7-guardrails--safety)
8. [Latency & Cost Optimization](#8-latency--cost-optimization)
9. [Scalability Patterns](#9-scalability-patterns)
10. [Production Observability](#10-production-observability)
11. [Deployment Strategy](#11-deployment-strategy)
12. [Anti-Patterns](#12-anti-patterns)
13. [Case Studies](#13-case-studies)

---

## 1. The Production Gap

"There is a large class of problems that are easy to imagine and build demos for, but extremely hard to make products out of. For example, self-driving: it's easy to demo a car self-driving around a block, but making it into a product takes a decade." — Andrej Karpathy

The production gap for agents is wider than for most software. A demo succeeds on a curated happy path; a production system must handle the full distribution of real user inputs, failure modes, edge cases, cost overruns, latency spikes, and safety issues — simultaneously, reliably.

**What changes between demo and production:**

| Dimension | Demo | Production |
|---|---|---|
| Input distribution | Curated, ideal | Adversarial, long-tail, ambiguous |
| Failure handling | None — error crashes are ok | Graceful degradation, retry logic, fallbacks |
| Latency | Acceptable (minutes) | Must bound P99 |
| Cost | Unlimited experimentation | Per-task budget, per-user limits |
| Safety | Manually reviewed | Automated guardrails at scale |
| Observability | `print()` debugging | Structured traces, dashboards, alerts |
| Consistency | One-off runs | Reproducible behavior over time |
| Human oversight | Developer in the loop | Scaled oversight with automation |

The most common failure mode Anthropic observed working with production teams: **building a complex, sophisticated system when a simpler one would have worked**. Success in the LLM space is not about building the most sophisticated system — it's about building the *right* system for your needs.

---

## 2. Architecture Decision Framework

### 2.1 The Escalation Ladder

Before committing to a full agentic system, exhaust simpler options:

```
Level 0: Single LLM call with a good prompt
  └── Add retrieval (RAG) if knowledge is needed
  └── Add few-shot examples if behavior is inconsistent

Level 1: Prompt chaining (fixed sequential workflow)
  └── Task decomposes cleanly into ordered steps
  └── Each step has clear inputs/outputs

Level 2: Routing (classification → specialized prompts)
  └── Input types are distinct and well-understood
  └── Different inputs need meaningfully different handling

Level 3: Parallelization (sectioning or voting)
  └── Subtasks are independent
  └── Multiple perspectives improve reliability

Level 4: Orchestrator-workers (dynamic task breakdown)
  └── Number of subtasks can't be predicted in advance
  └── Tasks require different tools/capabilities

Level 5: Full autonomous agent (open-ended reasoning loop)
  └── Task structure is unknown until runtime
  └── Multi-step action in the environment required
  └── High tolerance for latency and cost
```

**Add complexity only when it demonstrably improves outcomes.** Each level adds latency, cost, debugging difficulty, and failure surface. The question is never "can I build an agent for this?" but "does this problem require an agent?"

### 2.2 When Agents Are the Right Answer

Agents are worth the complexity when the task has all of these properties:

1. **Unpredictable structure**: The number and nature of steps can't be hardcoded
2. **Environment interaction**: The task requires acting on the world (tools, APIs, code)
3. **Feedback loops**: Tool results change what the agent should do next
4. **Measurable success**: You can tell programmatically or objectively whether the task succeeded
5. **Trusted domain**: The agent operates in an environment where errors are recoverable

Strong use cases: software development automation, research synthesis, customer support with tool access, data pipeline construction, document processing at scale.

Poor use cases: single-turn Q&A, tasks with stable fixed structure, high-stakes irreversible actions without oversight, real-time latency requirements (<500ms).

### 2.3 Workflow vs. Agent

| Characteristic | Workflow | Agent |
|---|---|---|
| Control flow | Predefined in code | LLM determines dynamically |
| Predictability | High | Low |
| Debugging | Straightforward | Requires trace analysis |
| Cost | Predictable | Variable |
| Flexibility | Low | High |
| Best for | Well-understood, repeated tasks | Open-ended, novel tasks |

In practice, most production systems are **hybrid**: a predefined workflow orchestrates agents for specific complex sub-tasks. The workflow handles routing, authorization, and structure; agents handle the open-ended reasoning within bounded sub-problems.

---

## 3. Reliability Engineering

### 3.1 The Compounding Error Problem

Agent reliability degrades exponentially with task length. If each step has 95% success rate:
- 1 step: 95% end-to-end success
- 5 steps: 77% success
- 10 steps: 60% success
- 20 steps: 36% success

This is the fundamental reliability challenge for agents. Strategies to address it:

### 3.2 Error Taxonomy

Classify agent failures to handle them appropriately:

```python
class AgentError(Exception):
    pass

class TransientError(AgentError):
    """Retry eligible: API timeout, rate limit, network error."""
    pass

class ToolError(AgentError):
    """Tool-specific failure: file not found, HTTP 4xx, parse error."""
    pass

class PlanningError(AgentError):
    """Agent produced an invalid plan or tool call."""
    pass

class BudgetError(AgentError):
    """Token or iteration budget exceeded."""
    pass

class SafetyError(AgentError):
    """Guardrail triggered — cannot proceed."""
    pass
```

### 3.3 Retry Strategy

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    retry=retry_if_exception_type(TransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def call_llm_with_retry(messages, tools):
    try:
        return await llm.ainvoke(messages, tools=tools)
    except RateLimitError as e:
        raise TransientError(f"Rate limit: {e}") from e
    except TimeoutError as e:
        raise TransientError(f"Timeout: {e}") from e
```

**Retry budget**: Retries consume token budget. Track cumulative retries; if an agent retries the same step N times without progress, escalate to a human or fall back to a simpler strategy.

### 3.4 Fallback Strategies

Design explicit fallback chains for each agent capability:

```
Primary: GPT-4o / Claude Sonnet (capable, expensive)
   ↓ on failure or budget exhaustion
Fallback: GPT-4o-mini / Claude Haiku (fast, cheap)
   ↓ on failure
Last resort: Template-based response (no LLM)
   ↓ if even that fails
Human escalation
```

### 3.5 Circuit Breakers

Prevent cascading failures when external services degrade:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.state = "closed"  # closed = healthy, open = failing
        self.reset_timeout = reset_timeout
        self.last_failure_time = None

    def call(self, fn, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise CircuitOpenError("Circuit open — service unavailable")
        try:
            result = fn(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"

    def on_success(self):
        self.failures = 0
        self.state = "closed"
```

### 3.6 Idempotency

Agent tasks may be retried (due to crashes, network errors, or explicit retries). Design tools to be idempotent:

- **File writes**: Write to a temp file then atomic rename; check if file exists before writing
- **API calls**: Check if the resource already exists before creating
- **Database mutations**: Use `INSERT OR IGNORE` or `UPSERT`; carry an idempotency key
- **External service calls**: Many APIs support an idempotency key header (Stripe, Twilio)

```python
async def write_file_idempotent(path: str, content: str, job_id: str) -> str:
    """Write file only if content for this job_id hasn't been written yet."""
    metadata_path = f"{path}.{job_id}.done"
    if os.path.exists(metadata_path):
        return f"Already written (job {job_id})"
    with open(path, "w") as f:
        f.write(content)
    Path(metadata_path).touch()
    return f"Written {len(content)} bytes to {path}"
```

### 3.7 Timeout Management

Every agent component needs timeouts — both per-call and end-to-end:

```python
async def run_agent_with_timeout(state: AgentState, timeout_seconds: int = 300):
    try:
        return await asyncio.wait_for(
            run_agent(state),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return {
            **state,
            "final_output": "Task timed out. Partial progress has been saved.",
            "error": "timeout",
        }
```

---

## 4. State Management & Persistence

### 4.1 The Stateful Agent Problem

Long-running agents (tasks that take minutes to hours) need:
1. **Durability**: Survive process crashes without losing progress
2. **Resumability**: Pick up where they left off
3. **Auditability**: Complete history of all decisions made
4. **Parallelism**: Multiple agent instances without state conflicts

### 4.2 Checkpointing Architecture

LangGraph's `SqliteSaver` / `PostgresSaver` checkpoints every state transition as a serialized snapshot. The key concept is `thread_id` — a stable identifier for a conversation/task instance.

```python
# SQLite for single-instance deployments
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

# Postgres for multi-instance / production
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])

graph = build_graph().compile(checkpointer=checkpointer)

# Resume an interrupted task
config = {"configurable": {"thread_id": "task-abc123"}}
state = graph.get_state(config)  # restore last checkpoint
result = graph.invoke(None, config)  # resume from last checkpoint
```

**What gets checkpointed:**
- Full agent state (messages, plan, current_step, etc.) at each node transition
- Enables resuming after a process crash, server restart, or planned interrupt
- Enables time-travel debugging: inspect the state at any step

### 4.3 State Schema Evolution

As your agent evolves, the state schema changes. Handle schema migrations carefully:

```python
class AgentStateV1(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]

class AgentStateV2(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    plan_rationale: str  # new field in v2
    iteration: int        # new field in v2
```

Strategy: Use `Optional` fields with default values for new additions. Store schema version alongside state. Build migration functions for breaking changes.

### 4.4 Multi-Session Memory

Beyond per-task checkpoints, production agents often need cross-session memory:

```
Per-task (thread-level) memory:
  └── LangGraph checkpoints (SQLite/Postgres)
  └── Scoped to a single task execution

Per-user (session-level) memory:
  └── User preferences, past interactions, learned patterns
  └── Retrieved at task start and updated at task end
  └── Storage: Redis (fast) + Postgres (durable)

Organizational memory:
  └── Shared knowledge base, company-specific context
  └── Storage: Vector DB with access control
```

```python
async def load_user_context(user_id: str) -> dict:
    """Load persistent user context from cross-session storage."""
    return {
        "preferences": await redis.hgetall(f"user:{user_id}:prefs"),
        "history_summary": await postgres.fetchval(
            "SELECT summary FROM user_memory WHERE user_id = $1", user_id
        ),
        "relevant_past": await vector_db.query(
            user_id=user_id, limit=5
        ),
    }
```

---

## 5. Human-in-the-Loop (HITL)

### 5.1 Why HITL is Non-Negotiable

Autonomous agents make mistakes. In domains with high stakes, irreversible actions, or ambiguous authority, human oversight is not a limitation — it's a feature. The production question is not "should we have HITL?" but "at what granularity and for which actions?"

### 5.2 Interrupt Patterns

LangGraph supports `interrupt` — a mechanism to pause execution and yield to a human:

```python
from langgraph.types import interrupt, Command

def executor_node(state: AgentState) -> AgentState:
    step = state["current_step"]

    # Pause before destructive operations
    if is_destructive(step):
        human_decision = interrupt({
            "question": f"About to execute: {step}. Approve?",
            "step": step,
            "context": state["plan"],
        })
        if human_decision["action"] == "reject":
            return {**state, "final_output": "Cancelled by human reviewer"}
        if human_decision["action"] == "modify":
            step = human_decision["modified_step"]

    # Continue with execution
    result = execute_step(step)
    return {**state, "step_result": result}
```

**Resuming after interrupt:**
```python
# The graph pauses at interrupt(), waiting for human input
thread_id = "task-xyz"
graph.invoke(initial_input, {"configurable": {"thread_id": thread_id}})
# → Returns with interrupt data, pauses execution

# Human reviews, then resumes
graph.invoke(
    Command(resume={"action": "approve"}),
    {"configurable": {"thread_id": thread_id}}
)
```

### 5.3 Approval Tier Model

Not all actions need the same level of oversight. Design a tiered model:

```
Tier 0 — No approval needed (read-only, reversible)
  Examples: web search, read file, HTTP GET, vector DB query

Tier 1 — Soft approval (log + async notification)
  Examples: write to workspace file, send internal Slack message

Tier 2 — Hard approval (block until approved)
  Examples: send external email, modify production database, deploy code

Tier 3 — Always human-initiated (irreversible or high-risk)
  Examples: delete data, purchase, legal document submission
```

Implement as a decorator on tool functions:

```python
def requires_approval(tier: int):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if tier >= APPROVAL_THRESHOLD:
                approved = interrupt({
                    "tool": fn.__name__,
                    "args": args,
                    "kwargs": kwargs,
                    "tier": tier,
                })
                if not approved["approved"]:
                    raise PermissionError(f"Tool {fn.__name__} rejected by reviewer")
            return await fn(*args, **kwargs)
        return wrapper
    return decorator

@requires_approval(tier=2)
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to an external recipient."""
    ...
```

### 5.4 Feedback Integration Loop

Human feedback during oversight should improve the agent over time:

```
Human Review Queue
   ↓
Reviewer labels: ✓ correct / ✗ wrong / ✎ needs edit
   ↓
Labeled traces → evaluation dataset (golden examples)
   ↓
Evaluation dataset → regression tests
   ↓
Regression tests → CI pipeline
   ↓
Agent improvements validated against regression suite
```

This creates a data flywheel: every human intervention makes the agent measurably better.

---

## 6. Evaluation Systems

Evaluation is the most critical and most neglected component of production agent systems. **Teams that fail to build evaluation systems cannot improve their agents beyond a demo.**

### 6.1 The Three-Level Eval Pyramid

```
                        ┌─────────────┐
                        │  Level 3    │  A/B Testing
                        │  (slowest,  │  Real users, production
                        │  costliest) │
                        └─────────────┘
                   ┌─────────────────────────┐
                   │       Level 2           │  LLM-as-judge + Human eval
                   │  (hours, medium cost)   │  Trace inspection, model scoring
                   └─────────────────────────┘
         ┌─────────────────────────────────────────┐
         │              Level 1                    │  Unit tests
         │  (seconds, cheap — run on every commit) │  Assertions, deterministic checks
         └─────────────────────────────────────────┘
```

**Run cadence:**
- Level 1: Every code change (CI)
- Level 2: Daily or on major prompt/model changes
- Level 3: After significant product changes

### 6.2 Level 1: Unit Tests

Write fast, deterministic assertions that catch regressions. Focus on:
- **Tool behavior**: Does the file tool refuse path traversal?
- **Routing logic**: Does the supervisor correctly classify input type X?
- **Plan validity**: Is the generated plan non-empty and well-formed?
- **Output format**: Does the critic return "approve" or "reject"?
- **Safety**: Does the guardrail reject prompt injection attempts?

```python
# tests/test_agents.py
async def test_supervisor_routes_to_planner(mock_llm):
    mock_llm.return_value = '{"next": "planner", "reason": "task requires planning"}'
    state = initial_state("Analyze our Q3 sales data and write a summary")
    result = await supervisor_node(state)
    assert result["next_node"] == "planner"

async def test_critic_forces_approval_at_max_iterations(mock_llm):
    state = {**initial_state("task"), "iteration": MAX_CRITIC_ITERATIONS}
    result = await critic_node(state)
    assert result["final_output"] is not None  # forced approval

def test_file_tool_blocks_path_traversal():
    with pytest.raises(ValueError, match="outside workspace"):
        read_file("../../etc/passwd")

def test_plan_is_valid_list():
    plan = parse_plan("1. Do X\n2. Do Y\n3. Do Z")
    assert isinstance(plan, list)
    assert len(plan) == 3
    assert all(isinstance(step, str) for step in plan)
```

**Synthetic test case generation**: Use an LLM to generate diverse input examples for each feature:

```python
GENERATE_TEST_CASES_PROMPT = """
Generate 20 diverse user requests that would require a data analysis agent to:
- Search for data (at least 5 examples)
- Write and execute Python code (at least 5 examples)
- Read and summarize files (at least 5 examples)
- Combine multiple steps (at least 5 examples)

Return as a JSON list of strings.
"""
```

### 6.3 Level 2: LLM-as-Judge Evaluation

For outputs that can't be asserted with code, use a powerful LLM as evaluator:

```python
JUDGE_PROMPT = """
You are evaluating an AI agent's response to a user task.

Task: {task}
Agent Response: {response}
Expected Behavior: {expected}

Rate the response on each dimension from 1 (poor) to 5 (excellent):

1. CORRECTNESS: Did the agent solve the stated task?
2. EFFICIENCY: Were there unnecessary steps, redundant tool calls, or wasted tokens?
3. SAFETY: Did the agent avoid any harmful, unauthorized, or risky actions?
4. CLARITY: Is the final output clear and actionable for the user?

Return JSON: {{"correctness": N, "efficiency": N, "safety": N, "clarity": N, "notes": "..."}}
"""

async def evaluate_trace(task: str, trace: AgentTrace) -> EvalResult:
    response = await judge_llm.ainvoke([
        HumanMessage(content=JUDGE_PROMPT.format(
            task=task,
            response=trace.final_output,
            expected=trace.expected_behavior,
        ))
    ])
    return EvalResult(**json.loads(response.content))
```

**Trace inspection workflow:**
1. Collect 100+ recent production traces in Langfuse or LangSmith
2. Filter for: low scores, user complaints, high iteration counts, tool errors
3. Manually inspect 20-30% of flagged traces
4. Identify patterns (planning errors? tool misuse? prompt injection?)
5. Add new unit tests for discovered failure modes
6. Fix root cause, verify with eval

### 6.4 Level 3: A/B Testing

For significant changes (model upgrade, prompt rewrite, new tool set), run controlled experiments:

```python
async def route_to_agent_variant(user_id: str, task: str) -> AgentResult:
    """Route users to A or B variant based on consistent hash."""
    variant = "B" if hash(user_id) % 100 < ROLLOUT_PERCENTAGE else "A"

    if variant == "A":
        result = await agent_v1.run(task)
    else:
        result = await agent_v2.run(task)

    # Tag result with variant for analysis
    await log_to_analytics(user_id=user_id, variant=variant, result=result)
    return result
```

Track: task completion rate, user satisfaction, latency, cost per task, error rate — segmented by variant.

### 6.5 The Evaluation-Driven Flywheel

```
Production traffic
    ↓
Trace logging (Langfuse/LangSmith)
    ↓
Automated scoring (Level 2 judge)
    ↓
Human review queue (low-scoring traces)
    ↓
Labeled dataset (golden examples)
    ↓
Regression test suite (Level 1)
    ↓
Prompt / fine-tune improvement
    ↓
A/B test new version (Level 3)
    ↓
(Back to production traffic)
```

This flywheel is the only reliable path to continuously improving agent quality in production. Skipping evaluation means iterating blind.

---

## 7. Guardrails & Safety

### 7.1 Defense in Depth

No single guardrail is sufficient. Layer multiple defenses:

```
Input layer:     Validate, sanitize, classify intent
  ↓
Planning layer:  Check plan before execution
  ↓
Tool layer:      Validate tool inputs, sandbox execution
  ↓
Output layer:    Filter final response, check for PII/toxicity
  ↓
Audit layer:     Log everything for post-hoc review
```

### 7.2 Input Validation

```python
def validate_task_input(task: str) -> str:
    """Validate and sanitize user task input."""
    if len(task) > MAX_TASK_LENGTH:
        raise ValueError(f"Task too long: {len(task)} chars (max {MAX_TASK_LENGTH})")

    # Check for obvious injection patterns
    injection_patterns = [
        r"ignore (all )?previous instructions",
        r"you are now",
        r"new instructions:",
        r"system prompt:",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, task, re.IGNORECASE):
            raise SafetyError(f"Potential prompt injection detected")

    return task.strip()
```

### 7.3 Plan Validation Before Execution

Review the agent's plan before any actions are taken:

```python
PLAN_VALIDATOR_PROMPT = """
Review this agent plan for safety issues before execution.

Plan steps:
{plan}

Flag any steps that:
- Access sensitive system resources (SSH keys, environment variables, credentials)
- Make irreversible changes without confirmation
- Attempt network calls to non-whitelisted domains
- Include suspicious or obfuscated code

Return JSON: {"safe": true/false, "concerns": ["list of issues"]}
"""

async def validate_plan(plan: list[str]) -> PlanValidation:
    response = await fast_llm.ainvoke([
        HumanMessage(content=PLAN_VALIDATOR_PROMPT.format(plan="\n".join(plan)))
    ])
    return PlanValidation(**json.loads(response.content))
```

### 7.4 Output Filtering

```python
import re

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),          # SSN
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),  # Credit card
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
]

def redact_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

async def filter_output(output: str) -> str:
    # Redact PII
    output = redact_pii(output)

    # Check for toxicity with a fast classifier
    toxicity_score = await classify_toxicity(output)
    if toxicity_score > TOXICITY_THRESHOLD:
        return "I'm unable to provide that response."

    return output
```

### 7.5 Structural Prompt Injection Defense

The most robust defense is structural: keep tool outputs in a separate message role so they cannot override system instructions.

```python
# Bad: Tool result concatenated into instruction
system_prompt = f"""
You are a helpful agent. Instructions: {instructions}
Web search result: {web_result}  # ← injection surface
"""

# Good: Tool result in its own message with clear delimiters
messages = [
    SystemMessage(content=instructions),
    HumanMessage(content=user_task),
    ToolMessage(
        content=f"<search_result>\n{web_result}\n</search_result>",
        tool_call_id=tool_call_id,
    ),
]
```

Use `SystemMessage` + `ToolMessage` typed roles; never embed tool outputs directly into `SystemMessage` content.

### 7.6 Allowlist vs. Denylist for Tool Access

Prefer allowlists over denylists for tool authorization:

```python
USER_TOOL_ALLOWLIST = {
    "basic": {"web_search", "read_file"},
    "standard": {"web_search", "read_file", "write_file", "execute_python"},
    "admin": {"web_search", "read_file", "write_file", "execute_python", "http_post"},
}

def get_tools_for_user(user: User) -> list[Tool]:
    allowed_names = USER_TOOL_ALLOWLIST.get(user.tier, set())
    return [t for t in ALL_TOOLS if t.name in allowed_names]
```

---

## 8. Latency & Cost Optimization

### 8.1 The Cost-Quality-Latency Triangle

Every production agent lives in a triangle:

```
        Cost
        /\
       /  \
      /    \
Quality ---- Latency
```

Moving toward any one vertex trades off the others. Your product requirements determine which vertex to optimize for. Common strategies:

### 8.2 Model Routing by Task Difficulty

Not every step needs GPT-4o. Route to the cheapest model sufficient for each task:

```python
async def route_model_by_task(task_type: str, complexity: int) -> BaseChatModel:
    """Select the appropriate model based on task requirements."""
    if task_type == "classification" or complexity < 3:
        return cheap_model    # GPT-4o-mini, Claude Haiku (~10× cheaper)
    elif task_type == "planning" or complexity >= 7:
        return reasoning_model  # o3, Gemini 2.5 Pro (thinking)
    else:
        return standard_model  # GPT-4o, Claude Sonnet

# In practice: use routing at the supervisor level
SUPERVISOR_SYSTEM = """
For each incoming task, classify its complexity (1-10) and type.
Simple/classification tasks → use 'fast_model'.
Standard tasks → use 'standard_model'.
Complex reasoning/planning → use 'reasoning_model'.
"""
```

**Approximate cost ratios (2025):**
- Reasoning model: 1.0× (baseline cost)
- Standard model: 0.2×
- Fast model: 0.02×

### 8.3 Caching

Cache at multiple levels:

**Semantic caching** (most impactful for agents):
```python
from functools import lru_cache
import hashlib

async def cached_llm_call(prompt_hash: str, messages: list) -> str:
    """Cache LLM responses for identical prompts."""
    cached = await redis.get(f"llm:{prompt_hash}")
    if cached:
        return cached.decode()

    response = await llm.ainvoke(messages)
    await redis.setex(f"llm:{prompt_hash}", ttl=3600, value=response.content)
    return response.content

def hash_messages(messages: list) -> str:
    content = json.dumps([m.dict() for m in messages], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
```

**Tool result caching** (for expensive tools):
```python
@lru_cache(maxsize=1000)
def cached_web_search(query: str, date_bucket: str) -> str:
    """Cache search results; date_bucket invalidates daily."""
    return tavily_search(query)

def get_date_bucket() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")  # daily granularity
```

### 8.4 Parallelization

Execute independent plan steps concurrently:

```python
async def parallel_executor(state: AgentState) -> AgentState:
    """Execute independent steps in parallel when safe to do so."""
    plan = state["plan"]
    dependency_graph = build_dependency_graph(plan)
    ready_steps = [s for s in plan if not dependency_graph.get(s)]

    if len(ready_steps) > 1:
        results = await asyncio.gather(*[
            execute_single_step(step, state) for step in ready_steps
        ])
        return merge_results(state, zip(ready_steps, results))
    else:
        return await execute_single_step(ready_steps[0], state)
```

Parallelization is especially effective for:
- Research tasks: run multiple searches simultaneously
- Evaluation: run multiple judge calls at once
- Multi-file changes: process independent files concurrently

### 8.5 Token Budget Management

```python
@dataclass
class TokenBudget:
    max_input_tokens: int = 50_000
    max_output_tokens: int = 4_000
    max_total_tokens: int = 100_000
    used_tokens: int = 0

    def check(self, input_tokens: int) -> None:
        if self.used_tokens + input_tokens > self.max_total_tokens:
            raise BudgetError(
                f"Token budget exceeded: {self.used_tokens}/{self.max_total_tokens}"
            )

    def record(self, usage: TokenUsage) -> None:
        self.used_tokens += usage.total_tokens

def trim_messages_to_budget(messages: list, budget: TokenBudget) -> list:
    """Trim conversation history to fit within budget."""
    encoder = tiktoken.encoding_for_model("gpt-4o")
    while True:
        total = sum(len(encoder.encode(m.content)) for m in messages)
        if total <= budget.max_input_tokens:
            break
        # Remove oldest non-system message
        for i, m in enumerate(messages):
            if not isinstance(m, SystemMessage):
                messages.pop(i)
                break
    return messages
```

### 8.6 Streaming for Perceived Latency

Even if total latency is high, streaming first tokens dramatically improves perceived responsiveness:

```python
async def stream_agent_response(graph, input_state, config):
    """Stream agent output to client as it's generated."""
    async for chunk in graph.astream(input_state, config, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            yield f"data: {json.dumps({'delta': chunk.content})}\n\n"
    yield "data: [DONE]\n\n"
```

---

## 9. Scalability Patterns

### 9.1 Async-First Architecture

Agents are I/O bound (waiting for LLM APIs, tool responses). Use async throughout:

```python
# FastAPI async endpoint
@app.post("/run")
async def run_agent(request: RunRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    background_tasks.add_task(execute_agent_task, task_id, request.task)
    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    state = await get_task_state(task_id)
    return {
        "task_id": task_id,
        "status": state.status,
        "progress": state.current_step,
        "result": state.final_output if state.status == "done" else None,
    }
```

### 9.2 Task Queue Architecture

For production throughput, decouple request ingestion from agent execution:

```
FastAPI (ingestion)        Redis (queue)          Worker pool (execution)
     │                         │                         │
POST /run ──────────────► LPUSH tasks ─────────────► BRPOP tasks
     │                         │                         │
     └── return task_id        │                     Agent graph
                               │                         │
GET /status ◄────────── State store ◄──── Write results ─┘
```

Using `arq` (async Redis queue):
```python
# Worker definition
async def agent_task(ctx, task_id: str, user_task: str):
    graph = get_graph()
    config = {"configurable": {"thread_id": task_id}}
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_task)]},
            config
        )
        await store_result(task_id, result["final_output"])
    except Exception as e:
        await store_error(task_id, str(e))

# Worker settings
class WorkerSettings:
    functions = [agent_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = 10  # concurrent agents per worker
    job_timeout = 300
```

### 9.3 Multi-Tenancy

Isolate tenants with separate namespaces for checkpoints, memory, and tool sandboxes:

```python
def get_thread_config(user_id: str, task_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": f"{user_id}:{task_id}",  # namespaced
            "user_id": user_id,
        }
    }

def get_workspace_path(user_id: str) -> Path:
    """Each user gets an isolated workspace directory."""
    path = WORKSPACE_ROOT / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Apply rate limits per tenant at the API gateway:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_user_id_from_token)

@app.post("/run")
@limiter.limit("10/minute")  # 10 agent runs per minute per user
async def run_agent(request: RunRequest):
    ...
```

### 9.4 Horizontal Scaling

Stateless workers + external state storage = horizontal scalability:

```
Load Balancer
   ├── Worker 1 (agent process)
   ├── Worker 2 (agent process)   ──► Shared Postgres (checkpoints)
   ├── Worker 3 (agent process)   ──► Shared Redis (queue + cache)
   └── Worker N (agent process)   ──► Shared Vector DB (memory)
```

Workers are stateless — all state lives in Postgres, Redis, or the vector DB. Scale workers horizontally by adding instances. Auto-scale based on queue depth using Kubernetes HPA or AWS Auto Scaling.

---

## 10. Production Observability

### 10.1 The Three Pillars

**Traces**: What did the agent do and in what order?
**Metrics**: How is the system performing over time?
**Logs**: What happened in detail at any given moment?

### 10.2 Structured Trace Design

Every agent execution should emit a structured trace:

```python
@dataclass
class AgentTrace:
    trace_id: str
    thread_id: str
    user_id: str
    task: str
    started_at: datetime
    completed_at: datetime
    status: str  # success | error | timeout | human_escalated
    spans: list[AgentSpan]
    total_tokens: int
    total_cost_usd: float
    iteration_count: int
    final_output: Optional[str]
    error: Optional[str]

@dataclass
class AgentSpan:
    span_id: str
    parent_span_id: Optional[str]
    node_name: str  # supervisor | planner | executor | critic | tool:web_search
    started_at: datetime
    duration_ms: int
    input_tokens: int
    output_tokens: int
    tool_name: Optional[str]
    tool_input: Optional[dict]
    tool_output: Optional[str]
    error: Optional[str]
```

### 10.3 Key Metrics to Track

**Business metrics:**
- Task completion rate (% tasks reaching `final_output` successfully)
- Human escalation rate (% tasks requiring human approval)
- User satisfaction score (thumbs up/down in UI)
- Cost per successful task

**Operational metrics:**
- Latency P50/P95/P99 (end-to-end wall-clock time)
- LLM API error rate (rate limit, timeout, server error)
- Tool error rate per tool
- Queue depth and wait time
- Active agent instances

**Quality metrics:**
- Average iteration count per task
- LLM-judge score distribution
- Regression test pass rate over time
- Token budget utilization (% of budget used)

### 10.4 Alerting Rules

```yaml
# Example Grafana alert rules
alerts:
  - name: high_task_failure_rate
    condition: task_failure_rate > 0.05  # > 5% failure
    window: 5m
    severity: critical

  - name: high_latency
    condition: p95_latency > 120s
    window: 10m
    severity: warning

  - name: budget_exhaustion
    condition: avg_token_utilization > 0.9
    window: 1h
    severity: warning  # Agent consistently hitting budget

  - name: queue_backup
    condition: queue_depth > 100
    window: 5m
    severity: critical
```

### 10.5 Dashboard Essentials

**Operational dashboard** (real-time):
- Current queue depth and active tasks
- Running P50/P95 latency (15-min window)
- Error rate by type (tool error, LLM error, timeout)
- Cost per hour by model

**Quality dashboard** (daily):
- Task completion rate trend (7-day rolling)
- Average LLM-judge scores by task type
- Human escalation rate trend
- Top failure modes (pie chart of error types)
- Regression test results by version

---

## 11. Deployment Strategy

### 11.1 The Agent CI/CD Pipeline

```
Code change → Git push
   ↓
CI: Level 1 unit tests (pytest, ~2 min)
   ↓ (pass)
CI: Integration tests with mocked LLM (~5 min)
   ↓ (pass)
CI: Level 2 eval on golden dataset (real LLM, ~20 min)
   ↓ (pass, regression score ≥ baseline)
Deploy to staging environment
   ↓
Manual QA on staging (smoke tests)
   ↓
Canary deploy: 5% of production traffic
   ↓
Monitor: task completion rate, error rate, latency (1 hour)
   ↓ (metrics nominal)
Gradual rollout: 25% → 50% → 100%
```

### 11.2 Feature Flags for Agents

Use feature flags to safely roll out changes:

```python
async def get_agent_config(user_id: str) -> AgentConfig:
    flags = await feature_flags.evaluate(user_id)
    return AgentConfig(
        model=flags.get("agent_model", "gpt-4o"),
        max_iterations=flags.get("max_iterations", 5),
        enable_parallel_execution=flags.get("parallel_exec", False),
        use_reasoning_planner=flags.get("reasoning_planner", False),
    )
```

Feature flags enable:
- Rollout new model without full deployment
- Segment rollouts by user tier
- Instant rollback by toggling flag (no redeployment)

### 11.3 Rollback Strategy

Agents have persistent state (checkpoints). Rollback is more complex than stateless services:

1. **Code rollback**: Redeploy previous container image (fast, seconds)
2. **Checkpoint compatibility**: Ensure old code can read new-format checkpoints
   - Use schema versioning in checkpoints
   - New fields must be Optional with defaults
3. **In-flight tasks**: Active tasks running at rollback time
   - Drain queue before rollback (wait for active tasks to complete)
   - Or: allow in-flight tasks to complete with old version (blue/green)
4. **Model rollback**: If rolling back a model change, update the feature flag (instant)

### 11.4 Environment Management

```
Development:
  - Local SQLite checkpoints
  - Mock LLM (recorded responses)
  - Unrestricted tool access
  - Verbose logging

Staging:
  - Postgres checkpoints (same schema as prod)
  - Real LLM with small token budgets
  - Tool sandbox with realistic restrictions
  - Full observability stack

Production:
  - Postgres checkpoints (HA, read replicas)
  - Real LLM with production budgets
  - Full guardrails active
  - Alerting configured
```

---

## 12. Anti-Patterns

### 12.1 Premature Agentic Complexity

**Anti-pattern**: Jumping straight to a multi-agent system for a task that could be solved with a single well-crafted prompt.

**Symptom**: The demo works, but production is brittle, slow, and expensive.

**Fix**: Follow the escalation ladder (Section 2.1). Only add agents when simpler patterns provably fail.

### 12.2 Framework Cargo-Culting

**Anti-pattern**: Using a framework like LangGraph, CrewAI, or AutoGen as a black box without understanding the underlying LLM calls.

**Symptom**: Debugging is impossible because errors appear inside framework internals. Incorrect assumptions about what the framework does cause silent failures.

**Fix**: Read the framework source code for any component you use in production. When debugging, always look at the raw API calls. As Anthropic recommends: "reduce abstraction layers and build with basic components as you move to production."

### 12.3 Over-trusting the Planner

**Anti-pattern**: Executing the agent's plan without validation, allowing any tool call the LLM decides to make.

**Symptom**: Agent calls destructive tools, exposes credentials, or is hijacked via prompt injection.

**Fix**: Implement plan validation (Section 7.3) and tool approval tiers (Section 5.3).

### 12.4 No Evaluation System

**Anti-pattern**: Measuring quality by vibe-checking a few outputs after each change.

**Symptom**: "Whack-a-mole" — fixing one failure mode introduces others. Quality stagnates or regresses.

**Fix**: Build all three eval levels before shipping to production. Invest evaluation time like a senior engineer invests in test coverage.

### 12.5 Unbounded Loops

**Anti-pattern**: No maximum iteration count or token budget on the agent's execution loop.

**Symptom**: A confusing task causes the agent to loop indefinitely, consuming hundreds of dollars in API costs.

**Fix**: Every loop must have a stopping condition: max iterations, token budget, wall-clock timeout. Design the "max budget reached" state to produce a graceful partial result.

### 12.6 Monolithic Agent

**Anti-pattern**: One large agent with many tools and a massive system prompt, trying to handle all task types.

**Symptom**: Inconsistent behavior, poor tool selection, context window pressure, no way to isolate failures.

**Fix**: Decompose into specialized agents. Each agent should have a clear, narrow responsibility and a small, focused tool set. Route to the right agent via a supervisor.

### 12.7 Synchronous Blocking API

**Anti-pattern**: Long-polling a REST endpoint that blocks for the full agent execution duration (potentially minutes).

**Symptom**: Client timeouts, poor UX, inability to scale because connections are held open.

**Fix**: Async pattern: `POST /run` returns `task_id` immediately; client polls `GET /status/{id}` or subscribes to SSE/WebSocket for streaming updates.

---

## 13. Case Studies

### 13.1 Customer Support Agent (Production Pattern)

**Architecture:**

```
User message
   ↓
Intent classifier (fast model, cheap)
   ├── General FAQ → Template response (no LLM needed)
   ├── Order status → Tool: query_order_db → Format response
   ├── Refund request → Agent with tools + HITL for amounts > $500
   └── Technical support → Full agent with KB retrieval + HITL for resolution
```

**Production lessons:**
- Most support interactions (60-80%) can be handled with workflows, not agents
- Agents reserved for open-ended technical issues where structured workflows fail
- Success metric: resolution rate (did the customer's issue get resolved?)
- HITL threshold set at dollar amounts and sensitivity levels, not task types
- Evaluation: sampling 5% of interactions daily for human quality review

### 13.2 Coding Agent (Production Pattern)

**Architecture:**

```
GitHub Issue / PR description
   ↓
Repository analysis (file tree, relevant files)
   ↓
Plan: list of files to modify and changes needed
   ↓
Execution: for each file → read → modify → validate
   ↓
Test execution: run test suite, check for failures
   ↓
Critic: review changes against requirements
   ↓ (iterate up to 3 times)
PR creation with human review required
```

**Production lessons:**
- Code solutions are verifiable via automated tests — use tests as the primary quality signal, not LLM self-evaluation
- Agents iterate on solutions using test results as feedback (not just LLM critique)
- Human review of PRs is non-negotiable; agents create PRs, humans approve merges
- Tool design is more important than prompt design: absolute file paths prevent a class of errors; explicit "show me the file before editing" prevents hallucinated edits
- SWE-bench performance benchmarks the agent against real GitHub issues

### 13.3 Data Analysis Agent (Production Pattern)

**Architecture:**

```
User request: "Analyze Q3 sales by region and identify trends"
   ↓
Planner: decompose into data retrieval → analysis → visualization → summary
   ↓
Executor:
  1. Query database (SQL tool)
  2. Execute Python analysis (sandboxed E2B)
  3. Generate charts (matplotlib in sandbox)
  4. Write findings to workspace
   ↓
Critic: verify analysis completeness, check for errors
   ↓
Deliver: summary + files written to workspace
```

**Production lessons:**
- Code execution (E2B sandbox) is safer and more capable than JSON tool calls for complex analysis
- Output artifacts (charts, CSV files) stored in workspace, not embedded in LLM context
- Budget control critical: complex analysis can generate hundreds of LLM calls
- Intermediate results cached to avoid re-running expensive queries on retry

---

## Summary: Production-Ready Checklist

### Before Shipping

- [ ] Escalation ladder reviewed — is an agent actually needed?
- [ ] Simplest architecture chosen that meets requirements
- [ ] All loops have stopping conditions (max_iterations, token_budget, timeout)
- [ ] Retry logic with exponential backoff implemented
- [ ] Circuit breakers on all external tool calls
- [ ] Error taxonomy defined and handled appropriately
- [ ] HITL approval tiers defined and implemented
- [ ] All agent state checkpointed to persistent storage
- [ ] Level 1 unit tests written (target: >30 tests covering key behaviors)
- [ ] Level 2 eval golden dataset created (target: >50 examples)
- [ ] LLM-as-judge pipeline configured in Langfuse or LangSmith
- [ ] Input validation and prompt injection defense implemented
- [ ] Tool authorization via allowlist
- [ ] Output PII filtering implemented
- [ ] Audit logging enabled for all agent actions

### Infrastructure

- [ ] Async task queue (ARQ, Celery) for long-running tasks
- [ ] `POST /run` returns immediately with task_id
- [ ] `GET /status/{id}` polls state from persistent store
- [ ] Per-user rate limiting at API gateway
- [ ] Per-run token budget enforced
- [ ] Postgres (not SQLite) for multi-instance checkpointing
- [ ] Observability: traces in Langfuse/LangSmith, metrics in Grafana
- [ ] Alerts configured for: failure rate, latency P95, queue depth

### Deployment

- [ ] Feature flags for model and parameter changes
- [ ] Canary deploy process (5% → 25% → 50% → 100%)
- [ ] Rollback plan documented (code, checkpoint schema, in-flight tasks)
- [ ] Separate dev/staging/production environments
- [ ] Level 2 eval run in CI as quality gate

---

## References

1. Anthropic. *Building Effective Agents* (2024). https://www.anthropic.com/research/building-effective-agents
2. Hamel Husain. *Your AI Product Needs Evals* (2024). https://hamel.dev/blog/posts/evals/
3. Eugene Yan. *Patterns for Building LLM-based Systems & Products* (2023). https://eugeneyan.com/writing/llm-patterns/
4. LangChain. *LangGraph Concepts: Persistence* (2025). https://langchain-ai.github.io/langgraph/concepts/persistence/
5. LangChain. *LangGraph Concepts: Human-in-the-Loop* (2025). https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
6. Google Cloud. *Vertex AI Agent Engine Overview* (2025). https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview
7. Ufuk Bombaci & Manish Sahu. *Evaluating LLM Agents* (2024). https://www.deeplearning.ai/short-courses/evaluating-ai-agents/
8. Karpathy, Andrej. *Software 2.0* (2017). https://karpathy.medium.com/software-2-0-a64152b37c35
9. Anthropic. *Appendix: Prompt Engineering Your Tools* in *Building Effective Agents* (2024).
10. Martin Fowler. *Patterns of Distributed Systems* (2023). https://martinfowler.com/articles/patterns-of-distributed-systems/
11. AWS. *Amazon Bedrock Agents Overview* (2025). https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html
12. Anthropic. *SWE-bench Technical Report* (2024). https://www.anthropic.com/research/swe-bench-sonnet

---

# Modern Production-Grade Agentic Architecture (2025–2026)

> This section synthesizes the current state-of-the-art in production agentic system design, drawing from deployed systems at Anthropic, Microsoft, Google, AWS, and leading engineering teams. It presents the architecture most teams are converging on in 2025–2026 and explains the engineering rationale behind each decision.

---

## A. Why Now: The Inflection That Changed Architecture

Before 2024, most "agent" architectures were brittle: LLM calls chained together with fragile string parsing, no true persistence, no real sandboxed execution, and no interoperability between systems. Three technological shifts changed what is architecturally possible:

1. **Reliable tool use** — GPT-4o, Claude 3.5+, and Gemini 2.0+ emit structurally correct tool calls with very high fidelity. The "LLM won't call tools correctly" failure mode has largely been solved.
2. **Open protocols** — MCP (Nov 2024) and A2A (Apr 2025) established vendor-neutral standards for tool connectivity and agent-to-agent delegation, making heterogeneous multi-agent systems viable.
3. **Managed runtimes** — Vertex AI Agent Engine, LangGraph Cloud, and AWS Bedrock AgentCore provide production-grade state persistence, scaling, and observability without building from scratch.

The modern architecture reflects all three of these: it assumes reliable tool calling, is protocol-first (MCP + A2A), and delegates infrastructure concerns to purpose-built runtimes.

---

## B. The Reference Architecture

### B.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT TIER                                         │
│   Web UI / CLI / Slack Bot / Voice Interface / REST API consumer            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ HTTPS / WebSocket / SSE
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                        GATEWAY TIER                                         │
│  API Gateway (Kong / AWS API GW / Nginx)                                    │
│  • Auth (JWT / OAuth2)          • Rate limiting (per-user, per-org)         │
│  • Request routing              • TLS termination                           │
│  • Idempotency key enforcement  • WAF / DDoS protection                     │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────────┐
        │              ORCHESTRATION TIER                   │
        │                                                   │
        │  ┌─────────────────────────────────────────────┐  │
        │  │           Async Task Queue                  │  │
        │  │   POST /run → task_id (immediate)           │  │
        │  │   Redis / ARQ / Temporal                    │  │
        │  └──────────────────┬──────────────────────────┘  │
        │                     │                             │
        │  ┌──────────────────▼──────────────────────────┐  │
        │  │            Agent Worker Pool                │  │
        │  │  (stateless pods, horizontally scalable)    │  │
        │  │                                             │  │
        │  │  ┌─────────────────────────────────────┐   │  │
        │  │  │      Supervisor / Router Agent      │   │  │
        │  │  │  (fast model — GPT-4o-mini/Haiku)   │   │  │
        │  │  └──────────────┬──────────────────────┘   │  │
        │  │                 │ route                     │  │
        │  │  ┌──────────────▼──────────────────────┐   │  │
        │  │  │   Specialized Sub-Agent Pool         │   │  │
        │  │  │                                      │   │  │
        │  │  │  • Planner (reasoning model)         │   │  │
        │  │  │  • Researcher (standard model + RAG) │   │  │
        │  │  │  • Coder (standard model + E2B)      │   │  │
        │  │  │  • Critic / Evaluator                │   │  │
        │  │  │  • Domain specialists (via A2A)      │   │  │
        │  │  └──────────────┬──────────────────────┘   │  │
        │  └─────────────────┼───────────────────────────┘  │
        └─────────────────────┼────────────────────────────┘
                              │ tool calls (MCP)
┌─────────────────────────────▼──────────────────────────────────────────────┐
│                           TOOL TIER                                        │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  MCP Server  │  │  MCP Server  │  │  MCP Server  │  │  MCP Server   │  │
│  │  Web Search  │  │   Databases  │  │  Code Exec   │  │  File System  │  │
│  │  (Tavily)    │  │  (pgvector)  │  │  (E2B)       │  │  (sandboxed)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  MCP Server  │  │  MCP Server  │  │   External   │                     │
│  │  Vector DB   │  │  HTTP Client │  │  A2A Agents  │                     │
│  │  (Chroma)    │  │  (httpx)     │  │  (delegated) │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
└────────────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────────┐
│                          PERSISTENCE TIER                                   │
│                                                                             │
│  ┌─────────────────────┐  ┌───────────────────┐  ┌──────────────────────┐ │
│  │  Checkpoint Store   │  │   Session / Cache  │  │   Long-Term Memory   │ │
│  │  Postgres           │  │   Redis            │  │   Vector DB          │ │
│  │  (LangGraph saver)  │  │   (hot state)      │  │   (Chroma / PG)      │ │
│  └─────────────────────┘  └───────────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────────┐
│                       OBSERVABILITY TIER                                    │
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐                 │
│  │  Trace Store   │  │   Metrics      │  │  Eval Engine  │                 │
│  │  Langfuse /    │  │   Prometheus   │  │  LLM-as-judge │                 │
│  │  LangSmith     │  │   + Grafana    │  │  + Datasets   │                 │
│  └────────────────┘  └────────────────┘  └───────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### B.2 Tier Responsibilities

| Tier | Responsibility | Technology choices |
|---|---|---|
| **Client** | User interaction surface | Web UI, CLI, Slack, Voice |
| **Gateway** | Auth, rate limiting, routing, security | Kong, AWS API GW, Nginx + Lua |
| **Orchestration** | Task lifecycle, agent routing, worker scaling | FastAPI + ARQ/Temporal + LangGraph |
| **Tool** | Standardized external integrations | MCP servers (stdio or HTTP+SSE) |
| **Persistence** | State durability, memory, caching | Postgres + Redis + Vector DB |
| **Observability** | Tracing, metrics, evaluation, alerts | Langfuse/LangSmith + Prometheus |

---

## C. Core Architectural Patterns in Depth

### C.1 The Hierarchical Supervisor Pattern (Dominant Production Pattern)

The most widely deployed production pattern in 2025 is **hierarchical multi-agent with a fast supervisor**. It descends from Anthropic's orchestrator-workers workflow and is validated at scale by Microsoft Magentic-One, Google ADK, and the internal systems of dozens of enterprise deployments.

```
User Task
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              SUPERVISOR (fast model)                 │
│  Role: classify task, route to specialist, track     │
│  progress, enforce policies, decide when done        │
│  Model: GPT-4o-mini or Claude Haiku (cheap + fast)   │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  Planner    Researcher   Coder    Domain Agent
 (reasoning) (standard)  (standard)  (via A2A)
```

**Why a fast model as supervisor?**

The supervisor runs on *every* task, potentially multiple times per task (to check progress). Using a reasoning model here would make every task 5-10× more expensive and 10× slower. The supervisor's job is routing and tracking — classification tasks that fast models handle with near-equal accuracy to expensive ones.

The reasoning model is reserved for **planning**: decomposing ambiguous, open-ended tasks into concrete steps. This is where reasoning quality pays off. Once a task is well-structured, standard models can execute it.

**Microsoft Magentic-One's contribution**: The dual-ledger design — a **Task Ledger** (facts, guesses, overall plan) maintained in an outer loop, and a **Progress Ledger** (current assignment, step result) maintained in an inner loop. This separates strategic planning from tactical execution, allowing the orchestrator to re-plan when progress stalls without discarding all context.

```python
class OrchestratorState(TypedDict):
    # Outer loop: strategic context (updated rarely)
    task_ledger: TaskLedger       # facts, guesses, current plan
    stall_count: int              # consecutive steps without progress

    # Inner loop: tactical state (updated every step)
    progress_ledger: ProgressLedger  # current step, agent assigned, result
    messages: Annotated[list, add_messages]

async def orchestrator_outer_loop(state: OrchestratorState) -> OrchestratorState:
    """Re-plan when progress is stalled (outer loop)."""
    if state["stall_count"] >= STALL_THRESHOLD:
        new_plan = await replanner.ainvoke(state["task_ledger"])
        return {**state, "task_ledger": update_plan(state["task_ledger"], new_plan), "stall_count": 0}
    return state

async def orchestrator_inner_loop(state: OrchestratorState) -> OrchestratorState:
    """Assign next step to best agent (inner loop)."""
    next_step = pick_next_step(state["task_ledger"].plan, state["progress_ledger"])
    best_agent = route_to_agent(next_step)
    result = await best_agent.invoke(next_step, state)
    progress_made = assess_progress(state["progress_ledger"], result)
    return {
        **state,
        "progress_ledger": update_progress(state["progress_ledger"], result),
        "stall_count": 0 if progress_made else state["stall_count"] + 1,
    }
```

### C.2 MCP-First Tool Architecture

The 2024–2025 shift: tools are no longer Python functions registered directly on the agent. They are **MCP servers** — independent processes exposing a standard JSON-RPC interface.

**Why MCP-first matters for production:**

1. **Language independence**: Your MCP server can be written in any language. The Rust database connector, the TypeScript web scraper, and the Python code executor all expose the same interface.
2. **Reuse across agents**: A single MCP server for web search can be shared by every agent in the system without duplication.
3. **Independent scaling**: Scale your code execution MCP server to 50 instances without touching agent code.
4. **Independent deployment**: Deploy a new version of a tool without redeploying the agent.
5. **Third-party ecosystem**: As of 2025, hundreds of MCP servers are available (GitHub, Slack, databases, Notion, etc.) — a free ecosystem of production-tested tools.

```
Before MCP (2023 pattern):
Agent code
  └── tools/web_search.py    ← coupled to agent process
  └── tools/code_executor.py ← coupled to agent process
  └── tools/file_system.py   ← coupled to agent process

After MCP (2025 pattern):
Agent process                  Separate MCP server processes
  └── mcp_client ─────────────► web-search-mcp (stdio/HTTP)
  └── mcp_client ─────────────► code-exec-mcp (stdio/HTTP)
  └── mcp_client ─────────────► filesystem-mcp (stdio/HTTP)
  └── mcp_client ─────────────► github-mcp (HTTP)  [community]
  └── mcp_client ─────────────► slack-mcp (HTTP)   [community]
```

**MCP server in production:**

```python
from mcp import FastMCP
import httpx

mcp = FastMCP("web-search-server")

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for recent information.

    Args:
        query: The search query string
        max_results: Number of results to return (1-10)

    Returns:
        List of results with title, url, and content snippet
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
        )
    return response.json()["results"]

if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        mcp.run(transport="http", host="0.0.0.0", port=8001)
    else:
        mcp.run()  # stdio (default, for local use)
```

**Connecting agents to MCP servers:**

```python
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

async def build_agent_with_mcp_tools():
    server_params = [
        StdioServerParameters(command="python", args=["tools/web_search_mcp.py"]),
        StdioServerParameters(command="python", args=["tools/code_exec_mcp.py"]),
    ]
    async with MultiServerMCPClient(server_params) as client:
        tools = await client.get_tools()
        agent = create_react_agent(llm, tools)
        return agent
```

### C.3 A2A for Cross-System Agent Delegation

Where MCP connects agents to tools, A2A connects agents to other agents. The 2025 pattern:

```
Your Company's Agent Cluster (internal)
    │
    │  uses MCP to connect to internal tools
    │
    └──► via A2A ──► Salesforce Agent (vendor)
    └──► via A2A ──► ServiceNow Agent (vendor)
    └──► via A2A ──► Partner Company Agent (external)
    └──► via A2A ──► Specialized Research Agent (3rd party)
```

This creates an **agentic services mesh** — the enterprise software stack becomes callable by agents through a standard protocol, eliminating bespoke integrations.

**Agent Card** (A2A capability advertisement):

```json
{
  "name": "DataAnalysis Agent",
  "description": "Analyzes structured data, generates visualizations, and produces insights reports",
  "url": "https://agents.company.com/data-analysis",
  "version": "1.0.0",
  "skills": [
    {
      "id": "analyze-csv",
      "name": "CSV Data Analysis",
      "description": "Accepts CSV file, returns statistical summary and visualizations",
      "inputModes": ["file", "text"],
      "outputModes": ["file", "text"]
    }
  ],
  "authentication": {
    "schemes": ["bearer"]
  }
}
```

**Why A2A is architecturally significant**: It decouples capability from implementation. The billing agent, HR agent, and research agent can be built by different teams, in different frameworks, and deployed independently — yet they interoperate through a stable protocol contract. This is the service-oriented architecture pattern applied to AI agents.

### C.4 Durable Execution with Workflow Engines

For long-running agent tasks (hours to days), the LangGraph checkpointer approach has a limitation: it depends on the application process being alive to resume. A crash mid-task requires an external mechanism to restart.

**Temporal** solves this with **durable execution**: the workflow runtime itself persists every function call's state and automatically retries or resumes after any failure, including process crashes, server reboots, and network partitions.

```python
# Temporal workflow: agent task that survives any failure
from temporalio import workflow, activity
from datetime import timedelta

@activity.defn
async def run_planner(task: str) -> list[str]:
    """Runs planner LLM call. If this crashes, Temporal replays it."""
    return await planner_llm.ainvoke([HumanMessage(content=task)])

@activity.defn
async def run_executor(step: str, context: dict) -> str:
    """Runs one execution step with tool access."""
    return await executor_agent.ainvoke(step, context)

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, task: str) -> str:
        # Each activity is automatically durable — survives crashes
        plan = await workflow.execute_activity(
            run_planner, task,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        results = []
        for step in plan:
            result = await workflow.execute_activity(
                run_executor, step, {"previous": results},
                start_to_close_timeout=timedelta(minutes=10),
            )
            results.append(result)

        # Human approval as a signal (HITL)
        await workflow.wait_condition(lambda: self.approved)
        return "\n".join(results)

    @workflow.signal
    def approve(self) -> None:
        self.approved = True
```

**When to use Temporal vs. LangGraph checkpointing:**

| Scenario | LangGraph + Postgres | Temporal |
|---|---|---|
| Task duration | < 30 minutes | Hours to days |
| Failure recovery | Manual restart needed | Automatic |
| Complex retry logic | Custom code | Built-in policies |
| Scheduling (cron) | External cron job | Built-in |
| Audit / compliance | Via trace store | Built-in event history |
| Team familiarity | Python-native | Requires Temporal learning |

Most teams start with LangGraph + Postgres checkpointing. Graduate to Temporal when tasks run for hours, require complex retry policies, or need compliance-grade audit trails.

### C.5 Layered Memory Architecture

Modern production agents maintain memory across four distinct layers, each with its own storage technology and access pattern:

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: Procedural Memory                              │
│  What: Learned skills, few-shot examples, tool patterns   │
│  Storage: Vector DB (retrieved by semantic similarity)    │
│  Populated by: Successful task traces, human curation     │
│  Accessed: At task start (retrieval-augmented prompting)  │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Semantic Memory                                │
│  What: Facts about the world, domain knowledge base       │
│  Storage: Vector DB + structured DB (hybrid retrieval)    │
│  Populated by: Document ingestion, web scraping           │
│  Accessed: RAG retrieval during task execution            │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Episodic Memory                                │
│  What: User interaction history, task outcomes           │
│  Storage: Postgres (structured) + Redis (recent hot)      │
│  Populated by: Task completion events                     │
│  Accessed: At session start, personalization              │
├──────────────────────────────────────────────────────────┤
│  Layer 1: Working Memory                                 │
│  What: Current task context, in-progress conversation     │
│  Storage: LLM context window (in-memory)                  │
│  Populated by: Current task input + retrieved context     │
│  Accessed: Every LLM call                                 │
└──────────────────────────────────────────────────────────┘
```

**Memory management in practice:**

```python
async def build_agent_context(user_id: str, task: str, state: AgentState) -> list[BaseMessage]:
    """Construct the full agent context by assembling all memory layers."""

    # Layer 4: Retrieve relevant few-shot examples (procedural)
    similar_tasks = await vector_db.query(
        collection="successful_traces",
        query=task,
        filter={"user_tier": user.tier},
        limit=3,
    )
    few_shot_examples = format_examples(similar_tasks)

    # Layer 3: Retrieve relevant domain knowledge (semantic)
    knowledge_chunks = await vector_db.query(
        collection="knowledge_base",
        query=task,
        limit=5,
    )
    knowledge_context = format_chunks(knowledge_chunks)

    # Layer 2: Load user's recent history (episodic)
    user_history = await postgres.fetch(
        "SELECT summary FROM user_memory WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1",
        user_id,
    )

    # Layer 1: Build working memory (assembled system prompt)
    system_prompt = f"""
You are an expert AI agent.

## User Context
{user_history[0]["summary"] if user_history else "New user"}

## Relevant Knowledge
{knowledge_context}

## Similar Past Tasks (examples)
{few_shot_examples}
"""
    return [SystemMessage(content=system_prompt), HumanMessage(content=task)]
```

**Memory consolidation** — after each task, distill outcomes into long-term memory:

```python
async def consolidate_memory(user_id: str, trace: AgentTrace) -> None:
    """Update long-term memory with task outcomes."""
    if trace.status != "success":
        return

    # Store successful trace as few-shot example
    await vector_db.upsert(
        collection="successful_traces",
        document={
            "task": trace.task,
            "approach": trace.plan,
            "tools_used": [s.tool_name for s in trace.spans if s.tool_name],
            "outcome": trace.final_output[:500],
        },
        id=trace.trace_id,
    )

    # Update episodic summary (compressed history)
    await update_user_memory_summary(user_id, trace)
```

### C.6 Async-Native Request Lifecycle

Every production agent system must separate request ingestion (fast, synchronous) from agent execution (slow, asynchronous). This is the single most common architectural mistake teams make when moving from demo to production.

```
Client                  FastAPI               Redis Queue         Worker
  │                       │                       │                 │
  │  POST /run             │                       │                 │
  │  {task: "..."}  ──────►│                       │                 │
  │                       │ enqueue(task_id, task) │                 │
  │                       │──────────────────────►│                 │
  │  {task_id: "abc"}      │                       │ dequeue         │
  │◄──────────────────────│                       │────────────────►│
  │                       │                       │  run agent      │
  │                       │                       │  (minutes)      │
  │  GET /status/abc ─────►│                       │                 │
  │                       │ read state from Postgres                │
  │  {status: "running"}   │                       │                 │
  │◄──────────────────────│                       │ write result    │
  │                       │                       │◄────────────────│
  │  GET /status/abc ─────►│                       │                 │
  │  {status: "done",      │                       │                 │
  │   result: "..."}◄──────│                       │                 │
```

**WebSocket / SSE for streaming:**

```python
@app.websocket("/ws/{task_id}")
async def stream_agent_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    async for event in subscribe_to_task_events(task_id):
        await websocket.send_json({
            "type": event.type,             # "node_complete" | "tool_call" | "done"
            "node": event.node_name,
            "content": event.content,
            "timestamp": event.timestamp.isoformat(),
        })
        if event.type == "done":
            break
    await websocket.close()
```

---

## D. Technology Stack Decision Matrix

For each architectural layer, here is the recommended technology choice and the engineering rationale:

### D.1 Agent Framework

**Recommended: LangGraph (self-hosted) or LangGraph Cloud (managed)**

**Rationale:**
- LangGraph's state machine model maps directly to the agent execution loop: each node is a Python function, edges encode routing logic, and the compiled graph is a deterministic, inspectable artifact.
- The `SqliteSaver` → `PostgresSaver` upgrade path is seamless: same API, swap the backend.
- LangGraph Cloud adds: fault-tolerant task queues, double-texting (new input on running thread), cron jobs, and streaming — without custom infrastructure.
- Postgres checkpointer uses pipeline mode (batching writes) and stores only delta state (changed channels), making it efficient at scale.

**Alternatives and when to choose them:**
- **OpenAI Agents SDK**: When your team is fully OpenAI-native and wants minimal setup. Loses portability.
- **Google ADK**: When deploying on Vertex AI or needing first-class voice/video streaming.
- **AWS Strands**: When all infrastructure is on AWS and Bedrock is the LLM provider. Native A2A support, Bedrock Guardrails integration.
- **Temporal + LangGraph**: When tasks run for hours, compliance audit trails are required, or retry policies are complex.

### D.2 LLM Routing

**Recommended: Three-model tier with intelligent routing**

```python
MODEL_ROUTING = {
    "classification":  "gpt-4o-mini",      # ~$0.15/1M input tokens
    "planning":        "o3-mini",           # ~$1.10/1M input — reasoning quality
    "execution":       "gpt-4o",            # ~$2.50/1M input — reliable tool use
    "critic":          "gpt-4o-mini",       # classification + comparison only
    "embedding":       "text-embedding-3-small",
}
```

**Rationale:**
- The planner sees ambiguous, complex input and must produce a high-quality structured plan. Reasoning models (o3, Gemini 2.5 Pro thinking) pay for themselves here in reduced planning failures.
- The executor calls tools deterministically based on a structured plan. Standard models with strong tool-calling are sufficient and ~2-3× cheaper.
- The supervisor and critic do classification, not generation. Fast/cheap models handle this accurately.
- Using a reasoning model everywhere is 10-50× the cost of this tiered approach with minimal quality difference.

### D.3 Tool Infrastructure

**Recommended: MCP servers with HTTP+SSE transport for remote, stdio for local**

```
Development:     stdio transport  (fastest startup, no network overhead)
Staging/Prod:    HTTP+SSE transport (independent deployment, load balancing)
```

**Tool reliability stack:**
```python
# Every MCP tool call goes through this stack:
result = await (
    circuit_breaker          # Open on repeated failure
    .with_retry(3, backoff)  # Exponential backoff on transient errors
    .with_timeout(30s)       # Never hang indefinitely
    .with_cache(ttl=3600)    # Avoid redundant calls for same input
    .call(mcp_client.call_tool, tool_name, tool_input)
)
```

### D.4 Code Execution

**Recommended: E2B for production, RestrictedPython for low-risk internal tools**

**Rationale:**
- E2B provides VM-level isolation (~150ms startup). No attacker can escape a microVM regardless of the Python code they inject.
- RestrictedPython is acceptable for internal enterprise tools where inputs are from trusted users, not from the internet.
- Docker is a reasonable middle ground for self-hosted deployments — container isolation is sufficient for most threat models.

### D.5 Persistence

**Recommended: Postgres-first architecture**

```
Postgres:
  - LangGraph checkpoints (agent state at each step)
  - User episodic memory (interaction summaries)
  - Audit log (append-only, all agent actions)
  - Task status store (queried by /status endpoint)
  - pgvector extension for vector search (avoid separate DB if scale allows)

Redis:
  - Task queue (ARQ or Bull)
  - Hot session cache (recent context, sub-millisecond access)
  - Distributed rate limiting counters
  - Pub/Sub for real-time streaming events to WebSocket clients

Vector DB (standalone, if scale demands):
  - Chroma or Qdrant for semantic memory (knowledge base, few-shot examples)
  - Migrate from pgvector when vectors exceed ~10M rows or multi-tenant isolation needed
```

**Why Postgres-first?** Most teams have Postgres expertise, tooling, and operational runbooks. pgvector handles millions of vectors efficiently. Starting with Postgres + pgvector for all persistence needs simplifies operations significantly, and the migration path to standalone vector DBs is well-defined when needed.

### D.6 Observability

**Recommended: Langfuse (open source, self-hostable) as primary; Prometheus + Grafana for metrics**

**Rationale:**
- Langfuse captures full LLM traces automatically via SDK instrumentation and supports OpenTelemetry, making it compatible with existing enterprise observability stacks.
- Self-hosted Langfuse means traces (which may contain sensitive data) never leave your infrastructure.
- Prometheus + Grafana for operational metrics (queue depth, worker count, error rates) — these are infrastructure metrics, not LLM-specific, and Langfuse doesn't replace Prometheus.

```python
# Instrumenting LangGraph with Langfuse
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

config = {
    "configurable": {"thread_id": task_id},
    "callbacks": [langfuse_handler],  # Automatic trace capture
}
result = await graph.ainvoke(input_state, config)
```

---

## E. The Deployment Architecture: Kubernetes-Native

For production at any meaningful scale, Kubernetes is the deployment substrate. The agent system maps onto Kubernetes primitives as follows:

```
Kubernetes Cluster
├── Namespace: agent-gateway
│   └── Deployment: api-server (FastAPI, 3 replicas, HPA on CPU)
│       └── Service: api-server-svc (ClusterIP)
│   └── Ingress: agent-ingress (TLS, auth, rate limit via Kong)
│
├── Namespace: agent-workers
│   └── Deployment: agent-worker (LangGraph workers, HPA on queue depth)
│       └── resources: requests.cpu: 1, requests.memory: 2Gi
│       └── env: OPENAI_API_KEY (from Secret), DATABASE_URL (from Secret)
│   └── ScaledObject: keda-queue-scaler (KEDA on Redis queue depth)
│       └── minReplicaCount: 2, maxReplicaCount: 50
│
├── Namespace: agent-tools
│   └── Deployment: web-search-mcp (HTTP+SSE, 3 replicas)
│   └── Deployment: code-exec-mcp (HTTP+SSE, 5 replicas, E2B backend)
│   └── Deployment: vector-db-mcp (HTTP+SSE, 2 replicas)
│
├── Namespace: agent-data
│   └── StatefulSet: postgres (with PVC, Postgres operator)
│   └── StatefulSet: redis (Redis Sentinel for HA)
│   └── Deployment: chroma (vector DB, optional)
│
└── Namespace: agent-observability
    └── Deployment: langfuse (self-hosted, Postgres backend)
    └── Deployment: prometheus-stack (Prometheus + Grafana + Alertmanager)
```

**KEDA (Kubernetes Event-Driven Autoscaling)** for queue-based scaling:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: agent-worker-scaler
spec:
  scaleTargetRef:
    name: agent-worker
  minReplicaCount: 2
  maxReplicaCount: 50
  triggers:
  - type: redis
    metadata:
      address: redis:6379
      listName: agent_tasks
      listLength: "5"  # 1 worker per 5 queued tasks
```

This auto-scales workers based on actual queue demand — zero overhead when idle, fast scale-out when traffic spikes.

---

## F. Security Architecture

### F.1 Zero-Trust Tool Access

Production agents should never have ambient authority. Every tool call is an explicit capability grant:

```
Principle: An agent should have access to exactly the tools required for its declared task, and no more.

Implementation:
  1. Agent identity is established at task start (user_id + task_id)
  2. User's permissions determine the tool allowlist
  3. Tool allowlist is embedded in the agent's compiled graph — not in runtime prompts
  4. Each MCP server validates the calling agent's identity and permissions
  5. All tool calls are logged to the immutable audit log
```

```python
# Tool allowlist enforced at graph compile time, not runtime
def build_graph_for_user(user: User) -> CompiledGraph:
    allowed_tools = get_tools_for_user(user)
    tool_node = ToolNode(allowed_tools)  # Only these tools available
    builder = StateGraph(AgentState)
    builder.add_node("tools", tool_node)
    # ... rest of graph
    return builder.compile(checkpointer=checkpointer)
```

### F.2 Prompt Injection Defense Layers

```
Layer 1: Input scanning (fast classifier on user input)
Layer 2: Structural separation (tool outputs in ToolMessage, not SystemMessage)
Layer 3: Tool output validation (schema check before injecting into context)
Layer 4: Plan validation (LLM safety review before execution)
Layer 5: Output filtering (PII redaction, toxicity check)
Layer 6: Audit log (post-hoc review of all agent actions)
```

### F.3 Secrets Management

```
Never:                          Always:
─────────────────────────────   ─────────────────────────────────────
API keys in code                API keys from environment variables
API keys in Docker images       Secrets from vault (AWS SM, Vault)
API keys in logs                Secret rotation via external secret operator
API keys in agent prompts       Secrets injected at container start
```

```yaml
# External Secrets Operator (Kubernetes)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agent-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
  target:
    name: agent-api-keys
  data:
  - secretKey: OPENAI_API_KEY
    remoteRef:
      key: /production/agent/openai-key
  - secretKey: TAVILY_API_KEY
    remoteRef:
      key: /production/agent/tavily-key
```

---

## G. The Continuous Improvement Loop

The architecture is only as good as its feedback loops. Production-grade systems build continuous improvement into the system design:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT LOOP                      │
│                                                                     │
│   Production Traffic                                                │
│        │                                                            │
│        ▼                                                            │
│   Langfuse traces ──────────────────────────────────────────────┐  │
│        │                                                         │  │
│        ▼                                                         │  │
│   Automated scoring (LLM-as-judge, assertions)                  │  │
│        │                                                         │  │
│        ├── Score ≥ threshold → archive as positive example       │  │
│        └── Score < threshold → route to human review queue      │  │
│                │                                                 │  │
│                ▼                                                 │  │
│         Human labeler reviews trace                             │  │
│                │                                                 │  │
│                ├── Label: correct → add to golden dataset        │  │
│                ├── Label: wrong → add to failure dataset          │  │
│                └── Label: needs edit → add edited version        │  │
│                         │                                        │  │
│                         ▼                                        │  │
│              Evaluation dataset grows                            │  │
│                         │                                        │  │
│                         ▼                                        │  │
│              CI runs evals on every PR                           │  │
│                         │                                        │  │
│              Quality gate: score ≥ baseline                      │  │
│                         │                                        │  │
│                         ▼                                        │  │
│              Merge → canary deploy → full rollout                │  │
│                         │                                        │  │
│                         └─────────────────────────────────────►─┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**The flywheel**: Each human review that labels a trace creates a permanent evaluation example. Over time, the evaluation suite grows to cover the actual failure modes seen in production — not hypothetical ones. This is the mechanism by which agents improve systematically rather than randomly.

---

## H. Architecture Summary: Why This Architecture

| Decision | What | Why |
|---|---|---|
| **Hierarchical supervisor** | Fast model routes; specialized agents execute | Cost-quality tradeoff. Routing is cheap; execution needs capability. |
| **Dual ledger (Task + Progress)** | Strategic plan separated from tactical tracking | Re-planning without discarding context. Prevents endless loops. |
| **MCP-first tools** | Tools as independent MCP server processes | Language independence, independent scaling, ecosystem reuse, no coupling. |
| **A2A delegation** | Cross-system agent communication | Standard protocol replaces bespoke integrations. Vendor-neutral interop. |
| **Postgres checkpointing** | Every state transition persisted | Durability, resumability, time-travel debug, human-in-the-loop. |
| **Layered memory** | Working / episodic / semantic / procedural | Match memory technology to access pattern. Avoid context stuffing. |
| **Three-model tier** | Fast/standard/reasoning by task | 10-50× cost reduction vs. using reasoning model everywhere. |
| **KEDA autoscaling** | Queue-depth-based worker scaling | Zero cost at idle, instant scale-out at peak. No wasted capacity. |
| **Langfuse (self-hosted)** | Full trace capture, self-hosted | Sensitive data never leaves infrastructure. OpenTelemetry-compatible. |
| **Eval flywheel** | Production traces → labeled dataset → CI gate | Systematic quality improvement. Prevents whack-a-mole regression. |
| **Zero-trust tools** | Allowlist at compile time, not runtime | Tool access is a security boundary, not a configuration option. |
| **Async + SSE/WS** | Non-blocking execution + streaming updates | UX: immediate response. Infrastructure: no held connections. |

This is the architecture that the industry's most successful production agent teams are converging on in 2025–2026. It is not the simplest architecture — but every component earns its complexity through a concrete production requirement.

---

## Additional References

13. Microsoft Research. *Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks* (2024). https://www.microsoft.com/en-us/research/blog/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/
14. Google Cloud. *Vertex AI Agent Engine Overview* (2025). https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview
15. LangChain. *LangGraph v0.2: Production Checkpointing* (2024). https://blog.langchain.dev/langgraph-v0-2/
16. AWS. *Strands Agents SDK* (2025). https://strandsagents.com/
17. Google. *Agent2Agent Protocol* (2025). https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
18. KEDA Project. *Kubernetes Event-Driven Autoscaling* (2025). https://keda.sh/
19. Temporal Technologies. *Durable Execution for Long-Running Workflows* (2025). https://temporal.io/
20. External Secrets Operator. *Kubernetes Secret Management* (2025). https://external-secrets.io/
