# Modern Agentic Architecture — Deep Dive

> **Compiled:** March 2026  
> **Sources:** ReAct (Yao et al., 2022), Reflexion (Shinn et al., 2023), Tree of Thoughts (Yao et al., 2023), CoALA (Sumers et al., 2024), AutoGen (Wu et al., 2023), Magentic-One (Fourney et al., 2024), LangGraph (LangChain, 2024), smolagents (Hugging Face, 2025), Anthropic MCP

---

## Table of Contents

1. [The Agency Spectrum](#1-the-agency-spectrum)
2. [The Fundamental Agent Loop](#2-the-fundamental-agent-loop)
3. [CoALA: A Cognitive Architecture Framework](#3-coala-a-cognitive-architecture-framework)
4. [Single-Agent Architectures](#4-single-agent-architectures)
   - 4.1 ReAct
   - 4.2 Reflexion
   - 4.3 Tree of Thoughts (ToT)
   - 4.4 Plan-and-Execute
   - 4.5 ReWOO
   - 4.6 LLMCompiler
   - 4.7 Code Agents
5. [Multi-Agent Architectures](#5-multi-agent-architectures)
   - 5.1 Supervisor Pattern
   - 5.2 Hierarchical Multi-Agent
   - 5.3 Swarm / Peer-to-Peer
   - 5.4 Pipeline / Sequential Handoff
   - 5.5 Magentic-One (Production Example)
6. [State Machine Architecture with LangGraph](#6-state-machine-architecture-with-langgraph)
7. [Action Space Taxonomy](#7-action-space-taxonomy)
8. [Code Actions vs JSON Tool Calls](#8-code-actions-vs-json-tool-calls)
9. [Human-in-the-Loop Architecture](#9-human-in-the-loop-architecture)
10. [Memory Architecture](#10-memory-architecture)
11. [Architecture Selection Guide](#11-architecture-selection-guide)
12. [Architecture Comparison Matrix](#12-architecture-comparison-matrix)
13. [References](#13-references)

---

## 1. The Agency Spectrum

Agency is not binary — it exists on a continuous spectrum based on how much control the LLM has over the program's execution flow.

| Agency Level | Pattern Name     | Description                                               | Code Structure                      |
| :----------: | ---------------- | --------------------------------------------------------- | ----------------------------------- |
|     ☆☆☆      | Simple Processor | LLM output has no impact on control flow                  | `process(llm_response)`             |
|     ★☆☆      | Router           | LLM output determines which branch to take                | `if llm_decision(): path_a()`       |
|     ★★☆      | Tool Caller      | LLM chooses which function to execute and with what args  | `run_fn(llm_tool, llm_args)`        |
|     ★★★      | Multi-Step Agent | LLM controls iteration — loop continues while LLM says so | `while llm_continue(): next_step()` |
|     ★★★      | Multi-Agent      | One agentic workflow triggers another agentic workflow    | `if llm_trigger(): run_agent()`     |

_Source: HuggingFace smolagents (2025)_

**Key design principle:** Regularize toward the lowest agency level that solves the problem reliably. Each step up the spectrum increases capability but also increases unpredictability, cost, and error surface.

---

## 2. The Fundamental Agent Loop

Every agentic architecture — regardless of complexity — is a variation of this core loop:

```
┌──────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                            │
│                                                              │
│   memory = [user_task]                                       │
│                                                              │
│   while llm_should_continue(memory):                         │
│       action = llm_get_next_action(memory)   ← LLM call     │
│       observation = execute_action(action)   ← Tool call     │
│       memory.append(action, observation)                     │
│                                                              │
│   return synthesize_final_answer(memory)                     │
└──────────────────────────────────────────────────────────────┘
```

The differences between architectures come from **how** each component is implemented:

- What goes into `memory` (messages, state, structured plan, scratchpad)
- How the LLM decides the `action` (single call, multi-turn, tree search)
- What `execute_action` can do (tool calls, code execution, spawning sub-agents)
- When and how the loop terminates (stopping condition, max iterations, critic approval)

---

## 3. CoALA: A Cognitive Architecture Framework

**CoALA** (_Cognitive Architectures for Language Agents_, Sumers et al., 2024) provides the most rigorous theoretical framework for understanding agent architectures. It draws on cognitive science and classical AI to describe language agents along three dimensions:

### 3.1 Memory Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                       │
│                                                              │
│  Working Memory        ←── Active context window            │
│  (in-context)               current task, history           │
│                                                              │
│  Episodic Memory       ←── Past interaction sequences       │
│  (external store)           retrieved via similarity search  │
│                                                              │
│  Semantic Memory       ←── World facts / user preferences   │
│  (external store)           retrieved as needed             │
│                                                              │
│  Procedural Memory     ←── LLM weights + agent code         │
│  (parametric)               rarely updated at runtime        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Action Space

CoALA organizes agent actions into four categories:

| Action Category           | Examples                                 | Description                       |
| ------------------------- | ---------------------------------------- | --------------------------------- |
| **Memory actions**        | read, write, retrieve, summarize         | Interact with the memory stores   |
| **Internal reasoning**    | chain-of-thought, reflection, planning   | Reasoning entirely within the LLM |
| **External grounding**    | web search, API calls, database queries  | Read from external environments   |
| **External side-effects** | file writes, email sends, code execution | Write to external environments    |

### 3.3 Decision-Making Process

The decision procedure forms the control flow:

```
                    ┌──────────────┐
                    │  PERCEIVE    │  ← sensor input (tool results, user msg)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  RETRIEVE    │  ← pull relevant memory
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    PLAN      │  ← reason about next action(s)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   EXECUTE    │  ← take action in environment
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    STORE     │  ← update memory with results
                    └──────────────┘
```

---

## 4. Single-Agent Architectures

### 4.1 ReAct (Reason + Act)

**Paper:** Yao et al., ICLR 2023  
**Core Idea:** Interleave _reasoning traces_ and _actions_ in a single prompt. Reasoning helps track and update the action plan; actions gather information to ground the reasoning.

```
┌─────────────────────────────────────────────────┐
│                  ReAct Loop                     │
│                                                 │
│  Thought: "I need the population of Tokyo"      │
│      │                                          │
│  Action: web_search("Tokyo population 2024")    │
│      │                                          │
│  Observation: "13.96 million (city proper)"     │
│      │                                          │
│  Thought: "I can now answer the question"       │
│      │                                          │
│  Action: finish("Tokyo population is 13.96M")  │
└─────────────────────────────────────────────────┘
```

**Strengths:**

- Simple, interpretable trajectories
- Grounding prevents hallucination (each action provides real-world feedback)
- 34% improvement over RL on ALFWorld; 10% on WebShop

**Weaknesses:**

- Requires one LLM call per tool invocation — expensive at scale
- Plans only one step at a time — can produce suboptimal trajectories
- No explicit backtracking

**When to use:** Tasks requiring flexible reasoning interleaved with real-world grounding, where the path can't be predetermined.

---

### 4.2 Reflexion

**Paper:** Shinn et al., NeurIPS 2023  
**Core Idea:** Agents _verbally reflect_ on their failures and maintain a _linguistic memory buffer_ of past reflections to improve on subsequent trials. Achieves 91% on HumanEval vs GPT-4 baseline of 80%.

```
┌─────────────────────────────────────────────────────────────┐
│                    Reflexion Loop                           │
│                                                             │
│  Trial 1:  Agent attempts task → fails                      │
│      │                                                      │
│  Reflect:  "I forgot to handle edge case X"                 │
│      │         → written to episodic memory buffer          │
│  Trial 2:  Agent attempts task with reflection in context   │
│      │         → better informed decision making            │
│  Reflect:  "Still failed — I need to check Y first"        │
│      │         → appended to memory buffer                  │
│  Trial 3:  Agent succeeds                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key properties:**

- Does **not** update model weights — all learning is in-context
- Works with scalar reward signals or free-form natural language feedback
- Memory buffer is bounded to prevent context overflow
- Particularly powerful for coding and sequential decision-making tasks

**Architecture components:**

1. **Actor** — generates actions and trajectories using ReAct or CoT
2. **Evaluator** — scores the generated trajectory (external reward or LLM judge)
3. **Self-Reflection model** — converts evaluation into linguistic critique stored in memory

---

### 4.3 Tree of Thoughts (ToT)

**Paper:** Yao et al., NeurIPS 2023  
**Core Idea:** Generalizes chain-of-thought by enabling the LLM to explore _multiple reasoning paths simultaneously_, self-evaluate, backtrack, and perform deliberate tree search.

```
                        [Initial State]
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           [Path A]       [Path B]       [Path C]
              │              │              │
          Score: 3       Score: 7       Score: 2
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         [B → B1]       [B → B2]       [B → B3]
              │              │              │
          Score: 6       Score: 9       Score: 4
                             │
                        [SOLUTION]
```

**Search strategies:**

- **Breadth-First Search (BFS)** — evaluate all paths at each level, keep top-k
- **Depth-First Search (DFS)** — explore deeply, backtrack on dead-ends

**Results:** Game of 24 — GPT-4 with CoT: 4% success; GPT-4 with ToT: 74% success.

**When to use:** Tasks requiring genuine exploration and backtracking (puzzles, proofs, creative writing) where greedy left-to-right generation consistently fails.

**Limitation:** Expensive — requires many LLM calls per decision. Rarely practical for high-throughput production use.

---

### 4.4 Plan-and-Execute

**Paper:** Wang et al., 2023 (Plan-and-Solve Prompting); also BabyAGI  
**Core Idea:** Separate the _planning_ phase from the _execution_ phase. A large "planner" LLM generates a complete multi-step plan; a lighter "executor" LLM or deterministic code runs each step.

```
┌──────────────────────────────────────────────────────────────┐
│                  Plan-and-Execute Architecture               │
│                                                              │
│  User Task                                                   │
│      │                                                       │
│  ┌───▼───────────────────────────────────────────────────┐  │
│  │   PLANNER (large model, called once)                  │  │
│  │   Output: ["step 1", "step 2", "step 3", ...]         │  │
│  └───┬───────────────────────────────────────────────────┘  │
│      │                                                       │
│  For each step:                                              │
│  ┌───▼───────────────────────────────────────────────────┐  │
│  │   EXECUTOR (smaller model or deterministic code)      │  │
│  │   Calls tools, produces observation                   │  │
│  └───┬───────────────────────────────────────────────────┘  │
│      │                                                       │
│  ┌───▼───────────────────────────────────────────────────┐  │
│  │   RE-PLANNER (optional — updates plan on new info)    │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Advantages over ReAct:**

- Reduces expensive large-LLM calls (planner runs once; executor can be cheap)
- Forces holistic task reasoning upfront
- Easier to inspect and debug (plan is explicit)

**Disadvantages:**

- Rigid plans may not adapt well to unexpected tool outputs
- Re-planning requires another expensive LLM call

---

### 4.5 ReWOO (Reasoning WithOut Observations)

**Paper:** Xu et al., 2023  
**Core Idea:** Extends Plan-and-Execute by allowing the planner to reference future execution outputs via _variable substitution_ (`#E1`, `#E2`, …), enabling serial execution without re-planning at each step.

```
Planner output:
  Plan: Find the Super Bowl teams
  E1: Search[Who is in the Super Bowl this year?]

  Plan: Find quarterback for each team
  E2: LLM[Quarterback for first team of #E1]
  E3: LLM[Quarterback for second team of #E1]

  Plan: Look up stats
  E4: Search[Stats for #E2]
  E5: Search[Stats for #E3]

Worker executes E1..E5 sequentially, substituting variables.
Solver synthesizes final answer from all E# outputs.
```

**Three components:**

1. **Planner** — emits plan + variable-referenced task list in one pass
2. **Worker** — executes tasks serially, substituting `#E` variables with prior outputs
3. **Solver** — integrates all outputs into a coherent final answer

**Advantage:** Each task gets precisely the context it needs — no bloated context with irrelevant tool outputs.

---

### 4.6 LLMCompiler

**Paper:** Kim et al., 2023  
**Core Idea:** Plans as a **DAG** (directed acyclic graph). Tasks without dependencies run in **parallel**; a task fetching unit schedules execution as dependencies are satisfied.

```
Planner emits (streamed):
  Task 1: Search["Super Bowl teams"]
  Task 2: LLM["Quarterback for team A from ${1}"]  (depends on 1)
  Task 3: LLM["Quarterback for team B from ${1}"]  (depends on 1)
  Task 4: Search["Stats for ${2}"]                  (depends on 2)
  Task 5: Search["Stats for ${3}"]                  (depends on 3)

Execution DAG:
  [Task 1]
     ├──→ [Task 2] ──→ [Task 4]
     └──→ [Task 3] ──→ [Task 5]

Tasks 2 & 3 run in parallel after Task 1.
Tasks 4 & 5 run in parallel after their respective deps.
```

**Components:**

1. **Planner** — streams a DAG with variable references and dependency lists
2. **Task Fetching Unit** — schedules tasks as their dependencies complete
3. **Joiner** — decides whether to return final answer or trigger re-planning

**Reported speedup:** 3.6× over serial execution in parallel tool calling scenarios.

---

### 4.7 Code Agents

**Core Idea:** Instead of emitting JSON tool-call specs, the agent writes _executable Python code_ as its action. The code is executed in a sandbox; stdout/return values become the observation.

```python
# JSON Tool Call (traditional)
{"tool": "search", "args": {"query": "population of Tokyo"}}

# Code Agent action
result = web_search("population of Tokyo")
capital = result.split("million")[0].strip()
print(f"Population: {capital} million")
```

**Advantages of code actions (Executable Code Actions paper, 2024):**

- **Composability** — functions, loops, conditionals are native constructs
- **Object management** — intermediate results stored in variables naturally
- **Generality** — any computation expressible in Python is achievable
- **Training data alignment** — LLMs are extensively trained on code

**Security requirement:** Must run in an isolated sandbox (E2B cloud, Docker, WASM, RestrictedPython).

**Frameworks:** HuggingFace `smolagents` (CodeAgent class), OpenAI Code Interpreter.

---

## 5. Multi-Agent Architectures

### 5.1 Supervisor Pattern

A central **Supervisor** agent routes tasks to specialized **Worker** agents and synthesises their outputs. The Supervisor maintains the global task context and decides which worker to invoke at each step.

```
                        ┌──────────────────┐
                        │   SUPERVISOR     │
                        │  (orchestrator)  │
                        └────────┬─────────┘
                                 │ routes based on task state
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼──────┐  ┌────────▼───────┐  ┌──────▼──────────┐
   │  Research Agent │  │  Coder Agent   │  │  Writer Agent   │
   │  (web search)   │  │  (code exec)   │  │  (synthesis)    │
   └─────────────────┘  └───────────────┘  └─────────────────┘
```

**LangGraph implementation:** Supervisor uses `structured_output` to emit `{"next": "research_agent" | "coder_agent" | "FINISH"}`. Each worker node returns to the supervisor after completing its step.

**Best for:** Tasks where different steps genuinely require different expertise, and the Supervisor can reliably determine which specialist to invoke.

---

### 5.2 Hierarchical Multi-Agent

Extends the Supervisor pattern with multiple levels. Sub-supervisors manage domain-specific teams; a top-level supervisor manages sub-supervisors.

```
                    ┌─────────────────┐
                    │  TOP SUPERVISOR │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
   ┌──────────▼──────┐  ┌────▼───────┐  ┌──▼──────────────┐
   │  Research Team  │  │ Code Team  │  │  Writing Team   │
   │   Supervisor    │  │ Supervisor │  │   Supervisor    │
   └─────────────────┘  └───────────┘  └─────────────────┘
         │                   │                 │
   ┌─────┴─────┐       ┌─────┴─────┐     ┌────┴───────┐
   │Web │ Arxiv│       │Python│SQL │     │Draft│Edit  │
   └────┘ └────┘       └──────┘└───┘     └─────┘└──────┘
```

**Used in:** Magentic-One's nested loop architecture, enterprise workflow automation.

---

### 5.3 Swarm / Peer-to-Peer

Agents communicate directly without a central coordinator. Each agent decides whether to handle a task itself or hand off to another agent. OpenAI's Swarm library popularised this pattern.

```
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ Agent A  │ ──────→ │ Agent B  │ ──────→ │ Agent C  │
   │          │ ←────── │          │ ←────── │          │
   └──────────┘         └──────────┘         └──────────┘
        │                                          │
        └──────────────────────────────────────────┘
                    (direct handoff)
```

**Handoff mechanism:** An agent calls a special `transfer_to_agent_X()` function, which redirects control flow to the target agent along with relevant context.

**Best for:** Customer service routing, triage systems, domain-specialist networks where routing rules are well-defined but numerous.

**Caution:** Without a coordinator, global state tracking becomes challenging; harder to debug and audit.

---

### 5.4 Pipeline / Sequential Handoff

Agents are arranged in a fixed sequence. Each agent's output is the next agent's input. The simplest multi-agent pattern — essentially prompt chaining at the agent level.

```
   User Input
       │
   ┌───▼──────────┐
   │  Intake Agent│  ← validates and structures the request
   └───┬──────────┘
       │
   ┌───▼──────────┐
   │ Research Agent│ ← gathers information
   └───┬──────────┘
       │
   ┌───▼──────────┐
   │ Analysis Agent│ ← synthesizes findings
   └───┬──────────┘
       │
   ┌───▼──────────┐
   │  Writer Agent│  ← produces final output
   └───┬──────────┘
       │
   Final Output
```

**Best for:** Well-defined, linear workflows (e.g., document processing, data transformation pipelines).

**Limitation:** Cannot handle tasks requiring iteration or re-work without explicit loop-back edges.

---

### 5.5 Magentic-One — Production Architecture

**Paper:** Fourney et al., Microsoft Research, 2024  
**Benchmark:** State-of-the-art on GAIA, without task-specific modifications.

```
┌────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                            │
│                                                                │
│  OUTER LOOP (Task Ledger)                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  facts | guesses | high-level plan                       │ │
│  │  Updated when inner loop detects "stall" (no progress    │ │
│  │  after N steps)                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  INNER LOOP (Progress Ledger)                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  current assignments | progress per agent                │ │
│  │  Decision: Task complete? → EXIT                         │ │
│  │            Progress being made? → continue               │ │
│  │            Stall count > 2? → escalate to outer loop     │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────┬──────────────────────────────────────────┬─────────┘
           │                                          │
    handoff to specialist                      handoff to specialist
           │                                          │
┌──────────▼──────────┐                    ┌──────────▼──────────┐
│     WEB SURFER      │                    │    FILE SURFER      │
│  Browse internet    │                    │  Navigate PDFs,     │
│  navigate pages,    │                    │  PPTXs, audio,      │
│  fill forms         │                    │  images             │
└─────────────────────┘                    └─────────────────────┘

┌──────────────────────┐                   ┌──────────────────────┐
│       CODER          │                   │  COMPUTER TERMINAL   │
│  Write & analyze     │ ─── executes ───→ │  Execute code,       │
│  Python, C++, SQL    │                   │  return output       │
└──────────────────────┘                   └──────────────────────┘
```

**Key architectural decisions:**

- **Dual-loop design** prevents both infinite micro-cycles (inner loop) and permanent high-level failures (outer loop rescues via re-planning)
- **Stall detection** — if 3 consecutive inner loop steps show no progress, escalate to outer loop
- **Modular specialists** — each agent has a single, narrow responsibility; easy to add/remove
- **Plug-and-play** — new specialist agents can be added without changing orchestrator logic

---

## 6. State Machine Architecture with LangGraph

LangGraph models agents as **typed state graphs** — explicit state machines where the state transitions are driven by LLM decisions. This gives developers fine-grained control over agent behaviour.

### 6.1 Core Abstractions

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Model                         │
│                                                             │
│  StateGraph(AgentState)                                     │
│       │                                                     │
│       ├── Nodes: functions that read state → emit updates   │
│       │         (supervisor, planner, executor, critic, …)  │
│       │                                                     │
│       ├── Edges: always-on transitions between nodes        │
│       │         ("planner" → "executor")                    │
│       │                                                     │
│       ├── Conditional Edges: LLM-driven routing             │
│       │         ("supervisor" → planner | end)              │
│       │                                                     │
│       └── Checkpointer: SQLite / Redis / Postgres           │
│                         (persists state per thread_id)      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 State Design

State is a `TypedDict` that acts as the agent's shared memory. Node functions receive the full state and return a partial update:

```python
class AgentState(TypedDict):
    # append-only via add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]
    # overwritten each iteration
    plan: list[str]
    current_step: int
    critique: str
    iteration: int
    final_output: str
```

Two state update modes:

- **Overwrite** — node returns new value; previous is replaced (e.g., `current_step`)
- **Append** — node returns delta; reducer merges it (e.g., `messages` via `add_messages`)

### 6.3 Why Cycles Matter

Traditional data pipelines (Airflow, DAGs) are acyclic — they cannot loop. Agent runtimes _require_ cycles: the LLM must be able to call a tool, receive a result, then decide to call another tool or finish. LangGraph's key innovation is making cycles first-class and type-safe.

```
           ┌──────────────────────────────────┐
           │                                  │
   LLM ───→ ToolNode ───→ LLM ───→ ToolNode ─┘
     │
     └─→  END
```

### 6.4 Checkpointing and Resumability

Every state transition is checkpointed to a persistent store. This enables:

- **Resume** — restart a run from the last good state after a crash
- **Time-travel** — replay from any historical checkpoint
- **Human-in-the-loop** — pause execution, get approval, continue from exact state
- **Branching** — fork a checkpoint to explore alternative execution paths

---

## 7. Action Space Taxonomy

A comprehensive taxonomy of all actions an agent can perform:

```
ACTION SPACE
│
├── MEMORY ACTIONS
│   ├── read   — retrieve from working / episodic / semantic memory
│   ├── write  — store to memory store
│   └── delete — remove from memory store
│
├── REASONING ACTIONS (internal, no tool call)
│   ├── chain-of-thought   — step-by-step reasoning in scratchpad
│   ├── reflection         — self-critique of prior output
│   ├── planning           — generate sequence of future steps
│   └── summarization      — compress long context
│
├── READ ACTIONS (external, no side-effect)
│   ├── web_search         — query search engine
│   ├── retrieval          — query vector/keyword database
│   ├── http_get           — call external REST API (read)
│   ├── read_file          — read from filesystem
│   └── observe_env        — receive sensor data / UI state
│
└── WRITE ACTIONS (external, side-effect)
    ├── http_post/put/del  — mutate external APIs
    ├── write_file         — modify filesystem
    ├── execute_code       — run code in sandbox
    ├── send_email/message — communicate externally
    ├── spawn_agent        — start a sub-agent
    └── browser_action     — click, type, navigate
```

**Risk gradient:** Memory actions < Reasoning actions < Read actions < Write actions  
Gate write actions with explicit confirmation and audit trails.

---

## 8. Code Actions vs JSON Tool Calls

Two dominant paradigms for expressing agent actions:

### JSON Tool Calls (Standard Function Calling)

```json
{
  "tool": "web_search",
  "args": { "query": "Tokyo population 2024" }
}
```

### Code Actions

```python
result = web_search("Tokyo population 2024")
population_num = float(result.split("million")[0].strip().split()[-1])
formatted = f"{population_num:.1f} million"
print(f"Tokyo population: {formatted}")
```

### Comparison

| Dimension                  | JSON Tool Calls                    | Code Actions                       |
| -------------------------- | ---------------------------------- | ---------------------------------- |
| **Composability**          | ❌ One tool per call               | ✅ Loops, conditionals, functions  |
| **Object management**      | ❌ No variable state               | ✅ Python variables                |
| **Parallelism**            | ⚠️ Requires explicit parallel spec | ✅ `asyncio`, `ThreadPoolExecutor` |
| **Debugging**              | ✅ Structured, auditable           | ⚠️ Requires code tracing           |
| **Security**               | ✅ Constrained to declared tools   | ⚠️ Requires sandboxing             |
| **LLM training alignment** | ✅ Widely supported                | ✅ Code is common in training data |
| **Expressiveness**         | ❌ Limited to declared functions   | ✅ Full Turing completeness        |

**Verdict:** Code actions are more expressive; JSON tool calls are safer by default. Choose based on the complexity of compositions needed and the trust level of the execution environment.

---

## 9. Human-in-the-Loop Architecture

Human oversight is crucial for agentic systems with write access. There are three patterns:

### 9.1 Approval Gate (Interrupt-Before)

The agent pauses before executing a risky action and waits for explicit human approval.

```
Agent decides to call delete_database()
         │
         ▼
  ┌──────────────────────────────┐
  │  PAUSE — Await Human Input   │
  │                              │
  │  "Agent wants to delete DB.  │
  │   Approve? [yes/no]"         │
  └──────────────┬───────────────┘
                 │
          yes ───┤─── no
                 │         │
         Execute │    Abort/Revise
```

### 9.2 Review Gate (Interrupt-After)

The agent completes work, then a human reviews and either accepts, rejects, or requests edits before the result is used downstream.

```
Agent generates → Draft → Human Review → Accept/Edit → Publish
```

### 9.3 Escalation on Uncertainty

The agent proceeds autonomously but escalates specific decisions to a human when its confidence falls below a threshold.

```
Agent confidence > 0.8 → proceed autonomously
Agent confidence < 0.8 → pause and ask human
```

### 9.4 Checkpoint-Based Review

Using LangGraph's checkpointing, the system can pause at predefined breakpoints (e.g., "after plan is generated", "before first write action"), show the current state to a human, and resume with or without modifications.

**Implementation:**

```python
# Interrupt after planner runs, before executor
graph.compile(
    checkpointer=checkpointer,
    interrupt_after=["planner"],  # or interrupt_before=["executor"]
)
```

---

## 10. Memory Architecture

A detailed breakdown of how memory is implemented in production systems:

```
┌──────────────────────────────────────────────────────────────────┐
│                   MEMORY ARCHITECTURE                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              IN-CONTEXT (Working Memory)                │    │
│  │  • Current conversation / messages list                 │    │
│  │  • Active plan, tool results, agent state               │    │
│  │  • Bounded by context window (128K-1M tokens)           │    │
│  │  • Lost at session end unless explicitly stored         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EXTERNAL STORES                            │    │
│  │                                                         │    │
│  │  Episodic (Vector DB)    Semantic (Key-Value / Graph)   │    │
│  │  • Past trajectories     • User preferences             │    │
│  │  • Retrieved by          • Domain facts                 │    │
│  │    similarity            • Org knowledge                │    │
│  │  → Few-shot prompting    → System prompt injection      │    │
│  │                                                         │    │
│  │  Procedural (Code)       Cache (KV store)               │    │
│  │  • System prompts        • Deterministic tool results   │    │
│  │  • Graph structure       • Embedding cache              │    │
│  │  • Agent code            • LLM response cache           │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Memory Update Strategies

| Strategy          | Trigger                                | Latency           | Best For                         |
| ----------------- | -------------------------------------- | ----------------- | -------------------------------- |
| **Hot-path**      | Agent calls `remember()` tool mid-turn | +latency per turn | Urgent facts the agent needs NOW |
| **Background**    | Async process after turn completion    | None              | Non-urgent personalization       |
| **Scheduled**     | Batch job (e.g., nightly)              | None              | Summary / compression            |
| **User-feedback** | User marks interaction as good/bad     | None              | Episodic preference learning     |

### Context Window Management

As agent sessions grow long, context management becomes critical:

1. **Sliding window** — drop oldest messages, keep recent N
2. **Summarization** — compress old messages into a summary node
3. **Message trimming** — remove tool messages after extracting key results
4. **Retrieval augmentation** — store old context externally; retrieve relevant chunks

---

## 11. Architecture Selection Guide

Use this decision tree to select the right architecture:

```
Is the task path fully predetermined?
│
├── YES → Use deterministic code or simple prompt chain
│         (NOT an agent — don't over-engineer)
│
└── NO → Can the task be decomposed into predictable phases?
         │
         ├── YES → Plan-and-Execute or ReWOO
         │         (separate planning from execution)
         │
         └── NO → Does the task require genuine exploration
                  or backtracking?
                  │
                  ├── YES → Tree of Thoughts
                  │         (expensive but thorough)
                  │
                  └── NO → Does the task require multiple
                           specialised skills?
                           │
                           ├── YES → Multi-Agent (Supervisor
                           │         or Hierarchical)
                           │
                           └── NO → ReAct
                                    (simple interleaved
                                     reasoning + action)
```

### Checklist Before Adding Complexity

- [ ] Have you verified a simpler solution is insufficient?
- [ ] Do you have a hard iteration cap to prevent runaway loops?
- [ ] Are all write actions gated with appropriate confirmation?
- [ ] Is every LLM call and tool call traced and observable?
- [ ] Have you tested in a sandboxed environment?
- [ ] Can the system resume from a checkpoint after failure?
- [ ] Is the maximum cost per run bounded and acceptable?

---

## 12. Architecture Comparison Matrix

| Architecture               | LLM Calls      | Parallelism       | Backtracking               | Interpretability | Cost        | Complexity |
| -------------------------- | -------------- | ----------------- | -------------------------- | ---------------- | ----------- | ---------- |
| **Simple Chain**           | N (fixed)      | ❌                | ❌                         | ✅ High          | Low         | Low        |
| **ReAct**                  | 1 per step     | ❌                | ⚠️ Implicit                | ✅ High          | Medium      | Low        |
| **Reflexion**              | 1 + reflection | ❌                | ✅ Via retry               | ✅ High          | Medium      | Medium     |
| **Tree of Thoughts**       | N×branching    | ✅ Parallel paths | ✅ Explicit                | ⚠️ Medium        | High        | High       |
| **Plan-and-Execute**       | 1 + N small    | ❌                | ⚠️ Re-plan                 | ✅ High          | Medium      | Medium     |
| **ReWOO**                  | 1 + N          | ❌ Serial         | ❌                         | ✅ High          | Medium      | Medium     |
| **LLMCompiler**            | 1 + parallel   | ✅ DAG parallel   | ⚠️ Re-plan                 | ⚠️ Medium        | Medium-High | High       |
| **Supervisor Multi-Agent** | 1 + N workers  | ⚠️ Optional       | ⚠️ Supervisor can re-route | ✅ High          | High        | High       |
| **Swarm**                  | Per agent      | ✅                | ⚠️ Handoff                 | ⚠️ Medium        | High        | Medium     |
| **Hierarchical**           | Many           | ✅ Team-level     | ✅ Supervisor              | ⚠️ Medium        | Highest     | Highest    |

---

## 13. References

| Paper / Resource                                     | Authors        | Year | Link                                                         |
| ---------------------------------------------------- | -------------- | ---- | ------------------------------------------------------------ |
| ReAct: Synergizing Reasoning and Acting              | Yao et al.     | 2022 | https://arxiv.org/abs/2210.03629                             |
| Reflexion: Language Agents with Verbal Reinforcement | Shinn et al.   | 2023 | https://arxiv.org/abs/2303.11366                             |
| Tree of Thoughts                                     | Yao et al.     | 2023 | https://arxiv.org/abs/2305.10601                             |
| Plan-and-Solve Prompting                             | Wang et al.    | 2023 | https://arxiv.org/abs/2305.04091                             |
| ReWOO: Reasoning WithOut Observations                | Xu et al.      | 2023 | https://arxiv.org/abs/2305.18323                             |
| LLMCompiler                                          | Kim et al.     | 2023 | https://arxiv.org/abs/2312.04511                             |
| Executable Code Actions Elicit Better LLM Agents     | Wang et al.    | 2024 | https://huggingface.co/papers/2402.01030                     |
| CoALA: Cognitive Architectures for Language Agents   | Sumers et al.  | 2024 | https://arxiv.org/abs/2309.02427                             |
| AutoGen: Multi-Agent Conversations                   | Wu et al.      | 2023 | https://arxiv.org/abs/2308.08155                             |
| Magentic-One                                         | Fourney et al. | 2024 | https://aka.ms/magentic-one-report                           |
| AgentBoard: Analytical Evaluation                    | Ma et al.      | 2024 | https://arxiv.org/abs/2401.13178                             |
| LangGraph Introduction                               | LangChain      | 2024 | https://blog.langchain.dev/langgraph/                        |
| smolagents                                           | HuggingFace    | 2025 | https://huggingface.co/blog/smolagents                       |
| Model Context Protocol                               | Anthropic      | 2024 | https://modelcontextprotocol.io                              |
| Building Effective Agents                            | Anthropic      | 2024 | https://www.anthropic.com/research/building-effective-agents |
