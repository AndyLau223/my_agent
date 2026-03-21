# Modern Agentic Engineering — Research Synthesis

> **Compiled:** March 2026  
> **Sources:** Anthropic Research, Chip Huyen (*AI Engineering*, 2025), Andrew Ng / DeepLearning.AI, LangChain Engineering Blog, Microsoft Research (Magentic-One), Model Context Protocol (Anthropic)

---

## Table of Contents

1. [What Is an Agentic System?](#1-what-is-an-agentic-system)
2. [Core Building Blocks](#2-core-building-blocks)
3. [The Four Agentic Design Patterns](#3-the-four-agentic-design-patterns)
4. [Architectural Topologies](#4-architectural-topologies)
5. [Planning in Depth](#5-planning-in-depth)
6. [Memory for Agents](#6-memory-for-agents)
7. [Tool Design and the MCP Standard](#7-tool-design-and-the-mcp-standard)
8. [Multi-Agent Systems](#8-multi-agent-systems)
9. [Production Engineering Concerns](#9-production-engineering-concerns)
10. [Framework Landscape](#10-framework-landscape)
11. [Key Takeaways & Principles](#11-key-takeaways--principles)
12. [References](#12-references)

---

## 1. What Is an Agentic System?

An **agent** is anything that can *perceive its environment* and *act upon that environment*. In the context of LLM-powered systems, AI is the "brain" that processes tasks, plans sequences of actions, invokes tools, and determines when the goal has been accomplished.

Anthropic draws a crisp architectural distinction:

| Concept | Definition |
|---------|-----------|
| **Workflow** | LLMs and tools orchestrated through *predefined code paths*. Predictable, consistent. |
| **Agent** | LLMs *dynamically direct* their own processes and tool usage. Flexible, model-driven. |

Both are called **agentic systems**. The right choice depends on task structure: workflows excel at well-defined, repeatable processes; agents are better when the path to a solution cannot be predetermined.

> **Rule of thumb:** Start with the simplest solution—often a single well-crafted LLM call. Only escalate to agentic complexity when simpler approaches measurably fall short.

---

## 2. Core Building Blocks

Every agentic system is assembled from three primitives:

### 2.1 The Augmented LLM

A bare LLM extended with:
- **Retrieval** — access to external knowledge (RAG, web search, SQL)
- **Tools** — callable functions that act on the world
- **Memory** — persistence of state across turns and sessions

The augmented LLM is the atomic unit; everything else is composition.

### 2.2 Tools

Tools give agents the ability to *perceive* (read actions) and *act* (write actions):

| Category | Examples |
|----------|---------|
| Knowledge augmentation | Web search, vector retrieval, SQL executor, email reader |
| Capability extension | Calculator, code interpreter, image captioner, translator |
| Write / side-effect actions | Database mutation, email sending, API POST, file writes |

Write actions are the most powerful but also the most dangerous — they can have irreversible real-world consequences.

### 2.3 Memory

LLMs have no innate memory. Agents must be designed to persist and retrieve relevant context. (See [Section 6](#6-memory-for-agents) for a full taxonomy.)

---

## 3. The Four Agentic Design Patterns

Andrew Ng (DeepLearning.AI) identified four foundational patterns. GPT-3.5 wrapped in agentic loops achieves **95.1%** on HumanEval versus 48.1% zero-shot — demonstrating that the *loop matters more than the model*.

### Pattern 1: Reflection

The agent examines its own output and iteratively refines it. Analogous to a human re-reading and editing their own work.

```
Generate → Self-Critique → Revise → (repeat until accepted)
```

- Most reliable and predictable of the four patterns
- Best for: writing, code generation, analysis with clear quality criteria

### Pattern 2: Tool Use

The agent is given access to callable functions. On each reasoning step it decides *which tool to call*, with what arguments, and then integrates the result.

- Implemented via *function calling* / *tool binding* in modern LLM APIs
- Key insight: more tools ≠ better — overly large tool inventories degrade selection quality

### Pattern 3: Planning

The agent explicitly decomposes a complex goal into an ordered sequence of sub-tasks *before* executing any of them. This forces reasoning over the entire problem upfront.

```
Task → [Planner] → Plan (step list) → [Executor] → Results → [Re-planner or Done]
```

- Decoupling planning from execution prevents runaway, fruitless execution
- Plans can be validated (heuristic or LLM-judge) before committing

### Pattern 4: Multi-Agent Collaboration

Multiple specialized agents work together, each optimized for a distinct role. Long-context handling degrades in a single agent — splitting attention across agents maintains quality.

```
Orchestrator → [Specialist A | Specialist B | Specialist C] → Synthesizer
```

- Ablation studies (AutoGen paper) show multi-agent consistently outperforms single-agent
- Design challenge: emergent, hard-to-predict behavior when agents interact freely

---

## 4. Architectural Topologies

### 4.1 Linear Pipeline (Prompt Chaining)

```
Input → LLM₁ → LLM₂ → LLM₃ → Output
```

Best for: tasks that decompose into fixed, sequential subtasks.  
Trade-off: latency scales linearly; intermediate gates can catch errors early.

### 4.2 Routing

```
Input → Classifier → [Route A | Route B | Route C]
```

Best for: distinct input categories that benefit from specialized handling (e.g., customer support triage).

### 4.3 Parallelization

```
Input → [Worker₁ | Worker₂ | Worker₃] → Aggregator
```

Two variants:
- **Sectioning** — independent subtasks run concurrently
- **Voting** — same task run N times; majority vote or best-of-N selected

### 4.4 Orchestrator–Workers

A central LLM *dynamically* decomposes the task and delegates to worker agents. Unlike parallelization, the number and nature of subtasks are not predetermined.

```
Orchestrator (plans dynamically)
    ├── Worker A
    ├── Worker B
    └── Worker C (spun up as needed)
```

### 4.5 Evaluator–Optimizer Loop

```
Generator → Evaluator → [Approved → Done | Rejected → Generator]
```

Best for: tasks where iterative refinement gives measurable value and clear evaluation criteria exist.

### 4.6 Hierarchical (Supervisor + Specialists)

The architecture implemented in this project:

```
Supervisor
    └── Planner → Executor (+ Tools) → Critic
                        ↑_______________|  (if rejected)
```

- Supervisor decides when the task is complete or needs re-planning
- Critic enforces quality; loops back to Planner up to N times

---

## 5. Planning in Depth

Planning is fundamentally a **search problem**: enumerate possible paths, estimate their outcomes, and select the most promising one.

### 5.1 ReAct (Reason + Act)

The canonical baseline. The model interleaves `Thought → Action → Observation` in a tight loop.

```
Thought: I need to find the population of Tokyo.
Act: web_search("Tokyo population 2024")
Observation: 13.96 million (city proper)
Thought: Now I can answer the question.
```

**Limitation:** requires an LLM call per tool invocation; plans only one step ahead.

### 5.2 Plan-and-Execute

Separates planning from execution into two distinct phases:
1. **Planner** generates a complete multi-step plan
2. **Executor** runs each step (optionally with a lighter/cheaper model)

After execution, an optional *re-plan* step updates the plan based on intermediate results.  
**Benefit:** reduces expensive LLM calls; forces holistic reasoning upfront.

### 5.3 ReWOO (Reasoning WithOut Observations)

Extends Plan-and-Execute by allowing the planner to reference future step outputs via variables (`#E1`, `#E2`, …). Steps can then be executed serially without a re-plan at each step.

```
Plan: Identify super bowl teams
E1: Search[Who is in the Super Bowl?]
Plan: Get quarterback names
E2: LLM[Quarterback for first team of #E1]
E3: LLM[Quarterback for second team of #E1]
```

**Benefit:** each execution step gets precisely the context it needs.

### 5.4 LLMCompiler

Plans as a **DAG** (directed acyclic graph) enabling parallel task execution. A streaming planner emits tasks; a task-fetching unit schedules them as their dependencies are satisfied. Claims 3.6× speedup over serial execution.

### 5.5 Key Planning Principles

1. **Decouple planning from execution** — validate the plan before committing expensive tool calls
2. **Compound error risk** — at 95% per-step accuracy, a 10-step plan has ~60% overall accuracy; 100 steps → 0.6%
3. **Backtracking** — agents can effectively backtrack by re-planning after observing a dead-end; this is not fundamentally blocked by auto-regressive generation
4. **Human-in-the-loop** — for risky write actions, require explicit human approval before execution

---

## 6. Memory for Agents

LLMs have no intrinsic memory. Memory must be deliberately engineered, and its optimal shape is **application-specific**.

Based on the CoALA framework (Sumers et al., 2024), there are four memory types mirroring human cognition:

### 6.1 Procedural Memory

How to perform tasks — embedded in LLM weights and agent code (system prompts, graph structure). Rarely updated at runtime, though some systems allow agents to rewrite their own system prompts.

### 6.2 Semantic Memory

Long-term factual knowledge. In agents, used for **personalisation**: extracting user preferences, domain facts, and organisational knowledge from conversations and storing them for retrieval in future sessions.

**Update pattern:** LLM extracts structured facts → stored in a key-value or vector store → injected into system prompt on next session.

### 6.3 Episodic Memory

Recall of specific past interactions. Enables *few-shot prompting* from real prior examples: when a user has a similar request to a historical one, relevant past trajectories are retrieved and prepended.

**Update pattern:** Store successful agent trajectories → dynamic few-shot retrieval at inference time.

### 6.4 Working Memory (In-Context)

The current conversation / agent state held in the active context window. Lost at session end unless explicitly persisted to one of the above stores.

### 6.5 Memory Update Strategies

| Strategy | Latency Impact | Separation of Concerns |
|----------|---------------|----------------------|
| **Hot-path** (agent decides to remember before responding) | Adds latency | Memory + agent logic coupled |
| **Background** (post-turn async process) | No latency | Clean separation |
| **User-feedback triggered** | None | Best for episodic / preference learning |

---

## 7. Tool Design and the MCP Standard

### 7.1 Principles for Tool Design

Good tools are as important as good prompts. Anthropic's guidance:

1. **Minimal surface area** — each tool should do one thing well
2. **Self-documenting** — docstrings are the agent's "manual"; write them for the LLM, not for humans
3. **Safe by default** — prefer read-only tools; gate write actions with explicit confirmation
4. **Predictable** — same inputs should produce the same outputs; avoid side-effectful defaults
5. **Right inventory size** — too many tools degrade selection quality (the "paradox of choice" for LLMs)

### 7.2 Function Calling

Modern LLM APIs (OpenAI, Anthropic, Gemini) expose **function calling** / **tool use** natively:
- The LLM outputs a structured `tool_call` object (name + JSON arguments) instead of free text
- The host application executes the function and returns the result
- The LLM resumes with the observation injected into context

### 7.3 Model Context Protocol (MCP)

MCP (released by Anthropic, Nov 2024) is an **open standard** analogous to USB-C for AI tools:

- Defines a server/client protocol for exposing tools, resources, and prompts to any LLM application
- Supported by: Claude, ChatGPT, VS Code Copilot, Cursor, and many others
- Enables "build once, integrate everywhere" — a tool server works with any MCP-compatible agent
- Dramatically reduces the N×M integration problem (N agents × M tool providers → N + M)

```
Agent (MCP Client)  ←──── MCP Protocol ────→  Tool Server (MCP Server)
```

MCP is rapidly becoming the de facto standard for tool interoperability in production agentic systems.

---

## 8. Multi-Agent Systems

### 8.1 Why Multiple Agents?

- **Context window limits** — long, complex inputs degrade LLM comprehension; focused per-agent context improves it
- **Specialisation** — different roles can have different system prompts, tools, even different models (e.g., GPT-4o for planning, GPT-4o-mini for execution)
- **Parallelism** — independent subtasks run concurrently, reducing wall-clock time
- **Modularity** — agents can be added/removed without redesigning the whole system

### 8.2 Communication Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Hierarchical** | Supervisor routes to sub-agents | Complex tasks with clear role separation |
| **Peer-to-peer / Swarm** | Agents message each other directly | Collaborative, emergent problem-solving |
| **Blackboard** | Agents read/write to shared state | Asynchronous, parallel contribution |
| **Sequential handoff** | Output of one becomes input to next | Linear pipelines, assembly-line workflows |

### 8.3 Magentic-One (Microsoft, 2024)

A real-world, production-grade multi-agent system:

```
Orchestrator (outer + inner loop)
    ├── WebSurfer (navigates browsers)
    ├── FileSurfer (reads PDFs, PPTXs, audio)
    ├── Coder (writes and analyzes code)
    └── Computer Terminal (executes code)
```

Key architectural features:
- **Task ledger** (outer loop): facts, guesses, high-level plan — updated when stuck
- **Progress ledger** (inner loop): current assignment and progress tracking
- **Stall detection**: if no progress after N steps, escalate to outer loop for re-planning
- Achieves state-of-the-art on GAIA benchmark without task-specific modifications

### 8.4 Tradeoffs of Multi-Agent Systems

| Pro | Con |
|-----|-----|
| Better task completion quality | Higher cost (more LLM calls) |
| Specialised per-role prompts | More complex debugging |
| Parallelism reduces latency | Emergent behaviour is unpredictable |
| Modular / extensible | Message routing adds latency |

---

## 9. Production Engineering Concerns

### 9.1 Reliability and Error Compounding

Accuracy compounds multiplicatively across steps. Mitigation strategies:
- **Max iteration limits** — hard caps on retries/loops
- **Structured outputs** — JSON schemas prevent malformed responses
- **Checkpointing** — persist state after each step; resume from last good state on failure
- **Sandboxed execution** — isolate code execution and file access

### 9.2 Latency and Cost

| Approach | Latency | Cost |
|----------|---------|------|
| Zero-shot single call | Lowest | Lowest |
| ReAct (1 call/tool) | Medium | Medium |
| Plan-and-Execute | Higher (planning + exec) | Lower (smaller exec model) |
| Multi-agent parallel | Can be low | Higher overall |

Optimisation levers:
- Use smaller/cheaper models for executor and worker roles
- Parallelise independent subtasks (LLMCompiler-style DAG)
- Cache deterministic tool results
- Stream responses to reduce perceived latency

### 9.3 Safety and Security

**Prompt injection** is the primary attack surface: a malicious tool response instructs the agent to take unintended actions.

Mitigations:
- Validate and sanitise all tool outputs before feeding back to the LLM
- Principle of least privilege — only give the agent the tools it needs
- Require human approval for irreversible write actions (delete, send, transfer)
- Sandbox code execution (RestrictedPython, Docker containers, WASM)
- Define explicit stopping conditions and action scopes upfront

### 9.4 Observability

Production agents need full traceability:
- **Trace every LLM call** with inputs, outputs, token counts, latency
- **Log tool calls** with arguments and results
- **Record the full agent trajectory** for debugging and auditing
- Tools: LangSmith, Langfuse, Helicone, OpenTelemetry-compatible exporters

### 9.5 Evaluation

Evaluating agents is harder than evaluating single LLM calls:
- **Trajectory evaluation** — not just the final answer, but the path taken
- **Step-level accuracy** — measure error at each decision point
- **Benchmark suites** — GAIA, SWE-bench, HumanEval, WebArena
- **LLM-as-judge** — use a separate model to assess output quality (the Critic pattern)
- Always test in sandboxed environments before production deployment

---

## 10. Framework Landscape

| Framework | Language | Approach | Best For |
|-----------|----------|----------|----------|
| **LangGraph** | Python, JS | Graph-based state machine; fine-grained control | Production systems needing explicit flow control |
| **AutoGen** (Microsoft) | Python | Conversation-based multi-agent | Research, complex collaboration |
| **CrewAI** | Python | Role-based multi-agent with tasks | Higher-level, quick multi-agent prototyping |
| **Strands Agents** (AWS) | Python | Tool-first, model-agnostic | AWS ecosystem integration |
| **OpenAI Agents SDK** | Python | Lightweight, handoff-based | OpenAI-first, simple agent networks |
| **LlamaIndex** | Python | RAG-heavy, data pipeline agents | Knowledge-intensive tasks |
| **Rivet / Vellum** | GUI | Visual graph builder | Non-technical builders, workflow design |

**Anthropic's recommendation:** Start with raw LLM API calls — many patterns need only a few dozen lines. Introduce frameworks when their abstractions genuinely accelerate you, not before. Always understand what the framework does under the hood.

---

## 11. Key Takeaways & Principles

### The Three Core Principles (Anthropic)
1. **Simplicity** — avoid unnecessary complexity; the most successful agents are often the simplest
2. **Transparency** — explicitly expose planning steps; make the agent's reasoning visible
3. **Agent-Computer Interface (ACI)** — invest heavily in tool documentation and testing, just as you would a human-facing API

### Engineering Heuristics
- **Start simple, escalate deliberately** — single LLM call → prompt chaining → full agentic loop
- **Decouple planning from execution** — validate plans before committing to expensive tool calls
- **Cap iteration loops** — all agentic loops must have a hard maximum to prevent runaway costs
- **Instrument everything** — you cannot improve what you cannot observe
- **Test in sandboxes** — agentic systems with write access can cause irreversible damage in production
- **Design for resumability** — checkpoint state so runs can be resumed after failures

### The Insight on Agent Loops (Andrew Ng)
> GPT-3.5 zero-shot: 48.1% on HumanEval.  
> GPT-3.5 in an agentic loop: **95.1%**.  
> The loop architecture matters more than the model version.

### The Insight on Complexity (Anthropic)
> "Success in the LLM space isn't about building the most sophisticated system. It's about building the *right* system for your needs."

---

## 12. References

| Source | URL |
|--------|-----|
| Anthropic — Building Effective Agents | https://www.anthropic.com/research/building-effective-agents |
| Chip Huyen — Agents (AI Engineering, 2025) | https://huyenchip.com/2025/01/07/agents.html |
| Andrew Ng — Agentic Design Patterns (Parts 1–5) | https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/ |
| LangChain — Planning Agents | https://blog.langchain.dev/planning-agents/ |
| LangChain — Memory for Agents | https://blog.langchain.dev/memory-for-agents/ |
| Microsoft Research — Magentic-One | https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/ |
| Model Context Protocol | https://modelcontextprotocol.io/introduction |
| CoALA Paper — Cognitive Architectures for Language Agents | https://arxiv.org/pdf/2309.02427 |
| ReAct Paper | https://arxiv.org/abs/2210.03629 |
| Plan-and-Solve Prompting | https://arxiv.org/abs/2305.04091 |
| ReWOO | https://arxiv.org/abs/2305.18323 |
| LLMCompiler | https://arxiv.org/abs/2312.04511 |
| AutoGen Paper | https://arxiv.org/abs/2308.08155 |
| MetaGPT | https://arxiv.org/abs/2308.00352 |
