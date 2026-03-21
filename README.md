# My Agent — Modern Hierarchical Multi-Agent System

A production-ready agentic system built with **LangGraph** and **OpenAI GPT-4o**, featuring a four-agent hierarchical pipeline with full tool access and persistent state.

## Architecture

```
User (CLI / REST API)
        │
        ▼
  Supervisor        ← orchestrates, routes, detects completion
        │
        ▼
   Planner          ← decomposes task into ordered sub-steps
        │
        ▼
  Executor          ← runs steps using tools (web search, code, files, HTTP)
        │
        ▼
   Critic           ← reviews quality; loops back to Planner (max 3×) or approves
```

## Tools

| Tool        | Description                         |
| ----------- | ----------------------------------- |
| Web Search  | Tavily-powered real-time web search |
| Python REPL | Sandboxed Python code execution     |
| File System | Read/write files in the workspace   |
| HTTP Client | Generic GET/POST to external APIs   |

## Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 3. Run CLI
my-agent run "Research the latest developments in quantum computing"

# 4. Run API server
uvicorn my_agent.api.server:app --reload
```

## CLI Usage

```bash
# Run a task
my-agent run "Your task here"

# Resume a previous run
my-agent run "Your task here" --thread-id <thread-id>

# Show run history
my-agent history <thread-id>
```

## REST API

| Endpoint               | Method | Description              |
| ---------------------- | ------ | ------------------------ |
| `/run`                 | POST   | Submit a new task        |
| `/status/{thread_id}`  | GET    | Get run status           |
| `/history/{thread_id}` | GET    | Get full message history |

## Development

```bash
# Run tests
pytest

# Run with verbose output
pytest -v
```
