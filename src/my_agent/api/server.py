import uuid
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from my_agent.graph.graph import get_graph

app = FastAPI(
    title="My Agent API",
    description="REST interface for the hierarchical multi-agent system.",
    version="0.1.0",
)


class RunRequest(BaseModel):
    task: str
    thread_id: str | None = None


class RunResponse(BaseModel):
    thread_id: str
    final_output: str
    iterations: int


class StatusResponse(BaseModel):
    thread_id: str
    current_step: int
    plan: list[str]
    iteration: int
    final_output: str
    critique: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]


@app.post("/run", response_model=RunResponse)
def run_task(request: RunRequest) -> RunResponse:
    """Submit a task and run it to completion. Returns the final output."""
    thread_id = request.thread_id or str(uuid.uuid4())
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=request.task)],
        "plan": [],
        "current_step": 0,
        "critique": "",
        "iteration": 0,
        "final_output": "",
    }

    final_output = ""
    iterations = 0

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            final_output = event.get("final_output", final_output)
            iterations = event.get("iteration", iterations)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        thread_id=thread_id,
        final_output=final_output,
        iterations=iterations,
    )


@app.get("/status/{thread_id}", response_model=StatusResponse)
def get_status(thread_id: str) -> StatusResponse:
    """Get the current state of a run by thread ID."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        v = state.values
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Thread not found: {exc}") from exc

    return StatusResponse(
        thread_id=thread_id,
        current_step=v.get("current_step", 0),
        plan=v.get("plan", []),
        iteration=v.get("iteration", 0),
        final_output=v.get("final_output", ""),
        critique=v.get("critique", ""),
    )


@app.get("/history/{thread_id}", response_model=HistoryResponse)
def get_history(thread_id: str) -> HistoryResponse:
    """Get the full message history for a thread."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Thread not found: {exc}") from exc

    serialized = [
        {
            "type": type(m).__name__,
            "content": m.content[:2000] if m.content else "",
        }
        for m in messages
    ]

    return HistoryResponse(thread_id=thread_id, messages=serialized)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
