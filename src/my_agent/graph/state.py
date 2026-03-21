from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Full conversation / agent message history (append-only via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]
    # Ordered list of sub-steps produced by the Planner
    plan: list[str]
    # Index into `plan` that the Executor is currently working on
    current_step: int
    # Latest critique from the Critic agent
    critique: str
    # How many times the Critic has sent work back to the Planner
    iteration: int
    # Final approved output text
    final_output: str
