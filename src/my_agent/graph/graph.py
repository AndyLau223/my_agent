from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from my_agent.graph.state import AgentState
from my_agent.graph.checkpoint import get_checkpointer
from my_agent.agents.supervisor import supervisor_node
from my_agent.agents.planner import planner_node
from my_agent.agents.executor import executor_node
from my_agent.agents.critic import critic_node
from my_agent.tools import ALL_TOOLS
from my_agent.config import settings


def _supervisor_router(state: AgentState) -> str:
    """Route from supervisor based on its 'next' field."""
    messages = state.get("messages", [])
    # The last message from supervisor contains routing info stored in state
    # We use a simple heuristic: if final_output is set and approved, end.
    if state.get("final_output"):
        return END
    return "planner"


def _executor_router(state: AgentState) -> str:
    """After executor runs, check if there are pending tool calls."""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
    return "critic"


def _critic_router(state: AgentState) -> str:
    """After critic review, decide: approved → supervisor, rejected → planner (if iterations left)."""
    if state.get("final_output"):
        return "supervisor"
    iteration = state.get("iteration", 0)
    if iteration >= settings.max_critic_iterations:
        return "supervisor"
    return "planner"


def _executor_step_router(state: AgentState) -> str:
    """Continue executing plan steps or move to critic when done."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    # Check last message for tool calls first
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
    if current_step < len(plan):
        return "executor"
    return "critic"


def build_graph():
    """Construct and compile the agent state graph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("critic", critic_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor → planner or END
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {"planner": "planner", END: END},
    )

    # Planner always goes to executor
    builder.add_edge("planner", "executor")

    # Executor → tools (if tool calls pending) or critic (when all steps done)
    builder.add_conditional_edges(
        "executor",
        _executor_step_router,
        {"tools": "tools", "executor": "executor", "critic": "critic"},
    )

    # After tools run, return to executor to continue
    builder.add_edge("tools", "executor")

    # Critic → planner (revision) or supervisor (done / max iterations)
    builder.add_conditional_edges(
        "critic",
        _critic_router,
        {"planner": "planner", "supervisor": "supervisor"},
    )

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph (lazy singleton pattern)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
