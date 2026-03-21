from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from my_agent.config import settings
from my_agent.graph.state import AgentState

_SYSTEM = """You are a Supervisor agent in a multi-agent system. Your job is to:
1. Understand the user's task.
2. Decide whether the task has been fully completed and an approved output is ready.
3. Route control to the appropriate next step.

Respond with a JSON object with a single key "next" whose value is one of:
- "planner"  — task needs to be planned or re-planned
- "end"      — final_output is satisfactory and we are done

Only reply with the JSON object, no additional text."""


def supervisor_node(state: AgentState) -> dict:
    """Supervisor decides whether to route to the planner or end the run."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    context = f"""Task: {_extract_task(state)}

Current iteration: {state.get('iteration', 0)}
Has plan: {bool(state.get('plan'))}
Final output available: {bool(state.get('final_output'))}
Last critique: {state.get('critique', 'none')}
"""

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=context),
        ]
    )

    import json
    try:
        decision = json.loads(response.content)
        next_node = decision.get("next", "planner")
    except (json.JSONDecodeError, AttributeError):
        next_node = "planner"

    return {"messages": [response], "next": next_node}


def _extract_task(state: AgentState) -> str:
    """Pull the original user task from the message history."""
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        if isinstance(msg, HumanMessage):
            return msg.content
    return "Unknown task"
