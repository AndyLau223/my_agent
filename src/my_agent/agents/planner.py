import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from my_agent.config import settings
from my_agent.graph.state import AgentState

_SYSTEM = """You are a Planner agent. Given a task and optional critique from a previous attempt,
produce an ordered list of concrete, actionable sub-steps that an Executor agent can carry out
one by one using available tools (web search, Python execution, file I/O, HTTP calls).

Respond ONLY with a JSON object:
{
  "plan": ["step 1 description", "step 2 description", ...]
}

Keep each step specific and achievable in a single tool call or a short sequence."""


def planner_node(state: AgentState) -> dict:
    """Break the task into an ordered list of sub-steps."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    task = _extract_task(state)
    critique = state.get("critique", "")

    prompt = f"Task: {task}"
    if critique:
        prompt += f"\n\nPrevious attempt critique (use this to improve the plan):\n{critique}"

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )

    try:
        data = json.loads(response.content)
        plan = data.get("plan", [])
    except (json.JSONDecodeError, AttributeError):
        plan = [response.content]

    return {
        "messages": [response],
        "plan": plan,
        "current_step": 0,
    }


def _extract_task(state: AgentState) -> str:
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            return msg.content
    return "Unknown task"
