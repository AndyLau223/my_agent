from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from my_agent.config import settings
from my_agent.graph.state import AgentState
from my_agent.tools import ALL_TOOLS

_SYSTEM = """You are an Executor agent. You will be given one specific sub-task to complete.
Use the available tools to complete it. Be precise and thorough.
After completing the task, summarise what you did and what you found."""


def executor_node(state: AgentState) -> dict:
    """Execute the current plan step using available tools."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).bind_tools(ALL_TOOLS)

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if not plan or current_step >= len(plan):
        return {"messages": [AIMessage(content="No more steps to execute.")]}

    step = plan[current_step]

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            *state.get("messages", []),
            HumanMessage(content=f"Execute this step: {step}"),
        ]
    )

    return {
        "messages": [response],
        "current_step": current_step + 1,
    }
