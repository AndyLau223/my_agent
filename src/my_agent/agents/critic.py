import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from my_agent.config import settings
from my_agent.graph.state import AgentState

_SYSTEM = """You are a Critic agent. Your job is to rigorously evaluate whether the Executor has
successfully completed the original task to a high standard.

Respond ONLY with a JSON object:
{
  "approved": true | false,
  "critique": "Detailed feedback if not approved, or 'Approved.' if approved.",
  "final_output": "The polished final answer / output to show the user (only when approved)."
}

Be strict — only approve if the task is fully and correctly completed."""


def critic_node(state: AgentState) -> dict:
    """Review the executor's work and either approve or request revision."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    task = _extract_task(state)
    iteration = state.get("iteration", 0)
    max_iterations = settings.max_critic_iterations

    # Summarise recent execution for the critic
    recent_messages = state.get("messages", [])[-6:]
    execution_summary = "\n".join(
        f"{type(m).__name__}: {m.content[:500]}" for m in recent_messages
    )

    prompt = (
        f"Original task: {task}\n\n"
        f"Execution output (last {len(recent_messages)} messages):\n{execution_summary}\n\n"
        f"Iteration: {iteration + 1} / {max_iterations}"
    )

    if iteration + 1 >= max_iterations:
        prompt += "\n\nThis is the final allowed iteration — you MUST approve the best available output."

    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )

    try:
        data = json.loads(response.content)
        approved = bool(data.get("approved", False))
        critique = data.get("critique", "")
        final_output = data.get("final_output", "") if approved else ""
    except (json.JSONDecodeError, AttributeError):
        approved = True
        critique = "Approved (parse fallback)."
        final_output = response.content

    new_iteration = iteration + 1

    return {
        "messages": [response],
        "critique": critique,
        "iteration": new_iteration,
        "final_output": final_output if approved else "",
    }


def _extract_task(state: AgentState) -> str:
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            return msg.content
    return "Unknown task"
