"""Unit tests for individual agent nodes."""
import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


def _make_state(**kwargs):
    defaults = {
        "messages": [HumanMessage(content="Write a haiku about Python.")],
        "plan": [],
        "current_step": 0,
        "critique": "",
        "iteration": 0,
        "final_output": "",
    }
    defaults.update(kwargs)
    return defaults


class TestSupervisorNode:
    def test_routes_to_planner_when_no_output(self):
        mock_response = AIMessage(content=json.dumps({"next": "planner"}))
        with patch("my_agent.agents.supervisor.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.supervisor import supervisor_node
            result = supervisor_node(_make_state())
        assert "messages" in result

    def test_routes_to_end_when_output_present(self):
        state = _make_state(final_output="Snakes in brackets, / Indentation rules the land, / Python breathes free.")
        mock_response = AIMessage(content=json.dumps({"next": "end"}))
        with patch("my_agent.agents.supervisor.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.supervisor import supervisor_node
            result = supervisor_node(state)
        assert "messages" in result


class TestPlannerNode:
    def test_returns_plan_list(self):
        plan_data = {"plan": ["Research haiku format", "Write 5-7-5 syllable poem"]}
        mock_response = AIMessage(content=json.dumps(plan_data))
        with patch("my_agent.agents.planner.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.planner import planner_node
            result = planner_node(_make_state())
        assert result["plan"] == plan_data["plan"]
        assert result["current_step"] == 0

    def test_handles_malformed_response(self):
        mock_response = AIMessage(content="not json")
        with patch("my_agent.agents.planner.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.planner import planner_node
            result = planner_node(_make_state())
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) == 1


class TestExecutorNode:
    def test_increments_current_step(self):
        state = _make_state(plan=["Step 1", "Step 2"], current_step=0)
        mock_response = AIMessage(content="Step 1 done.", tool_calls=[])
        with patch("my_agent.agents.executor.ChatOpenAI") as MockLLM:
            MockLLM.return_value.bind_tools.return_value.invoke.return_value = mock_response
            from my_agent.agents.executor import executor_node
            result = executor_node(state)
        assert result["current_step"] == 1

    def test_no_plan_returns_message(self):
        state = _make_state(plan=[], current_step=0)
        with patch("my_agent.agents.executor.ChatOpenAI"):
            from my_agent.agents.executor import executor_node
            result = executor_node(state)
        assert "messages" in result


class TestCriticNode:
    def test_approved_sets_final_output(self):
        critic_data = {
            "approved": True,
            "critique": "Approved.",
            "final_output": "Great haiku!",
        }
        mock_response = AIMessage(content=json.dumps(critic_data))
        with patch("my_agent.agents.critic.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.critic import critic_node
            result = critic_node(_make_state())
        assert result["final_output"] == "Great haiku!"
        assert result["iteration"] == 1

    def test_rejected_clears_final_output(self):
        critic_data = {
            "approved": False,
            "critique": "The haiku lacks imagery.",
            "final_output": "",
        }
        mock_response = AIMessage(content=json.dumps(critic_data))
        with patch("my_agent.agents.critic.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from my_agent.agents.critic import critic_node
            result = critic_node(_make_state())
        assert result["final_output"] == ""
        assert result["critique"] == "The haiku lacks imagery."
