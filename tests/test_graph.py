"""Integration tests for graph routing logic (no LLM calls)."""
import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.testclient import TestClient


class TestGraphRouters:
    def test_supervisor_router_goes_to_end_with_final_output(self):
        from my_agent.graph.graph import _supervisor_router
        state = {
            "messages": [],
            "plan": [],
            "current_step": 0,
            "critique": "",
            "iteration": 1,
            "final_output": "Done!",
        }
        from langgraph.graph import END
        assert _supervisor_router(state) == END

    def test_supervisor_router_goes_to_planner_without_final_output(self):
        from my_agent.graph.graph import _supervisor_router
        state = {
            "messages": [],
            "plan": [],
            "current_step": 0,
            "critique": "",
            "iteration": 0,
            "final_output": "",
        }
        assert _supervisor_router(state) == "planner"

    def test_critic_router_approved_goes_to_supervisor(self):
        from my_agent.graph.graph import _critic_router
        state = {
            "final_output": "Approved output",
            "iteration": 1,
        }
        assert _critic_router(state) == "supervisor"

    def test_critic_router_rejected_goes_to_planner(self):
        from my_agent.graph.graph import _critic_router
        state = {
            "final_output": "",
            "iteration": 1,
        }
        assert _critic_router(state) == "planner"

    def test_critic_router_max_iterations_goes_to_supervisor(self):
        from my_agent.graph.graph import _critic_router
        from my_agent.config import settings
        state = {
            "final_output": "",
            "iteration": settings.max_critic_iterations,
        }
        assert _critic_router(state) == "supervisor"

    def test_executor_router_detects_tool_calls(self):
        from my_agent.graph.graph import _executor_step_router
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "1"}],
        )
        state = {
            "messages": [tool_call_msg],
            "plan": ["step 1", "step 2"],
            "current_step": 1,
        }
        assert _executor_step_router(state) == "tools"

    def test_executor_router_continues_plan(self):
        from my_agent.graph.graph import _executor_step_router
        msg = AIMessage(content="done", tool_calls=[])
        state = {
            "messages": [msg],
            "plan": ["step 1", "step 2"],
            "current_step": 1,
        }
        assert _executor_step_router(state) == "executor"

    def test_executor_router_goes_to_critic_when_done(self):
        from my_agent.graph.graph import _executor_step_router
        msg = AIMessage(content="done", tool_calls=[])
        state = {
            "messages": [msg],
            "plan": ["step 1"],
            "current_step": 1,
        }
        assert _executor_step_router(state) == "critic"


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from my_agent.api.server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_run_endpoint_calls_graph(self, client):
        mock_events = [
            {"messages": [], "plan": ["step 1"], "current_step": 0, "iteration": 0,
             "final_output": "", "critique": ""},
            {"messages": [], "plan": ["step 1"], "current_step": 1, "iteration": 1,
             "final_output": "Task complete!", "critique": "Approved."},
        ]
        with patch("my_agent.api.server.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.stream.return_value = iter(mock_events)
            mock_get_graph.return_value = mock_graph

            response = client.post("/run", json={"task": "Write a haiku"})
            assert response.status_code == 200
            data = response.json()
            assert data["final_output"] == "Task complete!"
            assert "thread_id" in data

    def test_status_endpoint(self, client):
        mock_state = MagicMock()
        mock_state.values = {
            "current_step": 2,
            "plan": ["step 1", "step 2"],
            "iteration": 1,
            "final_output": "Done",
            "critique": "Approved.",
        }
        with patch("my_agent.api.server.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.get_state.return_value = mock_state
            mock_get_graph.return_value = mock_graph

            response = client.get("/status/some-thread-id")
            assert response.status_code == 200
            data = response.json()
            assert data["current_step"] == 2
            assert data["final_output"] == "Done"
