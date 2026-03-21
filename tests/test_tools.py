"""Unit tests for individual tools."""
import pytest
from unittest.mock import patch, MagicMock


class TestWebSearch:
    def test_web_search_returns_string(self):
        mock_tavily = MagicMock()
        mock_tavily.invoke.return_value = [
            {"title": "Result 1", "content": "Some content about topic."},
            {"title": "Result 2", "content": "More info here."},
        ]
        with patch("my_agent.tools.web_search._get_tavily", return_value=mock_tavily):
            from my_agent.tools.web_search import web_search
            result = web_search.invoke({"query": "test query"})
            assert isinstance(result, str)
            assert "Result 1" in result
            assert "Some content about topic." in result

    def test_web_search_handles_string_response(self):
        mock_tavily = MagicMock()
        mock_tavily.invoke.return_value = "plain string response"
        with patch("my_agent.tools.web_search._get_tavily", return_value=mock_tavily):
            from my_agent.tools.web_search import web_search
            result = web_search.invoke({"query": "test"})
            assert result == "plain string response"


class TestCodeExecutor:
    def test_simple_print(self):
        from my_agent.tools.code_executor import execute_python
        result = execute_python.invoke({"code": "print(1 + 1)"})
        assert "2" in result

    def test_syntax_error(self):
        from my_agent.tools.code_executor import execute_python
        result = execute_python.invoke({"code": "def bad syntax"})
        assert "SyntaxError" in result

    def test_runtime_error(self):
        from my_agent.tools.code_executor import execute_python
        result = execute_python.invoke({"code": "1 / 0"})
        assert "ZeroDivisionError" in result or "Error" in result

    def test_no_output(self):
        from my_agent.tools.code_executor import execute_python
        result = execute_python.invoke({"code": "x = 42"})
        assert result == "(no output)"


class TestFileSystem:
    def test_write_and_read_file(self, tmp_path, monkeypatch):
        import my_agent.tools.file_system as fs_module
        monkeypatch.setattr(fs_module, "_WORKSPACE", tmp_path)
        from my_agent.tools.file_system import write_file, read_file

        write_result = write_file.invoke({"file_path": "test.txt", "content": "hello world"})
        assert "Successfully wrote" in write_result

        read_result = read_file.invoke({"file_path": "test.txt"})
        assert read_result == "hello world"

    def test_read_missing_file(self, tmp_path, monkeypatch):
        import my_agent.tools.file_system as fs_module
        monkeypatch.setattr(fs_module, "_WORKSPACE", tmp_path)
        from my_agent.tools.file_system import read_file

        result = read_file.invoke({"file_path": "nonexistent.txt"})
        assert "not found" in result

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        import my_agent.tools.file_system as fs_module
        monkeypatch.setattr(fs_module, "_WORKSPACE", tmp_path)
        from my_agent.tools.file_system import read_file

        result = read_file.invoke({"file_path": "../../etc/passwd"})
        assert "Error" in result


class TestHttpClient:
    def test_http_get_success(self):
        with patch("my_agent.tools.http_client.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.text = "response body"
            mock_response.raise_for_status.return_value = None
            mock_httpx.get.return_value = mock_response
            mock_httpx.HTTPError = Exception

            from my_agent.tools.http_client import http_get
            result = http_get.invoke({"url": "https://example.com"})
            assert result == "response body"

    def test_http_post_success(self):
        with patch("my_agent.tools.http_client.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.text = '{"ok": true}'
            mock_response.raise_for_status.return_value = None
            mock_httpx.post.return_value = mock_response
            mock_httpx.HTTPError = Exception

            from my_agent.tools.http_client import http_post
            result = http_post.invoke({"url": "https://api.example.com", "body": {"key": "val"}})
            assert '{"ok": true}' in result
