import os
from pathlib import Path
from langchain_core.tools import tool

_WORKSPACE = Path("./workspace")


def _safe_path(file_path: str) -> Path:
    """Resolve path and ensure it stays within the workspace directory."""
    _WORKSPACE.mkdir(parents=True, exist_ok=True)
    resolved = (_WORKSPACE / file_path).resolve()
    if not str(resolved).startswith(str(_WORKSPACE.resolve())):
        raise PermissionError(f"Access denied: path '{file_path}' is outside the workspace.")
    return resolved


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file in the workspace directory.

    Args:
        file_path: Relative path to the file inside the workspace/ directory.

    Returns:
        File contents as a string, or an error message.
    """
    try:
        path = _safe_path(file_path)
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file '{file_path}' not found."
    except PermissionError as exc:
        return f"Error: {exc}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the workspace directory (creates or overwrites).

    Args:
        file_path: Relative path to the file inside the workspace/ directory.
        content: Text content to write.

    Returns:
        Confirmation message or an error message.
    """
    try:
        path = _safe_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    except PermissionError as exc:
        return f"Error: {exc}"
