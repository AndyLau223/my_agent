import io
import contextlib
from langchain_core.tools import tool
from RestrictedPython import compile_restricted, safe_globals, safe_builtins, PrintCollector


@tool
def execute_python(code: str) -> str:
    """Execute a snippet of Python code in a restricted sandbox and return stdout/stderr.

    Only safe built-ins are available. No file I/O, network, or os module access.

    Args:
        code: Valid Python source code to execute.

    Returns:
        Combined stdout and stderr output as a string.
    """
    try:
        byte_code = compile_restricted(code, "<string>", "exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    glb = {
        **safe_globals,
        "__builtins__": safe_builtins,
        "_print_": PrintCollector,
        "_write_": lambda x: x,
        "_getiter_": iter,
        "_getattr_": getattr,
    }
    try:
        exec(byte_code, glb)  # noqa: S102
        output = glb.get("_print", None)
        if output is not None:
            return str(output.txt) if output.txt else "(no output)"
        return "(no output)"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
