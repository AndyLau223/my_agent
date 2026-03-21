from langgraph.checkpoint.sqlite import SqliteSaver
from my_agent.config import settings


def get_checkpointer() -> SqliteSaver:
    """Return a SQLite-backed checkpointer for LangGraph."""
    return SqliteSaver.from_conn_string(settings.sqlite_db_path)
