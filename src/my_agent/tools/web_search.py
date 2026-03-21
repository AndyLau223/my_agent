from functools import lru_cache
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from my_agent.config import settings


@lru_cache(maxsize=1)
def _get_tavily() -> TavilySearch:
    return TavilySearch(max_results=5, tavily_api_key=settings.tavily_api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information using Tavily.

    Args:
        query: The search query string.

    Returns:
        A string summary of the top search results.
    """
    results = _get_tavily().invoke(query)
    if isinstance(results, list):
        return "\n\n".join(
            f"[{r.get('title', 'Result')}]\n{r.get('content', '')}" for r in results
        )
    return str(results)
