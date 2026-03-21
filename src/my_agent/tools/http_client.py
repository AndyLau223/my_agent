import json as _json
import httpx
from langchain_core.tools import tool


@tool
def http_get(url: str, params: dict | None = None) -> str:
    """Perform an HTTP GET request and return the response body.

    Args:
        url: The URL to request.
        params: Optional query parameters as a dict.

    Returns:
        Response body text (truncated to 4000 chars) or an error message.
    """
    try:
        response = httpx.get(url, params=params or {}, timeout=15, follow_redirects=True)
        response.raise_for_status()
        return response.text[:4000]
    except httpx.HTTPError as exc:
        return f"HTTP error: {exc}"


@tool
def http_post(url: str, body: dict | None = None, headers: dict | None = None) -> str:
    """Perform an HTTP POST request with a JSON body and return the response.

    Args:
        url: The URL to post to.
        body: Optional JSON-serialisable dict to send as the request body.
        headers: Optional dict of additional HTTP headers.

    Returns:
        Response body text (truncated to 4000 chars) or an error message.
    """
    try:
        response = httpx.post(
            url,
            json=body or {},
            headers=headers or {},
            timeout=15,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text[:4000]
    except httpx.HTTPError as exc:
        return f"HTTP error: {exc}"
