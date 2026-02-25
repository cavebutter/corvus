"""Web search integration via the ddgs library (DuckDuckGo backend).

Wraps the ddgs library with an async interface.
The library is synchronous; we use asyncio.to_thread() to avoid blocking.
"""

import asyncio
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
DEFAULT_REGION = "us-en"
DEFAULT_SAFESEARCH = "moderate"
DEFAULT_BACKEND = "duckduckgo"


class SearchResult(BaseModel):
    """A single web search result."""

    title: str
    url: str
    snippet: str


class SearchError(Exception):
    """Raised when a web search fails."""


def _search_sync(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
    safesearch: str = DEFAULT_SAFESEARCH,
    backend: str = DEFAULT_BACKEND,
    timelimit: str | None = None,
) -> list[SearchResult]:
    """Synchronous web search. Intended to be called via asyncio.to_thread().

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        region: Region for search results.
        safesearch: Safe search level (off, moderate, strict).
        backend: Search backend (duckduckgo, bing, brave, google, etc.).
        timelimit: Time limit for results (d=day, w=week, m=month, y=year).

    Returns:
        List of SearchResult objects.

    Raises:
        SearchError: If the search fails for any reason.
    """
    try:
        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            backend=backend,
            timelimit=timelimit,
        )

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in results
        ]
    except Exception as exc:
        raise SearchError(f"Web search failed: {exc}") from exc


async def web_search(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
    safesearch: str = DEFAULT_SAFESEARCH,
    backend: str = DEFAULT_BACKEND,
    timelimit: str | None = None,
) -> list[SearchResult]:
    """Async web search.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        region: Region for search results.
        safesearch: Safe search level (off, moderate, strict).
        backend: Search backend (duckduckgo, bing, brave, google, etc.).
        timelimit: Time limit for results (d=day, w=week, m=month, y=year).

    Returns:
        List of SearchResult objects.

    Raises:
        SearchError: If the search fails for any reason.
    """
    logger.info("Web search: %s (max_results=%d, backend=%s)", query, max_results, backend)

    return await asyncio.to_thread(
        _search_sync,
        query,
        max_results=max_results,
        region=region,
        safesearch=safesearch,
        backend=backend,
        timelimit=timelimit,
    )
