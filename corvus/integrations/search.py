"""Web search integration via the ddgs library (DuckDuckGo backend).

Wraps the ddgs library with an async interface.
The library is synchronous; we use asyncio.to_thread() to avoid blocking.
"""

import asyncio
import logging

import httpx
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
    page_content: str | None = None


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


_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _truncate_on_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text to max_chars on a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


async def _fetch_single_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_chars: int = 8000,
) -> str | None:
    """Fetch a single page and extract readable text via trafilatura.

    Returns None on ANY failure — never raises.
    """
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code < 200 or response.status_code >= 300:
            logger.debug("Non-2xx status %d for %s", response.status_code, url)
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            logger.debug("Non-HTML content-type '%s' for %s", content_type, url)
            return None

        html = response.text

        import trafilatura

        extracted = await asyncio.to_thread(trafilatura.extract, html)
        if not extracted:
            logger.debug("trafilatura extraction returned None for %s", url)
            return None

        return _truncate_on_word_boundary(extracted, max_chars)

    except Exception:
        logger.debug("Failed to fetch page content from %s", url, exc_info=True)
        return None


async def fetch_page_content(
    results: list[SearchResult],
    *,
    max_pages: int = 2,
    max_chars_per_page: int = 8000,
    timeout: int = 10,
) -> list[SearchResult]:
    """Fetch page content for the top N search results.

    Populates the page_content field on each result where extraction succeeds.
    Returns the same list (mutated in place).
    """
    if not results or max_pages <= 0:
        return results

    to_fetch = results[:max_pages]

    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        tasks = [
            _fetch_single_page(client, r.url, max_chars=max_chars_per_page)
            for r in to_fetch
        ]
        contents = await asyncio.gather(*tasks)

    for result, content in zip(to_fetch, contents):
        result.page_content = content

    return results
