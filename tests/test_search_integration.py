"""Tests for the DuckDuckGo web search integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from corvus.integrations.search import (
    SearchError,
    SearchResult,
    _fetch_single_page,
    _truncate_on_word_boundary,
    _search_sync,
    fetch_page_content,
    web_search,
)


# ------------------------------------------------------------------
# _search_sync
# ------------------------------------------------------------------


def test_search_sync_calls_ddgs():
    """_search_sync calls DDGS().text() and returns SearchResult objects."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
    ]

    with patch("ddgs.DDGS", return_value=mock_ddgs_instance):
        results = _search_sync("test query", max_results=2)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Result 1"
    assert results[0].url == "https://example.com/1"
    assert results[0].snippet == "Snippet 1"
    assert results[1].title == "Result 2"

    mock_ddgs_instance.text.assert_called_once_with(
        "test query",
        max_results=2,
        region="us-en",
        safesearch="moderate",
        backend="duckduckgo",
        timelimit=None,
    )


def test_search_sync_error_wraps_exception():
    """_search_sync wraps exceptions in SearchError."""
    with patch(
        "ddgs.DDGS",
        side_effect=RuntimeError("Connection failed"),
    ):
        with pytest.raises(SearchError, match="Web search failed"):
            _search_sync("test query")


def test_search_sync_passes_params():
    """_search_sync forwards all kwargs to DDGS().text()."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    with patch("ddgs.DDGS", return_value=mock_ddgs_instance):
        _search_sync(
            "test query",
            max_results=10,
            region="de-de",
            safesearch="off",
            backend="brave",
            timelimit="w",
        )

    mock_ddgs_instance.text.assert_called_once_with(
        "test query",
        max_results=10,
        region="de-de",
        safesearch="off",
        backend="brave",
        timelimit="w",
    )


# ------------------------------------------------------------------
# web_search (async wrapper)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_returns_results():
    """web_search returns SearchResult list via asyncio.to_thread."""
    mock_results = [
        SearchResult(title="Result 1", url="https://example.com/1", snippet="Snippet 1"),
    ]

    with patch(
        "corvus.integrations.search._search_sync",
        return_value=mock_results,
    ) as mock_sync:
        results = await web_search("test query", max_results=3)

    assert len(results) == 1
    assert results[0].title == "Result 1"
    mock_sync.assert_called_once_with(
        "test query",
        max_results=3,
        region="us-en",
        safesearch="moderate",
        backend="duckduckgo",
        timelimit=None,
    )


@pytest.mark.asyncio
async def test_web_search_empty_results():
    """web_search handles empty results."""
    with patch(
        "corvus.integrations.search._search_sync",
        return_value=[],
    ):
        results = await web_search("test query")

    assert results == []


# ------------------------------------------------------------------
# _truncate_on_word_boundary
# ------------------------------------------------------------------


def test_truncate_short_text_unchanged():
    """Text shorter than max_chars is returned unchanged."""
    assert _truncate_on_word_boundary("hello world", 100) == "hello world"


def test_truncate_long_text_on_word_boundary():
    """Long text is truncated at a word boundary."""
    text = "the quick brown fox jumps over the lazy dog"
    result = _truncate_on_word_boundary(text, 20)
    assert result.endswith("...")
    assert len(result) <= 24  # 20 + "..."
    assert "the quick brown fox" in result


# ------------------------------------------------------------------
# _fetch_single_page
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_single_page_extracts_text():
    """_fetch_single_page extracts text via trafilatura."""
    html = "<html><body><article><p>Hello world content here.</p></article></body></html>"
    mock_response = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=html,
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = mock_response

    with patch("trafilatura.extract", return_value="Hello world content here."):
        result = await _fetch_single_page(client, "https://example.com")

    assert result == "Hello world content here."


@pytest.mark.asyncio
async def test_fetch_single_page_truncates_to_max_chars():
    """_fetch_single_page truncates content to max_chars."""
    long_text = "word " * 500  # 2500 chars
    html = f"<html><body><article><p>{long_text}</p></article></body></html>"
    mock_response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text=html,
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = mock_response

    with patch("trafilatura.extract", return_value=long_text.strip()):
        result = await _fetch_single_page(client, "https://example.com", max_chars=50)

    assert result is not None
    assert len(result) <= 54  # 50 + "..."


@pytest.mark.asyncio
async def test_fetch_single_page_returns_none_on_timeout():
    """_fetch_single_page returns None on timeout."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.TimeoutException("timed out")

    result = await _fetch_single_page(client, "https://example.com")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_single_page_returns_none_on_http_error():
    """_fetch_single_page returns None on non-2xx status."""
    mock_response = httpx.Response(
        404,
        headers={"content-type": "text/html"},
        text="Not found",
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = mock_response

    result = await _fetch_single_page(client, "https://example.com")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_single_page_returns_none_on_non_html():
    """_fetch_single_page returns None for non-HTML content-type."""
    mock_response = httpx.Response(
        200,
        headers={"content-type": "application/pdf"},
        text="binary stuff",
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = mock_response

    result = await _fetch_single_page(client, "https://example.com")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_single_page_returns_none_on_extraction_failure():
    """_fetch_single_page returns None when trafilatura returns None."""
    mock_response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text="<html><body></body></html>",
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = mock_response

    with patch("trafilatura.extract", return_value=None):
        result = await _fetch_single_page(client, "https://example.com")

    assert result is None


# ------------------------------------------------------------------
# fetch_page_content
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_page_content_populates_top_n():
    """fetch_page_content populates page_content for top N results only."""
    results = [
        SearchResult(title="R1", url="https://a.com", snippet="S1"),
        SearchResult(title="R2", url="https://b.com", snippet="S2"),
        SearchResult(title="R3", url="https://c.com", snippet="S3"),
    ]

    with patch(
        "corvus.integrations.search._fetch_single_page",
        side_effect=["Content A", "Content B"],
    ):
        returned = await fetch_page_content(results, max_pages=2)

    assert returned is results
    assert results[0].page_content == "Content A"
    assert results[1].page_content == "Content B"
    assert results[2].page_content is None


@pytest.mark.asyncio
async def test_fetch_page_content_handles_partial_failures():
    """fetch_page_content handles partial failures (some pages fail)."""
    results = [
        SearchResult(title="R1", url="https://a.com", snippet="S1"),
        SearchResult(title="R2", url="https://b.com", snippet="S2"),
    ]

    with patch(
        "corvus.integrations.search._fetch_single_page",
        side_effect=["Content A", None],
    ):
        await fetch_page_content(results, max_pages=2)

    assert results[0].page_content == "Content A"
    assert results[1].page_content is None


@pytest.mark.asyncio
async def test_fetch_page_content_handles_all_failures():
    """fetch_page_content handles all fetches failing."""
    results = [
        SearchResult(title="R1", url="https://a.com", snippet="S1"),
        SearchResult(title="R2", url="https://b.com", snippet="S2"),
    ]

    with patch(
        "corvus.integrations.search._fetch_single_page",
        side_effect=[None, None],
    ):
        await fetch_page_content(results, max_pages=2)

    assert results[0].page_content is None
    assert results[1].page_content is None


@pytest.mark.asyncio
async def test_fetch_page_content_empty_list():
    """fetch_page_content handles empty results list."""
    results = await fetch_page_content([], max_pages=2)
    assert results == []
