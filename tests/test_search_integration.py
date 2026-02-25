"""Tests for the DuckDuckGo web search integration."""

from unittest.mock import MagicMock, patch

import pytest

from corvus.integrations.search import (
    SearchError,
    SearchResult,
    _search_sync,
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
