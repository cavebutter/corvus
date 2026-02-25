"""Tests for the orchestrator router."""

from unittest.mock import AsyncMock, patch

import pytest

from corvus.orchestrator.router import CONFIDENCE_THRESHOLD, dispatch
from corvus.schemas.orchestrator import (
    DigestResult,
    FetchPipelineResult,
    Intent,
    IntentClassification,
    OrchestratorAction,
    StatusResult,
    TagPipelineResult,
    WebSearchResult,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_classification(intent: Intent, confidence: float = 0.9, **kwargs):
    return IntentClassification(
        intent=intent,
        confidence=confidence,
        reasoning="Test reasoning",
        **kwargs,
    )


def _make_dispatch_kwargs(tmp_path):
    return dict(
        paperless=AsyncMock(),
        ollama=AsyncMock(),
        model="test-model",
        keep_alive="5m",
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        on_progress=None,
    )


# ------------------------------------------------------------------
# Confidence gate
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_returns_clarification(tmp_path):
    """Low confidence triggers NEEDS_CLARIFICATION."""
    classification = _make_classification(Intent.FETCH_DOCUMENT, confidence=0.5)

    response = await dispatch(
        classification,
        user_input="something vague",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.NEEDS_CLARIFICATION
    assert response.confidence == 0.5
    assert response.clarification_prompt is not None


@pytest.mark.asyncio
async def test_threshold_boundary(tmp_path):
    """Confidence exactly at threshold passes through."""
    classification = _make_classification(
        Intent.SHOW_STATUS, confidence=CONFIDENCE_THRESHOLD,
    )

    response = await dispatch(
        classification,
        user_input="status",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.DISPATCHED


# ------------------------------------------------------------------
# Interactive-only intents
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_queue_interactive_required(tmp_path):
    """REVIEW_QUEUE returns INTERACTIVE_REQUIRED."""
    classification = _make_classification(Intent.REVIEW_QUEUE)

    response = await dispatch(
        classification,
        user_input="review pending items",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.INTERACTIVE_REQUIRED
    assert "corvus review" in response.message


@pytest.mark.asyncio
async def test_watch_folder_interactive_required(tmp_path):
    """WATCH_FOLDER returns INTERACTIVE_REQUIRED."""
    classification = _make_classification(Intent.WATCH_FOLDER)

    response = await dispatch(
        classification,
        user_input="watch my scan folder",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.INTERACTIVE_REQUIRED
    assert "corvus watch" in response.message


# ------------------------------------------------------------------
# TAG_DOCUMENTS
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_tag(tmp_path):
    """TAG_DOCUMENTS dispatches to tag pipeline."""
    classification = _make_classification(
        Intent.TAG_DOCUMENTS, tag_limit=5,
    )
    mock_result = TagPipelineResult(processed=5, queued=5, auto_applied=0, errors=0)

    with patch(
        "corvus.orchestrator.pipelines.run_tag_pipeline",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_pipeline:
        response = await dispatch(
            classification,
            user_input="tag 5 documents",
            **_make_dispatch_kwargs(tmp_path),
        )

    assert response.action == OrchestratorAction.DISPATCHED
    assert response.intent == Intent.TAG_DOCUMENTS
    assert isinstance(response.result, TagPipelineResult)
    assert response.result.processed == 5
    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args.kwargs["limit"] == 5


# ------------------------------------------------------------------
# FETCH_DOCUMENT
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_fetch(tmp_path):
    """FETCH_DOCUMENT dispatches to fetch pipeline."""
    classification = _make_classification(
        Intent.FETCH_DOCUMENT, fetch_query="AT&T invoice",
    )
    mock_result = FetchPipelineResult(
        documents_found=1,
        documents=[{"id": 73, "title": "AT&T Invoice", "created": "2025-01-01"}],
    )

    with patch(
        "corvus.orchestrator.pipelines.run_fetch_pipeline",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_pipeline:
        response = await dispatch(
            classification,
            user_input="find my AT&T invoice",
            **_make_dispatch_kwargs(tmp_path),
        )

    assert response.action == OrchestratorAction.DISPATCHED
    assert response.intent == Intent.FETCH_DOCUMENT
    assert isinstance(response.result, FetchPipelineResult)
    assert response.result.documents_found == 1
    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args.kwargs["query"] == "AT&T invoice"


@pytest.mark.asyncio
async def test_dispatch_fetch_no_query(tmp_path):
    """FETCH_DOCUMENT without fetch_query returns NEEDS_CLARIFICATION."""
    classification = _make_classification(
        Intent.FETCH_DOCUMENT, fetch_query=None,
    )

    response = await dispatch(
        classification,
        user_input="find something",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.NEEDS_CLARIFICATION
    assert "document" in response.message.lower()


# ------------------------------------------------------------------
# SHOW_STATUS
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_status(tmp_path):
    """SHOW_STATUS dispatches to status pipeline."""
    classification = _make_classification(Intent.SHOW_STATUS)

    response = await dispatch(
        classification,
        user_input="status",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.DISPATCHED
    assert response.intent == Intent.SHOW_STATUS
    assert isinstance(response.result, StatusResult)


# ------------------------------------------------------------------
# SHOW_DIGEST
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_digest(tmp_path):
    """SHOW_DIGEST dispatches to digest pipeline."""
    classification = _make_classification(Intent.SHOW_DIGEST, digest_hours=48)

    response = await dispatch(
        classification,
        user_input="what happened in the last 2 days",
        **_make_dispatch_kwargs(tmp_path),
    )

    assert response.action == OrchestratorAction.DISPATCHED
    assert response.intent == Intent.SHOW_DIGEST
    assert isinstance(response.result, DigestResult)
    assert "Corvus Daily Digest" in response.result.rendered_text


@pytest.mark.asyncio
async def test_dispatch_digest_default_hours(tmp_path):
    """SHOW_DIGEST defaults to 24 hours when not specified."""
    classification = _make_classification(Intent.SHOW_DIGEST)

    with patch(
        "corvus.orchestrator.pipelines.run_digest_pipeline",
    ) as mock_pipeline:
        mock_pipeline.return_value = DigestResult(rendered_text="test")
        response = await dispatch(
            classification,
            user_input="show digest",
            **_make_dispatch_kwargs(tmp_path),
        )

    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args.kwargs["hours"] == 24


# ------------------------------------------------------------------
# GENERAL_CHAT
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_general_chat(tmp_path):
    """GENERAL_CHAT calls ollama.chat() and returns response."""
    classification = _make_classification(Intent.GENERAL_CHAT)
    kwargs = _make_dispatch_kwargs(tmp_path)
    kwargs["ollama"].chat.return_value = ("Hello! I'm Corvus.", AsyncMock())

    response = await dispatch(
        classification,
        user_input="hello",
        **kwargs,
    )

    assert response.action == OrchestratorAction.CHAT_RESPONSE
    assert response.intent == Intent.GENERAL_CHAT
    assert "Corvus" in response.message
    kwargs["ollama"].chat.assert_called_once()


# ------------------------------------------------------------------
# WEB_SEARCH
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_web_search(tmp_path):
    """WEB_SEARCH dispatches to search pipeline with search_query."""
    classification = _make_classification(
        Intent.WEB_SEARCH, search_query="weather in NYC",
    )
    mock_result = WebSearchResult(
        summary="It's sunny in NYC.",
        sources=[],
        query="weather in NYC",
    )

    with patch(
        "corvus.orchestrator.pipelines.run_search_pipeline",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_pipeline:
        response = await dispatch(
            classification,
            user_input="what's the weather in NYC",
            **_make_dispatch_kwargs(tmp_path),
        )

    assert response.action == OrchestratorAction.DISPATCHED
    assert response.intent == Intent.WEB_SEARCH
    assert isinstance(response.result, WebSearchResult)
    assert response.result.summary == "It's sunny in NYC."
    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args.kwargs["query"] == "weather in NYC"


@pytest.mark.asyncio
async def test_dispatch_web_search_no_query_uses_input(tmp_path):
    """WEB_SEARCH falls back to user_input when search_query is None."""
    classification = _make_classification(
        Intent.WEB_SEARCH, search_query=None,
    )
    mock_result = WebSearchResult(
        summary="Here are the results.",
        sources=[],
        query="latest news",
    )

    with patch(
        "corvus.orchestrator.pipelines.run_search_pipeline",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_pipeline:
        response = await dispatch(
            classification,
            user_input="latest news",
            **_make_dispatch_kwargs(tmp_path),
        )

    assert response.action == OrchestratorAction.DISPATCHED
    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args.kwargs["query"] == "latest news"
