"""Tests for the Corvus orchestrator pipeline handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    TagSuggestion,
)
from corvus.schemas.document_retrieval import QueryInterpretation, ResolvedSearchParams
from corvus.schemas.orchestrator import (
    DigestResult,
    FetchPipelineResult,
    StatusResult,
    TagPipelineResult,
)
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocument,
    PaperlessDocumentType,
    PaperlessTag,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SAMPLE_TAGS = [PaperlessTag(id=1, name="invoice", slug="invoice")]
SAMPLE_CORRESPONDENTS = [PaperlessCorrespondent(id=3, name="Acme Corp", slug="acme-corp")]
SAMPLE_DOC_TYPES = [PaperlessDocumentType(id=4, name="Invoice", slug="invoice")]


def _make_document(doc_id=1, title="Test Invoice"):
    return PaperlessDocument(
        id=doc_id,
        title=title,
        content="Invoice from Acme Corp for $100.00 due 2025-01-15",
        tags=[],
        correspondent=None,
        document_type=None,
        created="2025-01-01T00:00:00Z",
        added="2025-01-01T00:00:00Z",
        original_filename="invoice.pdf",
    )


def _make_task(doc_id=1, title="Test Invoice"):
    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=title,
        content_snippet="Invoice from Acme Corp...",
        result=DocumentTaggingResult(
            suggested_tags=[
                TagSuggestion(tag_name="invoice", confidence=0.95),
            ],
            suggested_correspondent="Acme Corp",
            suggested_document_type="Invoice",
            correspondent_confidence=0.9,
            document_type_confidence=0.88,
            reasoning="Document appears to be an invoice.",
        ),
        overall_confidence=0.78,
        gate_action=GateAction.FLAG_IN_DIGEST,
    )


def _make_routing_result(task=None, applied=False, action=GateAction.QUEUE_FOR_REVIEW):
    from corvus.router.tagging import RoutingResult

    task = task or _make_task()
    return RoutingResult(
        task=task,
        proposed_update=ProposedDocumentUpdate(
            document_id=task.document_id,
            add_tag_ids=[1],
        ),
        applied=applied,
        effective_action=action,
    )


def _mock_paperless(docs=None, count=0):
    mock = AsyncMock()
    mock.list_documents.return_value = (docs or [], count)
    mock.list_tags.return_value = SAMPLE_TAGS
    mock.list_correspondents.return_value = SAMPLE_CORRESPONDENTS
    mock.list_document_types.return_value = SAMPLE_DOC_TYPES
    return mock


def _mock_ollama():
    return AsyncMock()


# ------------------------------------------------------------------
# run_tag_pipeline
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_pipeline_no_documents(tmp_path):
    """Tag pipeline returns zeros when no documents to process."""
    from corvus.orchestrator.pipelines import run_tag_pipeline

    progress = []

    result = await run_tag_pipeline(
        paperless=_mock_paperless(),
        ollama=_mock_ollama(),
        model="test-model",
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        on_progress=progress.append,
    )

    assert isinstance(result, TagPipelineResult)
    assert result.processed == 0
    assert result.errors == 0
    assert any("No documents to process" in msg for msg in progress)


@pytest.mark.asyncio
async def test_tag_pipeline_processes_and_queues(tmp_path):
    """Tag pipeline processes documents and tracks counts."""
    from corvus.orchestrator.pipelines import run_tag_pipeline

    doc = _make_document()
    task = _make_task()
    routing = _make_routing_result(task, applied=False, action=GateAction.QUEUE_FOR_REVIEW)
    mock_raw = MagicMock()

    progress = []

    with (
        patch(
            "corvus.executors.document_tagger.tag_document",
            new_callable=AsyncMock,
            return_value=(task, mock_raw),
        ),
        patch(
            "corvus.router.tagging.resolve_and_route",
            new_callable=AsyncMock,
            return_value=routing,
        ),
    ):
        result = await run_tag_pipeline(
            paperless=_mock_paperless(docs=[doc], count=1),
            ollama=_mock_ollama(),
            model="test-model",
            limit=1,
            queue_db_path=str(tmp_path / "queue.db"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            on_progress=progress.append,
        )

    assert result.processed == 1
    assert result.queued == 1
    assert result.auto_applied == 0
    assert result.errors == 0
    assert len(result.details) == 1


@pytest.mark.asyncio
async def test_tag_pipeline_auto_applied(tmp_path):
    """Tag pipeline counts auto-applied documents."""
    from corvus.orchestrator.pipelines import run_tag_pipeline

    doc = _make_document()
    task = _make_task()
    routing = _make_routing_result(task, applied=True, action=GateAction.AUTO_EXECUTE)
    mock_raw = MagicMock()

    with (
        patch(
            "corvus.executors.document_tagger.tag_document",
            new_callable=AsyncMock,
            return_value=(task, mock_raw),
        ),
        patch(
            "corvus.router.tagging.resolve_and_route",
            new_callable=AsyncMock,
            return_value=routing,
        ),
    ):
        result = await run_tag_pipeline(
            paperless=_mock_paperless(docs=[doc], count=1),
            ollama=_mock_ollama(),
            model="test-model",
            limit=1,
            force_queue=False,
            queue_db_path=str(tmp_path / "queue.db"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
        )

    assert result.processed == 1
    assert result.auto_applied == 1
    assert result.queued == 0


@pytest.mark.asyncio
async def test_tag_pipeline_handles_errors(tmp_path):
    """Tag pipeline continues on document errors."""
    from corvus.orchestrator.pipelines import run_tag_pipeline

    doc = _make_document()
    progress = []

    with patch(
        "corvus.executors.document_tagger.tag_document",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ):
        result = await run_tag_pipeline(
            paperless=_mock_paperless(docs=[doc], count=1),
            ollama=_mock_ollama(),
            model="test-model",
            limit=1,
            queue_db_path=str(tmp_path / "queue.db"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            on_progress=progress.append,
        )

    assert result.processed == 0
    assert result.errors == 1
    assert any("ERROR" in msg for msg in progress)


@pytest.mark.asyncio
async def test_tag_pipeline_no_progress_callback(tmp_path):
    """Tag pipeline works with on_progress=None."""
    from corvus.orchestrator.pipelines import run_tag_pipeline

    result = await run_tag_pipeline(
        paperless=_mock_paperless(),
        ollama=_mock_ollama(),
        model="test-model",
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        on_progress=None,
    )

    assert result.processed == 0


# ------------------------------------------------------------------
# run_fetch_pipeline
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_pipeline_finds_documents():
    """Fetch pipeline returns document list."""
    from corvus.orchestrator.pipelines import run_fetch_pipeline

    doc = _make_document(doc_id=73, title="AT&T Invoice")
    interp = QueryInterpretation(
        confidence=0.9,
        reasoning="Looking for AT&T invoice",
        correspondent_name="AT&T",
        sort_order="newest",
    )
    params = ResolvedSearchParams(correspondent_id=10)
    mock_raw = MagicMock()
    progress = []

    with (
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, [doc], 1),
        ),
    ):
        result = await run_fetch_pipeline(
            paperless=_mock_paperless(),
            ollama=_mock_ollama(),
            model="test-model",
            query="find AT&T invoice",
            on_progress=progress.append,
        )

    assert isinstance(result, FetchPipelineResult)
    assert result.documents_found == 1
    assert len(result.documents) == 1
    assert result.documents[0]["id"] == 73
    assert result.documents[0]["title"] == "AT&T Invoice"
    assert result.interpretation_confidence == 0.9


@pytest.mark.asyncio
async def test_fetch_pipeline_no_results():
    """Fetch pipeline handles zero results."""
    from corvus.orchestrator.pipelines import run_fetch_pipeline

    interp = QueryInterpretation(
        confidence=0.8,
        reasoning="Test",
        sort_order="newest",
    )
    params = ResolvedSearchParams()
    mock_raw = MagicMock()

    with (
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, [], 0),
        ),
    ):
        result = await run_fetch_pipeline(
            paperless=_mock_paperless(),
            ollama=_mock_ollama(),
            model="test-model",
            query="nonexistent document",
        )

    assert result.documents_found == 0
    assert result.documents == []


@pytest.mark.asyncio
async def test_fetch_pipeline_with_warnings():
    """Fetch pipeline includes warnings from search params."""
    from corvus.orchestrator.pipelines import run_fetch_pipeline

    interp = QueryInterpretation(
        confidence=0.7,
        reasoning="Test",
        correspondent_name="Unknown Corp",
        sort_order="newest",
    )
    params = ResolvedSearchParams(
        warnings=["Correspondent not found: 'Unknown Corp'"],
    )
    mock_raw = MagicMock()
    progress = []

    with (
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, [], 0),
        ),
    ):
        result = await run_fetch_pipeline(
            paperless=_mock_paperless(),
            ollama=_mock_ollama(),
            model="test-model",
            query="docs from Unknown Corp",
            on_progress=progress.append,
        )

    assert len(result.warnings) == 1
    assert "Unknown Corp" in result.warnings[0]
    assert any("Warning" in msg for msg in progress)


@pytest.mark.asyncio
async def test_fetch_pipeline_with_fallback():
    """Fetch pipeline reports fallback usage."""
    from corvus.orchestrator.pipelines import run_fetch_pipeline

    doc = _make_document()
    interp = QueryInterpretation(confidence=0.8, reasoning="Test", sort_order="newest")
    params = ResolvedSearchParams(used_fallback=True)
    mock_raw = MagicMock()

    with (
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, [doc], 1),
        ),
    ):
        result = await run_fetch_pipeline(
            paperless=_mock_paperless(),
            ollama=_mock_ollama(),
            model="test-model",
            query="some query",
        )

    assert result.used_fallback is True


# ------------------------------------------------------------------
# run_status_pipeline
# ------------------------------------------------------------------


def test_status_pipeline_empty(tmp_path):
    """Status pipeline works with empty queue and audit log."""
    from corvus.orchestrator.pipelines import run_status_pipeline

    result = run_status_pipeline(
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )

    assert isinstance(result, StatusResult)
    assert result.pending_count == 0
    assert result.processed_24h == 0
    assert result.reviewed_24h == 0


def test_status_pipeline_with_pending(tmp_path):
    """Status pipeline reports pending count from queue."""
    from corvus.orchestrator.pipelines import run_status_pipeline
    from corvus.queue.review import ReviewQueue

    task = _make_task()
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1])
    with ReviewQueue(str(tmp_path / "queue.db")) as q:
        q.add(task, proposed)

    result = run_status_pipeline(
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )

    assert result.pending_count == 1


# ------------------------------------------------------------------
# run_digest_pipeline
# ------------------------------------------------------------------


def test_digest_pipeline_empty(tmp_path):
    """Digest pipeline works with no activity."""
    from corvus.orchestrator.pipelines import run_digest_pipeline

    result = run_digest_pipeline(
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        hours=1,
    )

    assert isinstance(result, DigestResult)
    assert "Corvus Daily Digest" in result.rendered_text
    assert "No activity during this period." in result.rendered_text


def test_digest_pipeline_custom_hours(tmp_path):
    """Digest pipeline accepts custom hours parameter."""
    from corvus.orchestrator.pipelines import run_digest_pipeline

    result = run_digest_pipeline(
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        hours=48,
    )

    assert isinstance(result, DigestResult)
    assert "Corvus Daily Digest" in result.rendered_text
