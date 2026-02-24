"""Tests for the Corvus CLI entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import click.testing
import pytest

from corvus.cli import cli
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    ReviewStatus,
    TagSuggestion,
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


def _make_document(doc_id: int = 1, title: str = "Test Invoice", tags: list[int] | None = None):
    return PaperlessDocument(
        id=doc_id,
        title=title,
        content="Invoice from Acme Corp for $100.00 due 2025-01-15",
        tags=tags or [],
        correspondent=None,
        document_type=None,
        created="2025-01-01T00:00:00Z",
        added="2025-01-01T00:00:00Z",
        original_filename="invoice.pdf",
    )


def _make_task(doc_id: int = 1, title: str = "Test Invoice"):
    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=title,
        content_snippet="Invoice from Acme Corp...",
        result=DocumentTaggingResult(
            suggested_tags=[
                TagSuggestion(tag_name="invoice", confidence=0.95),
                TagSuggestion(tag_name="acme-corp", confidence=0.85),
            ],
            suggested_correspondent="Acme Corp",
            suggested_document_type="Invoice",
            correspondent_confidence=0.9,
            document_type_confidence=0.88,
            reasoning="Document appears to be an invoice from Acme Corp.",
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
            add_tag_ids=[1, 2],
            set_correspondent_id=3,
            set_document_type_id=4,
        ),
        applied=applied,
        effective_action=action,
    )


SAMPLE_TAGS = [PaperlessTag(id=1, name="invoice", slug="invoice")]
SAMPLE_CORRESPONDENTS = [PaperlessCorrespondent(id=3, name="Acme Corp", slug="acme-corp")]
SAMPLE_DOC_TYPES = [PaperlessDocumentType(id=4, name="Invoice", slug="invoice")]


@pytest.fixture()
def runner():
    return click.testing.CliRunner()


def _mock_ollama(model_name="gemma3:latest"):
    """Create a mock OllamaClient context manager."""
    mock = AsyncMock()
    mock.pick_instruct_model.return_value = model_name
    return mock


def _mock_paperless(docs=None, count=0):
    """Create a mock PaperlessClient context manager."""
    mock = AsyncMock()
    mock.list_documents.return_value = (docs or [], count)
    mock.list_tags.return_value = SAMPLE_TAGS
    mock.list_correspondents.return_value = SAMPLE_CORRESPONDENTS
    mock.list_document_types.return_value = SAMPLE_DOC_TYPES
    # get_document_url is sync — use MagicMock to avoid coroutine return
    mock.get_document_url = MagicMock(return_value="")
    return mock


def _patch_async_context(target, mock_instance):
    """Create a patch that makes a class act as an async context manager returning mock_instance."""
    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return patch(target, mock_cls)


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def test_tag_rejects_missing_config(runner, monkeypatch):
    """Tag command should fail if Paperless config is missing."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "")
    result = runner.invoke(cli, ["tag"])
    assert result.exit_code != 0
    assert "Missing required config" in result.output


def test_review_rejects_missing_config(runner, monkeypatch):
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "placeholder")
    result = runner.invoke(cli, ["review"])
    assert result.exit_code != 0
    assert "Missing required config" in result.output


# ------------------------------------------------------------------
# corvus status
# ------------------------------------------------------------------


def test_status_empty(runner, tmp_path, monkeypatch):
    """Status command works with empty queue and audit log."""
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Pending review:" in result.output
    assert "Processed (24h):" in result.output
    assert "Reviewed (24h):" in result.output


# ------------------------------------------------------------------
# corvus digest
# ------------------------------------------------------------------


def test_digest_empty(runner, tmp_path, monkeypatch):
    """Digest command works with no activity."""
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    result = runner.invoke(cli, ["digest", "--hours", "1"])
    assert result.exit_code == 0
    assert "Corvus Daily Digest" in result.output
    assert "No activity during this period." in result.output


# ------------------------------------------------------------------
# corvus tag
# ------------------------------------------------------------------


def test_tag_no_documents(runner, monkeypatch):
    """Tag command handles empty document list gracefully."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", _mock_ollama()),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", _mock_paperless()),
    ):
        result = runner.invoke(cli, ["tag", "-n", "5"])

    assert result.exit_code == 0
    assert "No documents to process" in result.output


def test_tag_processes_documents(runner, tmp_path, monkeypatch):
    """Tag command processes documents and queues them for review."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    doc = _make_document()
    task = _make_task()
    routing_result = _make_routing_result(task, applied=False, action=GateAction.QUEUE_FOR_REVIEW)

    mock_raw = MagicMock()
    mock_raw.done = True

    with (
        _patch_async_context(
            "corvus.integrations.ollama.OllamaClient", _mock_ollama()
        ),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient",
            _mock_paperless(docs=[doc], count=1),
        ),
        patch(
            "corvus.executors.document_tagger.tag_document",
            new_callable=AsyncMock,
            return_value=(task, mock_raw),
        ),
        patch(
            "corvus.router.tagging.resolve_and_route",
            new_callable=AsyncMock,
            return_value=routing_result,
        ),
    ):
        result = runner.invoke(cli, ["tag", "-n", "1"])

    assert result.exit_code == 0
    assert "Test Invoice" in result.output
    assert "invoice" in result.output
    assert "Processed: 1" in result.output
    assert "Errors: 0" in result.output


def test_tag_handles_document_error(runner, tmp_path, monkeypatch):
    """Tag command continues batch when a single document fails."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    doc = _make_document()

    with (
        _patch_async_context(
            "corvus.integrations.ollama.OllamaClient", _mock_ollama()
        ),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient",
            _mock_paperless(docs=[doc], count=1),
        ),
        patch(
            "corvus.executors.document_tagger.tag_document",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM down"),
        ),
    ):
        result = runner.invoke(cli, ["tag", "-n", "1"])

    assert result.exit_code == 0
    assert "Errors: 1" in result.output


def test_tag_no_model_available(runner, monkeypatch):
    """Tag command fails if no Ollama models are available."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    with (
        _patch_async_context(
            "corvus.integrations.ollama.OllamaClient", _mock_ollama(model_name=None)
        ),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient", _mock_paperless()
        ),
    ):
        result = runner.invoke(cli, ["tag"])

    assert result.exit_code != 0
    assert "No models available" in result.output


# ------------------------------------------------------------------
# corvus review
# ------------------------------------------------------------------


def test_review_no_pending(runner, tmp_path, monkeypatch):
    """Review command handles empty queue."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))

    result = runner.invoke(cli, ["review"])
    assert result.exit_code == 0
    assert "No pending items" in result.output


def test_review_approve(runner, tmp_path, monkeypatch):
    """Review command can approve an item."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from corvus.queue.review import ReviewQueue

    task = _make_task()
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1, 2])
    with ReviewQueue(tmp_path / "queue.db") as q:
        q.add(task, proposed)

    routing_result = _make_routing_result(task, applied=True, action=GateAction.AUTO_EXECUTE)

    with (
        patch(
            "corvus.router.tagging.apply_approved_update",
            new_callable=AsyncMock,
            return_value=routing_result,
        ),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient", AsyncMock()
        ),
    ):
        result = runner.invoke(cli, ["review"], input="a\n")

    assert result.exit_code == 0
    assert "Approved and applied" in result.output
    assert "Approved: 1" in result.output


def test_review_reject(runner, tmp_path, monkeypatch):
    """Review command can reject an item."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from corvus.queue.review import ReviewQueue

    task = _make_task()
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1, 2])
    with ReviewQueue(tmp_path / "queue.db") as q:
        q.add(task, proposed)

    with _patch_async_context(
        "corvus.integrations.paperless.PaperlessClient", AsyncMock()
    ):
        result = runner.invoke(cli, ["review"], input="r\nNot relevant\n")

    assert result.exit_code == 0
    assert "Rejected" in result.output
    assert "Rejected: 1" in result.output


def test_review_edit_adds_tags(runner, tmp_path, monkeypatch):
    """Review edit option prompts for tags and applies them."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from corvus.queue.review import ReviewQueue

    task = _make_task()
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1, 2])
    with ReviewQueue(tmp_path / "queue.db") as q:
        q.add(task, proposed)

    routing_result = _make_routing_result(task, applied=True, action=GateAction.AUTO_EXECUTE)
    mock_apply = AsyncMock(return_value=routing_result)

    with (
        patch("corvus.router.tagging.apply_approved_update", mock_apply),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient", AsyncMock()
        ),
    ):
        result = runner.invoke(cli, ["review"], input="e\nat&t, utilities\n")

    assert result.exit_code == 0
    assert "extra tags" in result.output
    assert "at&t" in result.output
    assert "Approved: 1" in result.output

    # Verify apply_approved_update was called with extra_tag_names
    mock_apply.assert_called_once()
    call_kwargs = mock_apply.call_args.kwargs
    assert call_kwargs["extra_tag_names"] == ["at&t", "utilities"]

    # Verify queue item marked as modified
    with ReviewQueue(tmp_path / "queue.db") as q:
        items = q.list_all()
        assert items[0].status == ReviewStatus.MODIFIED
        assert "at&t" in items[0].reviewer_notes


def test_review_edit_empty_tags_approves_normally(runner, tmp_path, monkeypatch):
    """Edit with empty tag input falls back to normal approve."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from corvus.queue.review import ReviewQueue

    task = _make_task()
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1, 2])
    with ReviewQueue(tmp_path / "queue.db") as q:
        q.add(task, proposed)

    routing_result = _make_routing_result(task, applied=True, action=GateAction.AUTO_EXECUTE)
    mock_apply = AsyncMock(return_value=routing_result)

    with (
        patch("corvus.router.tagging.apply_approved_update", mock_apply),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient", AsyncMock()
        ),
    ):
        result = runner.invoke(cli, ["review"], input="e\n\n")

    assert result.exit_code == 0
    assert "no tags entered" in result.output
    assert "Approved and applied" in result.output

    # Verify apply_approved_update was called without extra_tag_names
    mock_apply.assert_called_once()
    call_kwargs = mock_apply.call_args.kwargs
    assert call_kwargs["extra_tag_names"] is None

    # Verify queue item marked as approved (not modified)
    with ReviewQueue(tmp_path / "queue.db") as q:
        items = q.list_all()
        assert items[0].status == ReviewStatus.APPROVED


def test_review_skip_and_quit(runner, tmp_path, monkeypatch):
    """Review command supports skip and quit."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")
    monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

    from corvus.queue.review import ReviewQueue

    task1 = _make_task(doc_id=1, title="Doc 1")
    task2 = _make_task(doc_id=2, title="Doc 2")
    proposed = ProposedDocumentUpdate(document_id=1, add_tag_ids=[1])
    proposed2 = ProposedDocumentUpdate(document_id=2, add_tag_ids=[2])
    with ReviewQueue(tmp_path / "queue.db") as q:
        q.add(task1, proposed)
        q.add(task2, proposed2)

    with _patch_async_context(
        "corvus.integrations.paperless.PaperlessClient", AsyncMock()
    ):
        result = runner.invoke(cli, ["review"], input="s\nq\n")

    assert result.exit_code == 0
    assert "Skipped" in result.output
    assert "Quitting review" in result.output
    assert "Skipped: 1" in result.output


# ------------------------------------------------------------------
# corvus fetch
# ------------------------------------------------------------------


def _make_interpretation(confidence=0.9, correspondent=None, doc_type=None, **kwargs):
    from corvus.schemas.document_retrieval import QueryInterpretation

    defaults = {
        "confidence": confidence,
        "reasoning": "Test interpretation",
        "correspondent_name": correspondent,
        "document_type_name": doc_type,
        "sort_order": "newest",
    }
    defaults.update(kwargs)
    return QueryInterpretation(**defaults)


def _make_resolved_params(**kwargs):
    from corvus.schemas.document_retrieval import ResolvedSearchParams

    return ResolvedSearchParams(**kwargs)


def test_fetch_rejects_missing_config(runner, monkeypatch):
    """Fetch command should fail if Paperless config is missing."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "")
    result = runner.invoke(cli, ["fetch", "test"])
    assert result.exit_code != 0
    assert "Missing required config" in result.output


def test_fetch_help(runner):
    result = runner.invoke(cli, ["fetch", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--method" in result.output
    assert "--download-dir" in result.output
    assert "--keep-alive" in result.output


def test_fetch_single_result_browser(runner, monkeypatch):
    """Fetch with a single result auto-delivers via browser."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    doc = _make_document(doc_id=73, title="Invoice - AT&T Wireless")
    interp = _make_interpretation(correspondent="AT&T", doc_type="Invoice")
    params = _make_resolved_params(correspondent_id=10, document_type_id=20)
    mock_raw = MagicMock()
    mock_raw.done = True

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=[doc], count=1)
    mock_paperless.get_document_url.return_value = "http://localhost:8000/documents/73/details"

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
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
        patch("webbrowser.open") as mock_browser,
    ):
        result = runner.invoke(cli, ["fetch", "invoice", "from", "AT&T"])

    assert result.exit_code == 0
    assert "Invoice - AT&T Wireless" in result.output
    assert "Opening" in result.output
    mock_browser.assert_called_once_with("http://localhost:8000/documents/73/details")


def test_fetch_multiple_results_select(runner, monkeypatch):
    """Fetch with multiple results shows numbered list and accepts selection."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    docs = [
        _make_document(doc_id=73, title="Invoice - AT&T March"),
        _make_document(doc_id=58, title="Invoice - AT&T February"),
    ]
    interp = _make_interpretation(correspondent="AT&T")
    params = _make_resolved_params(correspondent_id=10)
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=docs, count=2)
    mock_paperless.get_document_url.return_value = "http://localhost:8000/documents/58/details"

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, docs, 2),
        ),
        patch("webbrowser.open") as mock_browser,
    ):
        result = runner.invoke(cli, ["fetch", "invoices", "from", "AT&T"], input="2\n")

    assert result.exit_code == 0
    assert "Found 2 document(s)" in result.output
    assert "Invoice - AT&T March" in result.output
    assert "Invoice - AT&T February" in result.output
    mock_browser.assert_called_once()


def test_fetch_no_results(runner, monkeypatch):
    """Fetch with no results displays a message."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    interp = _make_interpretation(correspondent="Nobody")
    params = _make_resolved_params(warnings=["Correspondent not found: 'Nobody'"])
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless()

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
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
        result = runner.invoke(cli, ["fetch", "stuff", "from", "Nobody"])

    assert result.exit_code == 0
    assert "No documents found" in result.output
    assert "Warning" in result.output


def test_fetch_low_confidence_abort(runner, monkeypatch):
    """Fetch aborts when confidence is low and user declines."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    interp = _make_interpretation(confidence=0.3)
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless()

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
    ):
        result = runner.invoke(cli, ["fetch", "vague", "query"], input="n\n")

    assert result.exit_code == 0
    assert "Low confidence" in result.output
    assert "Aborted" in result.output


def test_fetch_low_confidence_continue(runner, monkeypatch):
    """Fetch continues when confidence is low and user confirms."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    doc = _make_document(doc_id=1, title="Some Doc")
    interp = _make_interpretation(confidence=0.3)
    params = _make_resolved_params()
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=[doc], count=1)
    mock_paperless.get_document_url.return_value = "http://localhost:8000/documents/1/details"

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
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
        patch("webbrowser.open"),
    ):
        result = runner.invoke(cli, ["fetch", "vague", "query"], input="y\n")

    assert result.exit_code == 0
    assert "Some Doc" in result.output


def test_fetch_download(runner, monkeypatch, tmp_path):
    """Fetch with --method=download saves file locally."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    doc = _make_document(doc_id=42, title="Invoice PDF")
    interp = _make_interpretation()
    params = _make_resolved_params()
    mock_raw = MagicMock()
    download_path = tmp_path / "invoice.pdf"

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=[doc], count=1)
    mock_paperless.download_document.return_value = download_path

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
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
        result = runner.invoke(
            cli,
            ["fetch", "invoice", "--method", "download", "--download-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert "Downloaded" in result.output
    mock_paperless.download_document.assert_called_once()


def test_fetch_quit_selection(runner, monkeypatch):
    """User can quit at the selection prompt."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    docs = [_make_document(1, "Doc 1"), _make_document(2, "Doc 2")]
    interp = _make_interpretation()
    params = _make_resolved_params()
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=docs, count=2)

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, docs, 2),
        ),
    ):
        result = runner.invoke(cli, ["fetch", "docs"], input="q\n")

    assert result.exit_code == 0
    assert "Aborted" in result.output


def test_fetch_no_model_available(runner, monkeypatch):
    """Fetch command fails if no Ollama models are available."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    with (
        _patch_async_context(
            "corvus.integrations.ollama.OllamaClient", _mock_ollama(model_name=None)
        ),
        _patch_async_context(
            "corvus.integrations.paperless.PaperlessClient", _mock_paperless()
        ),
    ):
        result = runner.invoke(cli, ["fetch", "test"])

    assert result.exit_code != 0
    assert "No models available" in result.output


def test_fetch_many_results_truncation(runner, monkeypatch):
    """Fetch with >10 results shows first 10 and suggests refining."""
    monkeypatch.setattr("corvus.cli.PAPERLESS_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr("corvus.cli.PAPERLESS_API_TOKEN", "test-token")

    docs = [_make_document(doc_id=i, title=f"Doc {i}") for i in range(1, 16)]
    interp = _make_interpretation()
    params = _make_resolved_params()
    mock_raw = MagicMock()

    mock_ollama = _mock_ollama()
    mock_paperless = _mock_paperless(docs=docs, count=25)
    mock_paperless.get_document_url.return_value = "http://localhost:8000/documents/5/details"

    with (
        _patch_async_context("corvus.integrations.ollama.OllamaClient", mock_ollama),
        _patch_async_context("corvus.integrations.paperless.PaperlessClient", mock_paperless),
        patch(
            "corvus.executors.query_interpreter.interpret_query",
            new_callable=AsyncMock,
            return_value=(interp, mock_raw),
        ),
        patch(
            "corvus.router.retrieval.resolve_and_search",
            new_callable=AsyncMock,
            return_value=(params, docs, 25),
        ),
        patch("webbrowser.open"),
    ):
        result = runner.invoke(cli, ["fetch", "docs"], input="5\n")

    assert result.exit_code == 0
    assert "25 document(s)" in result.output
    assert "refining" in result.output
    # Only first 10 shown
    assert "Doc 10" in result.output
    assert "Doc 11" not in result.output


# ------------------------------------------------------------------
# corvus --help
# ------------------------------------------------------------------


def test_help(runner):
    """Main --help shows all commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "tag" in result.output
    assert "review" in result.output
    assert "digest" in result.output
    assert "status" in result.output
    assert "fetch" in result.output


def test_tag_help(runner):
    result = runner.invoke(cli, ["tag", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.output
    assert "--model" in result.output
    assert "--force-queue" in result.output
