"""Tests for the Corvus CLI entry point."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import click.testing
import pytest

from corvus.cli import cli
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    ReviewQueueItem,
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


def test_tag_help(runner):
    result = runner.invoke(cli, ["tag", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.output
    assert "--model" in result.output
    assert "--force-queue" in result.output
