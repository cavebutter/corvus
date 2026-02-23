"""Tests for daily digest generation."""

import pytest

from corvus.audit.logger import AuditLog
from corvus.digest.daily import generate_digest, render_text
from corvus.queue.review import ReviewQueue
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    TagSuggestion,
)


@pytest.fixture()
def audit_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture()
def review_queue(tmp_path):
    with ReviewQueue(tmp_path / "queue.db") as q:
        yield q


def _make_task(
    doc_id: int = 1,
    gate_action: GateAction = GateAction.QUEUE_FOR_REVIEW,
    confidence: float = 0.65,
) -> DocumentTaggingTask:
    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=f"Document {doc_id}",
        content_snippet="Content...",
        result=DocumentTaggingResult(
            suggested_tags=[
                TagSuggestion(tag_name="invoice", confidence=0.85),
                TagSuggestion(tag_name="utility", confidence=0.7),
            ],
            suggested_correspondent="AT&T",
            correspondent_confidence=0.8,
            suggested_document_type="Invoice",
            document_type_confidence=0.9,
            reasoning="Looks like an AT&T invoice.",
        ),
        overall_confidence=confidence,
        gate_action=gate_action,
    )


def _make_update(doc_id: int = 1) -> ProposedDocumentUpdate:
    return ProposedDocumentUpdate(
        document_id=doc_id,
        add_tag_ids=[1, 2],
        set_correspondent_id=10,
        set_document_type_id=20,
    )


class TestGenerateDigest:
    def test_empty_digest(self, audit_log, review_queue):
        digest = generate_digest(audit_log, review_queue)

        assert digest.total_processed == 0
        assert digest.total_reviewed == 0
        assert digest.pending_review_count == 0

    def test_auto_applied_entries(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )
        audit_log.log_auto_applied(
            _make_task(2, GateAction.AUTO_EXECUTE, 0.92), _make_update(2)
        )

        digest = generate_digest(audit_log, review_queue)

        assert len(digest.auto_applied) == 2
        assert digest.auto_applied[0].document_id == 1
        assert digest.auto_applied[0].confidence == 0.95

    def test_flagged_entries(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.FLAG_IN_DIGEST, 0.8), _make_update(1)
        )

        digest = generate_digest(audit_log, review_queue)

        assert len(digest.flagged) == 1
        assert len(digest.auto_applied) == 0
        assert digest.flagged[0].document_id == 1

    def test_queued_entries(self, audit_log, review_queue):
        task = _make_task(1, GateAction.QUEUE_FOR_REVIEW, 0.5)
        update = _make_update(1)
        audit_log.log_queued_for_review(task, update)
        review_queue.add(task, update)

        digest = generate_digest(audit_log, review_queue)

        assert len(digest.queued_for_review) == 1
        assert digest.pending_review_count == 1

    def test_review_outcomes(self, audit_log, review_queue):
        audit_log.log_review_approved(_make_task(1), _make_update(1))
        audit_log.log_review_rejected(_make_task(2), _make_update(2))

        digest = generate_digest(audit_log, review_queue)

        assert len(digest.review_approved) == 1
        assert len(digest.review_rejected) == 1
        assert digest.total_reviewed == 2

    def test_mixed_activity(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )
        audit_log.log_auto_applied(
            _make_task(2, GateAction.FLAG_IN_DIGEST, 0.8), _make_update(2)
        )
        task3 = _make_task(3, GateAction.QUEUE_FOR_REVIEW, 0.5)
        audit_log.log_queued_for_review(task3, _make_update(3))
        review_queue.add(task3, _make_update(3))
        audit_log.log_review_approved(_make_task(4), _make_update(4))

        digest = generate_digest(audit_log, review_queue)

        assert len(digest.auto_applied) == 1
        assert len(digest.flagged) == 1
        assert len(digest.queued_for_review) == 1
        assert len(digest.review_approved) == 1
        assert digest.total_processed == 3
        assert digest.total_reviewed == 1
        assert digest.pending_review_count == 1

    def test_digest_item_fields(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )

        digest = generate_digest(audit_log, review_queue)
        item = digest.auto_applied[0]

        assert item.tags == ["invoice", "utility"]
        assert item.correspondent == "AT&T"
        assert item.document_type == "Invoice"
        assert item.reasoning == "Looks like an AT&T invoice."


class TestRenderText:
    def test_empty_digest_renders(self, audit_log, review_queue):
        digest = generate_digest(audit_log, review_queue)
        text = render_text(digest)

        assert "Corvus Daily Digest" in text
        assert "No activity during this period." in text

    def test_renders_all_sections(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )
        audit_log.log_auto_applied(
            _make_task(2, GateAction.FLAG_IN_DIGEST, 0.8), _make_update(2)
        )
        audit_log.log_queued_for_review(
            _make_task(3, GateAction.QUEUE_FOR_REVIEW, 0.5), _make_update(3)
        )
        audit_log.log_review_approved(_make_task(4), _make_update(4))

        digest = generate_digest(audit_log, review_queue)
        text = render_text(digest)

        assert "## Auto-Applied" in text
        assert "## Flagged Items" in text
        assert "## Queued for Review" in text
        assert "## Approved" in text
        assert "Document 1" in text
        assert "Document 2" in text

    def test_renders_item_details(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )

        digest = generate_digest(audit_log, review_queue)
        text = render_text(digest)

        assert "95%" in text
        assert "invoice, utility" in text
        assert "AT&T" in text
        assert "Invoice" in text

    def test_summary_counts(self, audit_log, review_queue):
        audit_log.log_auto_applied(
            _make_task(1, GateAction.AUTO_EXECUTE, 0.95), _make_update(1)
        )
        audit_log.log_queued_for_review(
            _make_task(2, GateAction.QUEUE_FOR_REVIEW, 0.5), _make_update(2)
        )

        digest = generate_digest(audit_log, review_queue)
        text = render_text(digest)

        assert "Auto-applied: 1" in text
        assert "Queued for review: 1" in text
