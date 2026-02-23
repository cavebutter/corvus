"""Tests for the JSONL audit log."""

from datetime import UTC, datetime

from corvus.audit.logger import AuditLog
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    TagSuggestion,
)


def _make_task(doc_id: int = 42) -> DocumentTaggingTask:
    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=f"Test Document {doc_id}",
        content_snippet="Some content...",
        result=DocumentTaggingResult(
            suggested_tags=[TagSuggestion(tag_name="invoice", confidence=0.85)],
            suggested_correspondent="AT&T",
            correspondent_confidence=0.8,
            reasoning="Test reasoning",
        ),
        overall_confidence=0.65,
        gate_action=GateAction.QUEUE_FOR_REVIEW,
    )


def _make_proposed_update(doc_id: int = 42) -> ProposedDocumentUpdate:
    return ProposedDocumentUpdate(
        document_id=doc_id,
        add_tag_ids=[1, 2],
        set_correspondent_id=10,
    )


class TestLogActions:
    def test_log_auto_applied(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        entry = audit.log_auto_applied(_make_task(), _make_proposed_update())

        assert entry.action == "auto_applied"
        assert entry.applied is True
        assert entry.document_id == 42

    def test_log_queued_for_review(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        entry = audit.log_queued_for_review(_make_task(), _make_proposed_update())

        assert entry.action == "queued_for_review"
        assert entry.applied is False

    def test_log_review_approved(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        entry = audit.log_review_approved(_make_task(), _make_proposed_update())

        assert entry.action == "review_approved"
        assert entry.applied is True

    def test_log_review_rejected(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        entry = audit.log_review_rejected(_make_task(), _make_proposed_update())

        assert entry.action == "review_rejected"
        assert entry.applied is False

    def test_entry_has_timestamp(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        before = datetime.now(UTC)
        entry = audit.log_auto_applied(_make_task(), _make_proposed_update())
        after = datetime.now(UTC)

        assert before <= entry.timestamp <= after


class TestPersistence:
    def test_entries_persist_to_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLog(log_path)

        audit.log_auto_applied(_make_task(1), _make_proposed_update(1))
        audit.log_queued_for_review(_make_task(2), _make_proposed_update(2))

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_read_entries_roundtrip(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")

        audit.log_auto_applied(_make_task(1), _make_proposed_update(1))
        audit.log_queued_for_review(_make_task(2), _make_proposed_update(2))

        entries = audit.read_entries()
        assert len(entries) == 2
        assert entries[0].action == "auto_applied"
        assert entries[0].document_id == 1
        assert entries[1].action == "queued_for_review"
        assert entries[1].document_id == 2

    def test_read_entries_preserves_nested_data(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        audit.log_auto_applied(_make_task(), _make_proposed_update())

        entries = audit.read_entries()
        assert entries[0].task.result.suggested_tags[0].tag_name == "invoice"
        assert entries[0].proposed_update.set_correspondent_id == 10

    def test_read_empty_file(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")
        assert audit.read_entries() == []


class TestFiltering:
    def test_filter_by_since(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")

        audit.log_auto_applied(_make_task(1), _make_proposed_update(1))
        cutoff = datetime.now(UTC)
        audit.log_queued_for_review(_make_task(2), _make_proposed_update(2))

        entries = audit.read_entries(since=cutoff)
        assert len(entries) == 1
        assert entries[0].document_id == 2

    def test_limit(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")

        for i in range(5):
            audit.log_auto_applied(_make_task(i), _make_proposed_update(i))

        entries = audit.read_entries(limit=3)
        assert len(entries) == 3
        # Limit takes the newest (last) entries
        assert entries[0].document_id == 2
        assert entries[2].document_id == 4

    def test_since_and_limit_combined(self, tmp_path):
        audit = AuditLog(tmp_path / "audit.jsonl")

        audit.log_auto_applied(_make_task(1), _make_proposed_update(1))
        cutoff = datetime.now(UTC)
        for i in range(2, 6):
            audit.log_auto_applied(_make_task(i), _make_proposed_update(i))

        entries = audit.read_entries(since=cutoff, limit=2)
        assert len(entries) == 2
        assert entries[0].document_id == 4
        assert entries[1].document_id == 5


class TestEdgeCases:
    def test_creates_parent_dirs(self, tmp_path):
        audit = AuditLog(tmp_path / "deep" / "nested" / "audit.jsonl")
        audit.log_auto_applied(_make_task(), _make_proposed_update())

        entries = audit.read_entries()
        assert len(entries) == 1

    def test_append_across_instances(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"

        audit1 = AuditLog(log_path)
        audit1.log_auto_applied(_make_task(1), _make_proposed_update(1))

        audit2 = AuditLog(log_path)
        audit2.log_queued_for_review(_make_task(2), _make_proposed_update(2))

        entries = audit2.read_entries()
        assert len(entries) == 2
