"""Tests for the SQLite-backed review queue."""

import pytest

from corvus.queue.review import ReviewQueue
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
    ReviewStatus,
    TagSuggestion,
)


@pytest.fixture()
def queue(tmp_path):
    """Create a ReviewQueue with a temporary database."""
    db_path = tmp_path / "test_queue.db"
    with ReviewQueue(db_path) as q:
        yield q


def _make_task(doc_id: int = 42) -> DocumentTaggingTask:
    return DocumentTaggingTask(
        document_id=doc_id,
        document_title=f"Test Document {doc_id}",
        content_snippet="Some content...",
        result=DocumentTaggingResult(
            suggested_tags=[
                TagSuggestion(tag_name="invoice", confidence=0.85),
                TagSuggestion(tag_name="utility", confidence=0.7),
            ],
            suggested_correspondent="AT&T",
            correspondent_confidence=0.8,
            suggested_document_type="Invoice",
            document_type_confidence=0.9,
            reasoning="Looks like a utility invoice from AT&T.",
        ),
        overall_confidence=0.65,
        gate_action=GateAction.QUEUE_FOR_REVIEW,
    )


def _make_proposed_update(doc_id: int = 42) -> ProposedDocumentUpdate:
    return ProposedDocumentUpdate(
        document_id=doc_id,
        add_tag_ids=[1, 2],
        set_correspondent_id=10,
        set_document_type_id=20,
    )


class TestAdd:
    def test_add_returns_item(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        assert item.status == ReviewStatus.PENDING
        assert item.task.document_id == 42
        assert item.proposed_update.add_tag_ids == [1, 2]
        assert item.reviewed_at is None
        assert len(item.id) == 36  # UUID format

    def test_add_persists(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        fetched = queue.get(item.id)
        assert fetched is not None
        assert fetched.id == item.id
        assert fetched.task.document_id == 42
        assert fetched.proposed_update.set_correspondent_id == 10

    def test_add_multiple(self, queue: ReviewQueue):
        queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))
        queue.add(_make_task(3), _make_proposed_update(3))
        assert queue.count_pending() == 3


class TestGet:
    def test_get_nonexistent(self, queue: ReviewQueue):
        assert queue.get("nonexistent-id") is None

    def test_get_roundtrip_preserves_data(self, queue: ReviewQueue):
        task = _make_task()
        proposed = _make_proposed_update()
        item = queue.add(task, proposed)
        fetched = queue.get(item.id)

        assert fetched.task.result.suggested_tags[0].tag_name == "invoice"
        assert fetched.task.result.suggested_correspondent == "AT&T"
        assert fetched.task.result.reasoning == "Looks like a utility invoice from AT&T."
        assert fetched.proposed_update.set_document_type_id == 20


class TestListPending:
    def test_empty_queue(self, queue: ReviewQueue):
        assert queue.list_pending() == []

    def test_returns_only_pending(self, queue: ReviewQueue):
        item1 = queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))
        queue.approve(item1.id)

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].task.document_id == 2

    def test_ordered_oldest_first(self, queue: ReviewQueue):
        queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))

        pending = queue.list_pending()
        assert pending[0].task.document_id == 1
        assert pending[1].task.document_id == 2


class TestApproveReject:
    def test_approve(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        approved = queue.approve(item.id, notes="Looks good")

        assert approved.status == ReviewStatus.APPROVED
        assert approved.reviewed_at is not None
        assert approved.reviewer_notes == "Looks good"

    def test_reject(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        rejected = queue.reject(item.id, notes="Wrong tags")

        assert rejected.status == ReviewStatus.REJECTED
        assert rejected.reviewed_at is not None
        assert rejected.reviewer_notes == "Wrong tags"

    def test_approve_without_notes(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        approved = queue.approve(item.id)
        assert approved.reviewer_notes is None

    def test_cannot_approve_nonexistent(self, queue: ReviewQueue):
        with pytest.raises(ValueError, match="not found"):
            queue.approve("nonexistent-id")

    def test_cannot_approve_already_approved(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        queue.approve(item.id)
        with pytest.raises(ValueError, match="current status is approved"):
            queue.approve(item.id)

    def test_cannot_reject_already_rejected(self, queue: ReviewQueue):
        item = queue.add(_make_task(), _make_proposed_update())
        queue.reject(item.id)
        with pytest.raises(ValueError, match="current status is rejected"):
            queue.reject(item.id)


class TestListAll:
    def test_list_all_newest_first(self, queue: ReviewQueue):
        queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))

        items = queue.list_all()
        assert items[0].task.document_id == 2
        assert items[1].task.document_id == 1

    def test_list_all_includes_all_statuses(self, queue: ReviewQueue):
        item1 = queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))
        queue.approve(item1.id)

        items = queue.list_all()
        assert len(items) == 2

    def test_list_all_respects_limit(self, queue: ReviewQueue):
        for i in range(5):
            queue.add(_make_task(i), _make_proposed_update(i))

        items = queue.list_all(limit=3)
        assert len(items) == 3


class TestCountPending:
    def test_empty(self, queue: ReviewQueue):
        assert queue.count_pending() == 0

    def test_after_add_and_approve(self, queue: ReviewQueue):
        item1 = queue.add(_make_task(1), _make_proposed_update(1))
        queue.add(_make_task(2), _make_proposed_update(2))
        assert queue.count_pending() == 2

        queue.approve(item1.id)
        assert queue.count_pending() == 1
