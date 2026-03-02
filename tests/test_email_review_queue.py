"""Tests for the email review queue purge functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from corvus.queue.email_review import EmailReviewQueue
from corvus.schemas.document_tagging import GateAction, ReviewStatus
from corvus.schemas.email import (
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailTriageTask,
)


@pytest.fixture()
def queue(tmp_path):
    db_path = tmp_path / "email_queue.db"
    with EmailReviewQueue(db_path) as q:
        yield q


def _make_task(uid: str = "100") -> EmailTriageTask:
    return EmailTriageTask(
        uid=uid,
        account_email="test@example.com",
        subject=f"Test Email {uid}",
        from_address="sender@example.com",
        classification=EmailClassification(
            category=EmailCategory.SPAM,
            confidence=0.65,
            reasoning="Test reasoning",
            suggested_action="delete",
        ),
        proposed_action=EmailAction(action_type=EmailActionType.DELETE),
        overall_confidence=0.65,
        gate_action=GateAction.QUEUE_FOR_REVIEW,
    )


class TestPurgeResolved:
    def test_purge_deletes_old_resolved(self, queue: EmailReviewQueue):
        item1 = queue.add(_make_task("1"))
        item2 = queue.add(_make_task("2"))
        queue.approve(item1.id)
        queue.reject(item2.id)

        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        purged = queue.purge_resolved(cutoff)
        assert purged == 2
        assert queue.list_all() == []

    def test_purge_keeps_pending_regardless_of_age(self, queue: EmailReviewQueue):
        queue.add(_make_task("1"))  # pending
        item2 = queue.add(_make_task("2"))
        queue.approve(item2.id)

        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        purged = queue.purge_resolved(cutoff)
        assert purged == 1

        remaining = queue.list_all()
        assert len(remaining) == 1
        assert remaining[0].status == ReviewStatus.PENDING

    def test_purge_returns_correct_count(self, queue: EmailReviewQueue):
        items = []
        for i in range(5):
            items.append(queue.add(_make_task(str(i))))

        for item in items[:3]:
            queue.approve(item.id)

        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        purged = queue.purge_resolved(cutoff)
        assert purged == 3
        assert queue.count_pending() == 2

    def test_purge_respects_cutoff(self, queue: EmailReviewQueue):
        item = queue.add(_make_task("1"))
        queue.approve(item.id)

        # Cutoff before the review happened
        cutoff = datetime.now(UTC) - timedelta(days=1)
        purged = queue.purge_resolved(cutoff)
        assert purged == 0
