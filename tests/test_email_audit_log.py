"""Tests for the email JSONL audit log purge functionality."""

from datetime import UTC, datetime, timedelta

from corvus.audit.email_logger import EmailAuditLog
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailTriageTask,
)


def _make_task(uid: str = "100") -> EmailTriageTask:
    return EmailTriageTask(
        uid=uid,
        account_email="test@example.com",
        subject="Test Email",
        from_address="sender@example.com",
        classification=EmailClassification(
            category=EmailCategory.SPAM,
            confidence=0.9,
            reasoning="Test reasoning",
            suggested_action="delete",
        ),
        proposed_action=EmailAction(action_type=EmailActionType.DELETE),
        overall_confidence=0.9,
        gate_action=GateAction.AUTO_EXECUTE,
    )


class TestPurge:
    def test_purge_removes_old_keeps_recent(self, tmp_path):
        audit = EmailAuditLog(tmp_path / "email_audit.jsonl")

        audit.log_auto_applied(_make_task("1"))
        cutoff = datetime.now(UTC)
        audit.log_auto_applied(_make_task("2"))

        purged = audit.purge_before(cutoff)
        assert purged == 1

        remaining = audit.read_entries()
        assert len(remaining) == 1
        assert remaining[0].uid == "2"

    def test_purge_returns_correct_count(self, tmp_path):
        audit = EmailAuditLog(tmp_path / "email_audit.jsonl")

        for i in range(5):
            audit.log_auto_applied(_make_task(str(i)))

        cutoff = datetime.now(UTC)
        audit.log_auto_applied(_make_task("99"))

        purged = audit.purge_before(cutoff)
        assert purged == 5

    def test_purge_noop_when_nothing_old(self, tmp_path):
        log_path = tmp_path / "email_audit.jsonl"
        audit = EmailAuditLog(log_path)

        audit.log_auto_applied(_make_task())

        old_mtime = log_path.stat().st_mtime

        purged = audit.purge_before(datetime.now(UTC) - timedelta(days=999))
        assert purged == 0
        assert log_path.stat().st_mtime == old_mtime

    def test_purge_nonexistent_file(self, tmp_path):
        audit = EmailAuditLog(tmp_path / "nonexistent.jsonl")
        assert audit.purge_before(datetime.now(UTC)) == 0
