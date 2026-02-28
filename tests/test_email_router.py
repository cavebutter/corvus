"""Tests for the email triage router (corvus/router/email.py).

Verifies confidence-gate logic and IMAP action dispatch without
requiring a real mail server or LLM.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from corvus.router.email import execute_email_action, route_email_action
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailTriageTask,
)


# --- Helpers ---


def _make_task(
    gate_action: GateAction = GateAction.AUTO_EXECUTE,
    action_type: EmailActionType = EmailActionType.DELETE,
    confidence: float = 0.95,
    target_folder: str | None = None,
    flag_name: str | None = None,
) -> EmailTriageTask:
    """Build an EmailTriageTask for testing."""
    return EmailTriageTask(
        uid="123",
        account_email="test@example.com",
        subject="Test Subject",
        from_address="sender@example.com",
        classification=EmailClassification(
            category=EmailCategory.SPAM,
            confidence=confidence,
            reasoning="Test reasoning",
            suggested_action="delete",
        ),
        proposed_action=EmailAction(
            action_type=action_type,
            target_folder=target_folder,
            flag_name=flag_name,
        ),
        overall_confidence=confidence,
        gate_action=gate_action,
    )


def _mock_imap() -> AsyncMock:
    """Create a mock ImapClient with all async action methods."""
    imap = AsyncMock()
    imap.delete = AsyncMock()
    imap.move = AsyncMock()
    imap.flag = AsyncMock()
    imap.mark_read = AsyncMock()
    return imap


def _mock_audit_log() -> MagicMock:
    """Create a mock EmailAuditLog."""
    audit = MagicMock()
    audit.log_auto_applied = MagicMock()
    audit.log_queued_for_review = MagicMock()
    return audit


def _mock_review_queue() -> MagicMock:
    """Create a mock EmailReviewQueue."""
    queue = MagicMock()
    queue.add = MagicMock()
    return queue


# --- route_email_action tests ---


class TestRouteForceQueue:
    """force_queue=True always queues, regardless of gate_action."""

    async def test_force_queue_overrides_auto_execute(self):
        task = _make_task(gate_action=GateAction.AUTO_EXECUTE, confidence=0.99)
        imap = _mock_imap()
        audit = _mock_audit_log()
        queue = _mock_review_queue()

        result = await route_email_action(
            task,
            imap=imap,
            force_queue=True,
            audit_log=audit,
            review_queue=queue,
        )

        assert result is False
        queue.add.assert_called_once_with(task)
        audit.log_queued_for_review.assert_called_once_with(task)
        imap.delete.assert_not_called()

    async def test_force_queue_overrides_flag_in_digest(self):
        task = _make_task(gate_action=GateAction.FLAG_IN_DIGEST)
        imap = _mock_imap()
        audit = _mock_audit_log()
        queue = _mock_review_queue()

        result = await route_email_action(
            task,
            imap=imap,
            force_queue=True,
            audit_log=audit,
            review_queue=queue,
        )

        assert result is False
        queue.add.assert_called_once_with(task)
        audit.log_queued_for_review.assert_called_once_with(task)


class TestRouteGateActions:
    """Verify behaviour for each GateAction when force_queue=False."""

    async def test_auto_execute_applies_action_and_logs(self):
        task = _make_task(gate_action=GateAction.AUTO_EXECUTE)
        imap = _mock_imap()
        audit = _mock_audit_log()
        queue = _mock_review_queue()

        result = await route_email_action(
            task,
            imap=imap,
            force_queue=False,
            audit_log=audit,
            review_queue=queue,
        )

        assert result is True
        imap.delete.assert_awaited_once_with(["123"])
        audit.log_auto_applied.assert_called_once_with(task)
        queue.add.assert_not_called()

    async def test_queue_for_review_queues_and_logs(self):
        task = _make_task(gate_action=GateAction.QUEUE_FOR_REVIEW)
        imap = _mock_imap()
        audit = _mock_audit_log()
        queue = _mock_review_queue()

        result = await route_email_action(
            task,
            imap=imap,
            force_queue=False,
            audit_log=audit,
            review_queue=queue,
        )

        assert result is False
        queue.add.assert_called_once_with(task)
        audit.log_queued_for_review.assert_called_once_with(task)
        imap.delete.assert_not_called()

    async def test_flag_in_digest_executes_and_logs(self):
        task = _make_task(gate_action=GateAction.FLAG_IN_DIGEST)
        imap = _mock_imap()
        audit = _mock_audit_log()
        queue = _mock_review_queue()

        result = await route_email_action(
            task,
            imap=imap,
            force_queue=False,
            audit_log=audit,
            review_queue=queue,
        )

        assert result is True
        imap.delete.assert_awaited_once_with(["123"])
        audit.log_auto_applied.assert_called_once_with(task)
        queue.add.assert_not_called()


# --- execute_email_action tests ---


class TestExecuteEmailAction:
    """Verify each EmailActionType dispatches to the correct imap method."""

    async def test_delete_calls_imap_delete(self):
        task = _make_task(action_type=EmailActionType.DELETE)
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.delete.assert_awaited_once_with(["123"])
        imap.move.assert_not_called()
        imap.flag.assert_not_called()

    async def test_move_calls_imap_move(self):
        task = _make_task(
            action_type=EmailActionType.MOVE,
            target_folder="Corvus/Receipts",
        )
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.move.assert_awaited_once_with(["123"], "Corvus/Receipts")
        imap.delete.assert_not_called()

    async def test_move_without_folder_skips(self):
        """MOVE with no target_folder logs a warning and does nothing."""
        task = _make_task(
            action_type=EmailActionType.MOVE,
            target_folder=None,
        )
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.move.assert_not_called()
        imap.delete.assert_not_called()

    async def test_flag_calls_imap_flag(self):
        task = _make_task(
            action_type=EmailActionType.FLAG,
            flag_name="\\Important",
        )
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.flag.assert_awaited_once_with(["123"], "\\Important")

    async def test_flag_defaults_to_flagged(self):
        """FLAG with no flag_name should default to \\Flagged."""
        task = _make_task(
            action_type=EmailActionType.FLAG,
            flag_name=None,
        )
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.flag.assert_awaited_once_with(["123"], "\\Flagged")

    async def test_keep_does_nothing(self):
        task = _make_task(action_type=EmailActionType.KEEP)
        imap = _mock_imap()

        await execute_email_action(task, imap=imap)

        imap.delete.assert_not_called()
        imap.move.assert_not_called()
        imap.flag.assert_not_called()
        imap.mark_read.assert_not_called()
