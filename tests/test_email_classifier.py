"""Tests for corvus.executors.email_classifier — classification and gate logic."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvus.executors.email_classifier import (
    _build_action,
    _determine_gate_action,
    classify_email,
)
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailMessage,
)


# --- Helpers ---


def _make_email(**overrides) -> EmailMessage:
    defaults = dict(
        uid="100",
        account_email="user@example.com",
        from_address="sender@example.com",
        from_name="Test Sender",
        to=["user@example.com"],
        subject="Test Email",
        date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        body_text="This is a test email body.",
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


def _make_classification(
    category: EmailCategory = EmailCategory.SPAM,
    confidence: float = 0.95,
    **overrides,
) -> EmailClassification:
    defaults = dict(
        category=category,
        confidence=confidence,
        reasoning="Test reasoning",
        suggested_action="delete",
        is_automated=True,
        summary="Test email summary",
    )
    defaults.update(overrides)
    return EmailClassification(**defaults)


def _make_mock_ollama(classification: EmailClassification) -> AsyncMock:
    """Create a mock OllamaClient that returns the given classification."""
    mock_ollama = AsyncMock()
    mock_raw = MagicMock()
    mock_raw.done = True
    mock_ollama.generate_structured = AsyncMock(return_value=(classification, mock_raw))
    return mock_ollama


# --- _determine_gate_action ---


class TestDetermineGateAction:
    def test_high_confidence_auto_executes(self):
        assert _determine_gate_action(0.95) == GateAction.AUTO_EXECUTE
        assert _determine_gate_action(0.9) == GateAction.AUTO_EXECUTE
        assert _determine_gate_action(1.0) == GateAction.AUTO_EXECUTE

    def test_medium_confidence_flagged(self):
        assert _determine_gate_action(0.85) == GateAction.FLAG_IN_DIGEST
        assert _determine_gate_action(0.7) == GateAction.FLAG_IN_DIGEST
        assert _determine_gate_action(0.89) == GateAction.FLAG_IN_DIGEST

    def test_low_confidence_queued_for_review(self):
        assert _determine_gate_action(0.69) == GateAction.QUEUE_FOR_REVIEW
        assert _determine_gate_action(0.5) == GateAction.QUEUE_FOR_REVIEW
        assert _determine_gate_action(0.0) == GateAction.QUEUE_FOR_REVIEW


# --- _build_action ---


class TestBuildAction:
    def test_spam_maps_to_delete(self):
        folders = {"receipts": "Corvus/Receipts", "processed": "Corvus/Processed"}
        action = _build_action(EmailCategory.SPAM, folders)
        assert action.action_type == EmailActionType.DELETE
        assert action.target_folder is None

    def test_receipt_maps_to_move_receipts(self):
        folders = {"receipts": "Corvus/Receipts", "processed": "Corvus/Processed"}
        action = _build_action(EmailCategory.RECEIPT, folders)
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder == "Corvus/Receipts"

    def test_invoice_maps_to_move_receipts(self):
        folders = {"receipts": "Corvus/Receipts", "processed": "Corvus/Processed"}
        action = _build_action(EmailCategory.INVOICE, folders)
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder == "Corvus/Receipts"

    def test_newsletter_maps_to_move_processed(self):
        folders = {"receipts": "Corvus/Receipts", "processed": "Corvus/Processed"}
        action = _build_action(EmailCategory.NEWSLETTER, folders)
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder == "Corvus/Processed"

    def test_package_notice_maps_to_move_processed(self):
        folders = {"processed": "Corvus/Processed"}
        action = _build_action(EmailCategory.PACKAGE_NOTICE, folders)
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder == "Corvus/Processed"

    def test_action_required_maps_to_flag(self):
        action = _build_action(EmailCategory.ACTION_REQUIRED, {})
        assert action.action_type == EmailActionType.FLAG
        assert action.flag_name == "\\Flagged"

    def test_important_maps_to_flag(self):
        action = _build_action(EmailCategory.IMPORTANT, {})
        assert action.action_type == EmailActionType.FLAG
        assert action.flag_name == "\\Flagged"

    def test_personal_maps_to_keep(self):
        action = _build_action(EmailCategory.PERSONAL, {})
        assert action.action_type == EmailActionType.KEEP

    def test_other_maps_to_keep(self):
        action = _build_action(EmailCategory.OTHER, {})
        assert action.action_type == EmailActionType.KEEP

    def test_missing_folder_key_gives_none_target(self):
        """If the folder mapping doesn't have the needed key, target_folder is None."""
        action = _build_action(EmailCategory.RECEIPT, {})
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder is None


# --- classify_email ---


class TestClassifyEmail:
    async def test_returns_triage_task_with_correct_fields(self):
        email = _make_email(uid="42", subject="Win a prize!")
        classification = _make_classification(
            category=EmailCategory.SPAM,
            confidence=0.95,
        )
        mock_ollama = _make_mock_ollama(classification)
        folders = {"receipts": "Corvus/Receipts", "processed": "Corvus/Processed"}

        task, raw = await classify_email(
            email,
            ollama=mock_ollama,
            model="test-model",
            folders=folders,
        )

        assert task.uid == "42"
        assert task.account_email == "user@example.com"
        assert task.subject == "Win a prize!"
        assert task.from_address == "sender@example.com"
        assert task.classification.category == EmailCategory.SPAM
        assert task.overall_confidence == 0.95
        assert task.task_type == "email_triage"

    async def test_spam_classification_produces_delete_action(self):
        email = _make_email()
        classification = _make_classification(
            category=EmailCategory.SPAM, confidence=0.95,
        )
        mock_ollama = _make_mock_ollama(classification)

        task, _ = await classify_email(email, ollama=mock_ollama, model="m")

        assert task.proposed_action.action_type == EmailActionType.DELETE
        assert task.gate_action == GateAction.AUTO_EXECUTE

    async def test_receipt_classification_produces_move_action(self):
        email = _make_email()
        classification = _make_classification(
            category=EmailCategory.RECEIPT,
            confidence=0.85,
            suggested_action="move_to_receipts",
        )
        mock_ollama = _make_mock_ollama(classification)
        folders = {"receipts": "Corvus/Receipts"}

        task, _ = await classify_email(
            email, ollama=mock_ollama, model="m", folders=folders,
        )

        assert task.proposed_action.action_type == EmailActionType.MOVE
        assert task.proposed_action.target_folder == "Corvus/Receipts"
        assert task.gate_action == GateAction.FLAG_IN_DIGEST

    async def test_high_confidence_gets_auto_execute(self):
        email = _make_email()
        classification = _make_classification(confidence=0.92)
        mock_ollama = _make_mock_ollama(classification)

        task, _ = await classify_email(email, ollama=mock_ollama, model="m")
        assert task.gate_action == GateAction.AUTO_EXECUTE

    async def test_medium_confidence_gets_flag_in_digest(self):
        email = _make_email()
        classification = _make_classification(
            category=EmailCategory.NEWSLETTER, confidence=0.75,
        )
        mock_ollama = _make_mock_ollama(classification)

        task, _ = await classify_email(email, ollama=mock_ollama, model="m")
        assert task.gate_action == GateAction.FLAG_IN_DIGEST

    async def test_low_confidence_gets_queue_for_review(self):
        email = _make_email()
        classification = _make_classification(
            category=EmailCategory.OTHER, confidence=0.5,
        )
        mock_ollama = _make_mock_ollama(classification)

        task, _ = await classify_email(email, ollama=mock_ollama, model="m")
        assert task.gate_action == GateAction.QUEUE_FOR_REVIEW
