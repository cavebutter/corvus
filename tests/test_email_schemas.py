"""Tests for corvus.schemas.email — validation, defaults, enums, constraints."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from corvus.schemas.document_tagging import GateAction, ReviewStatus
from corvus.schemas.email import (
    ActionItem,
    EmailAccountConfig,
    EmailAction,
    EmailActionType,
    EmailAuditEntry,
    EmailCategory,
    EmailClassification,
    EmailEnvelope,
    EmailExtractionResult,
    EmailMessage,
    EmailReviewQueueItem,
    EmailSummaryResult,
    EmailTriageResult,
    EmailTriageTask,
    InvoiceData,
)


# --- EmailAccountConfig ---


class TestEmailAccountConfig:
    def test_required_fields(self):
        config = EmailAccountConfig(
            name="personal",
            server="imap.example.com",
            email="user@example.com",
            password="secret",
            folders={"inbox": "INBOX"},
        )
        assert config.name == "personal"
        assert config.server == "imap.example.com"
        assert config.email == "user@example.com"
        assert config.password == "secret"
        assert config.folders == {"inbox": "INBOX"}

    def test_defaults(self):
        config = EmailAccountConfig(
            name="test",
            server="imap.test.com",
            email="test@test.com",
            password="pw",
            folders={},
        )
        assert config.port == 993
        assert config.ssl is True
        assert config.is_gmail is False

    def test_gmail_account(self):
        config = EmailAccountConfig(
            name="gmail",
            server="imap.gmail.com",
            email="me@gmail.com",
            password="app-password",
            folders={"inbox": "INBOX", "receipts": "[Gmail]/Receipts"},
            is_gmail=True,
            port=993,
        )
        assert config.is_gmail is True

    def test_custom_port_and_no_ssl(self):
        config = EmailAccountConfig(
            name="local",
            server="localhost",
            email="local@test.com",
            password="pw",
            folders={},
            port=143,
            ssl=False,
        )
        assert config.port == 143
        assert config.ssl is False


# --- EmailEnvelope ---


class TestEmailEnvelope:
    def test_required_fields(self):
        env = EmailEnvelope(
            uid="100",
            account_email="user@example.com",
            from_address="sender@example.com",
            subject="Hello",
            date=datetime(2024, 6, 15, tzinfo=timezone.utc),
        )
        assert env.uid == "100"
        assert env.account_email == "user@example.com"
        assert env.from_address == "sender@example.com"
        assert env.subject == "Hello"

    def test_defaults(self):
        env = EmailEnvelope(
            uid="1",
            account_email="a@b.com",
            from_address="x@y.com",
            subject="Test",
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert env.from_name == ""
        assert env.to == []
        assert env.flags == []
        assert env.size_bytes == 0

    def test_full_construction(self):
        env = EmailEnvelope(
            uid="200",
            account_email="user@example.com",
            from_address="sender@corp.com",
            from_name="Alice Sender",
            to=["user@example.com", "other@example.com"],
            subject="Meeting",
            date=datetime(2024, 7, 1, 10, 30, tzinfo=timezone.utc),
            flags=["\\Seen", "\\Flagged"],
            size_bytes=4096,
        )
        assert env.from_name == "Alice Sender"
        assert len(env.to) == 2
        assert env.size_bytes == 4096
        assert "\\Seen" in env.flags


# --- EmailMessage ---


class TestEmailMessage:
    def test_inherits_from_envelope(self):
        msg = EmailMessage(
            uid="300",
            account_email="user@example.com",
            from_address="sender@example.com",
            subject="Invoice",
            date=datetime(2024, 8, 1, tzinfo=timezone.utc),
        )
        # Inherited fields work
        assert msg.uid == "300"
        assert msg.subject == "Invoice"

    def test_body_defaults(self):
        msg = EmailMessage(
            uid="1",
            account_email="a@b.com",
            from_address="x@y.com",
            subject="Test",
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert msg.body_text == ""
        assert msg.body_html == ""
        assert msg.has_attachments is False
        assert msg.attachment_names == []

    def test_full_construction(self):
        msg = EmailMessage(
            uid="400",
            account_email="user@example.com",
            from_address="billing@vendor.com",
            subject="Your receipt",
            date=datetime(2024, 9, 1, tzinfo=timezone.utc),
            body_text="Thank you for your purchase.",
            body_html="<p>Thank you for your purchase.</p>",
            has_attachments=True,
            attachment_names=["receipt.pdf"],
        )
        assert msg.body_text == "Thank you for your purchase."
        assert msg.has_attachments is True
        assert msg.attachment_names == ["receipt.pdf"]


# --- EmailCategory ---


class TestEmailCategory:
    def test_all_expected_values(self):
        expected = {
            "spam", "newsletter", "job_alert", "receipt", "invoice",
            "package_notice", "action_required", "personal", "important", "other",
        }
        actual = {c.value for c in EmailCategory}
        assert actual == expected

    def test_count(self):
        assert len(EmailCategory) == 10


# --- EmailClassification ---


class TestEmailClassification:
    def test_valid_classification(self):
        cls = EmailClassification(
            category=EmailCategory.SPAM,
            confidence=0.95,
            reasoning="Clearly spam",
            suggested_action="delete",
            is_automated=True,
            summary="Spam email",
        )
        assert cls.category == EmailCategory.SPAM
        assert cls.confidence == 0.95
        assert cls.is_automated is True

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            EmailClassification(
                category=EmailCategory.OTHER,
                confidence=1.5,
                reasoning="test",
                suggested_action="keep",
            )
        with pytest.raises(ValidationError):
            EmailClassification(
                category=EmailCategory.OTHER,
                confidence=-0.1,
                reasoning="test",
                suggested_action="keep",
            )

    def test_defaults(self):
        cls = EmailClassification(
            category=EmailCategory.PERSONAL,
            confidence=0.8,
            reasoning="Looks personal",
            suggested_action="keep",
        )
        assert cls.is_automated is False
        assert cls.summary == ""

    def test_confidence_edge_values(self):
        """Boundary values 0.0 and 1.0 should be valid."""
        low = EmailClassification(
            category=EmailCategory.OTHER,
            confidence=0.0,
            reasoning="uncertain",
            suggested_action="keep",
        )
        high = EmailClassification(
            category=EmailCategory.SPAM,
            confidence=1.0,
            reasoning="certain",
            suggested_action="delete",
        )
        assert low.confidence == 0.0
        assert high.confidence == 1.0


# --- EmailActionType ---


class TestEmailActionType:
    def test_all_expected_values(self):
        expected = {"move", "delete", "flag", "mark_read", "keep"}
        actual = {a.value for a in EmailActionType}
        assert actual == expected


# --- EmailAction ---


class TestEmailAction:
    def test_move_action(self):
        action = EmailAction(
            action_type=EmailActionType.MOVE,
            target_folder="Receipts",
        )
        assert action.action_type == EmailActionType.MOVE
        assert action.target_folder == "Receipts"
        assert action.flag_name is None

    def test_flag_action(self):
        action = EmailAction(
            action_type=EmailActionType.FLAG,
            flag_name="\\Flagged",
        )
        assert action.flag_name == "\\Flagged"
        assert action.target_folder is None

    def test_optional_fields_default_to_none(self):
        action = EmailAction(action_type=EmailActionType.DELETE)
        assert action.target_folder is None
        assert action.flag_name is None

    def test_keep_action(self):
        action = EmailAction(action_type=EmailActionType.KEEP)
        assert action.action_type == EmailActionType.KEEP


# --- EmailTriageTask ---


class TestEmailTriageTask:
    def _make_task(self, **overrides):
        classification = EmailClassification(
            category=EmailCategory.RECEIPT,
            confidence=0.85,
            reasoning="Looks like a receipt",
            suggested_action="move_to_receipts",
        )
        action = EmailAction(
            action_type=EmailActionType.MOVE,
            target_folder="Receipts",
        )
        defaults = dict(
            uid="500",
            account_email="user@example.com",
            subject="Your order",
            from_address="orders@shop.com",
            classification=classification,
            proposed_action=action,
            overall_confidence=0.85,
            gate_action=GateAction.FLAG_IN_DIGEST,
        )
        defaults.update(overrides)
        return EmailTriageTask(**defaults)

    def test_creation(self):
        task = self._make_task()
        assert task.task_type == "email_triage"
        assert task.uid == "500"
        assert task.classification.category == EmailCategory.RECEIPT
        assert task.gate_action == GateAction.FLAG_IN_DIGEST

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._make_task(overall_confidence=1.5)
        with pytest.raises(ValidationError):
            self._make_task(overall_confidence=-0.1)


# --- InvoiceData ---


class TestInvoiceData:
    def test_required_fields(self):
        invoice = InvoiceData(
            vendor="Acme Corp",
            confidence=0.9,
        )
        assert invoice.vendor == "Acme Corp"
        assert invoice.confidence == 0.9

    def test_defaults(self):
        invoice = InvoiceData(vendor="Test", confidence=0.5)
        assert invoice.amount is None
        assert invoice.currency == "USD"
        assert invoice.due_date is None
        assert invoice.invoice_number is None

    def test_full_construction(self):
        invoice = InvoiceData(
            vendor="AT&T",
            amount=85.50,
            currency="USD",
            due_date="2024-02-15",
            invoice_number="INV-12345",
            confidence=0.95,
        )
        assert invoice.amount == 85.50
        assert invoice.invoice_number == "INV-12345"


# --- ActionItem ---


class TestActionItem:
    def test_required_fields(self):
        item = ActionItem(description="Reply to Bob")
        assert item.description == "Reply to Bob"

    def test_defaults(self):
        item = ActionItem(description="Task")
        assert item.deadline is None
        assert item.priority == "normal"

    def test_full_construction(self):
        item = ActionItem(
            description="Submit report",
            deadline="2024-12-31",
            priority="high",
        )
        assert item.deadline == "2024-12-31"
        assert item.priority == "high"


# --- EmailExtractionResult ---


class TestEmailExtractionResult:
    def test_defaults(self):
        result = EmailExtractionResult()
        assert result.invoice is None
        assert result.action_items == []
        assert result.key_dates == []

    def test_with_invoice(self):
        invoice = InvoiceData(vendor="Test", confidence=0.8)
        result = EmailExtractionResult(invoice=invoice)
        assert result.invoice is not None
        assert result.invoice.vendor == "Test"

    def test_with_action_items(self):
        items = [ActionItem(description="Do this"), ActionItem(description="Do that")]
        result = EmailExtractionResult(action_items=items)
        assert len(result.action_items) == 2


# --- EmailTriageResult ---


class TestEmailTriageResult:
    def test_defaults(self):
        result = EmailTriageResult(account_email="user@example.com")
        assert result.processed == 0
        assert result.auto_acted == 0
        assert result.queued == 0
        assert result.errors == 0
        assert result.categories == {}

    def test_with_counts(self):
        result = EmailTriageResult(
            account_email="user@example.com",
            processed=10,
            auto_acted=7,
            queued=2,
            errors=1,
            categories={"spam": 3, "receipt": 4, "personal": 3},
        )
        assert result.processed == 10
        assert sum(result.categories.values()) == 10


# --- EmailSummaryResult ---


class TestEmailSummaryResult:
    def test_defaults(self):
        result = EmailSummaryResult(account_email="user@example.com")
        assert result.total_unread == 0
        assert result.summary == ""
        assert result.by_category == {}
        assert result.important_subjects == []
        assert result.action_items == []

    def test_with_data(self):
        result = EmailSummaryResult(
            account_email="user@example.com",
            total_unread=5,
            summary="5 new emails",
            by_category={"receipt": 2, "personal": 3},
            important_subjects=["Urgent: review needed"],
            action_items=[ActionItem(description="Review document")],
        )
        assert result.total_unread == 5
        assert len(result.action_items) == 1


# --- EmailReviewQueueItem ---


class TestEmailReviewQueueItem:
    def test_creation(self):
        classification = EmailClassification(
            category=EmailCategory.INVOICE,
            confidence=0.65,
            reasoning="Might be an invoice",
            suggested_action="move_to_receipts",
        )
        action = EmailAction(action_type=EmailActionType.MOVE, target_folder="Receipts")
        task = EmailTriageTask(
            uid="600",
            account_email="user@example.com",
            subject="Invoice #123",
            from_address="billing@vendor.com",
            classification=classification,
            proposed_action=action,
            overall_confidence=0.65,
            gate_action=GateAction.QUEUE_FOR_REVIEW,
        )
        now = datetime(2024, 10, 1, tzinfo=timezone.utc)
        item = EmailReviewQueueItem(id="q-001", created_at=now, task=task)

        assert item.id == "q-001"
        assert item.status == ReviewStatus.PENDING
        assert item.reviewed_at is None
        assert item.reviewer_notes is None
        assert item.task.uid == "600"


# --- EmailAuditEntry ---


class TestEmailAuditEntry:
    def test_creation(self):
        entry = EmailAuditEntry(
            timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc),
            action="auto_applied",
            account_email="user@example.com",
            uid="700",
            subject="Receipt from Amazon",
            from_address="noreply@amazon.com",
            category=EmailCategory.RECEIPT,
            email_action=EmailAction(
                action_type=EmailActionType.MOVE,
                target_folder="Receipts",
            ),
            gate_action=GateAction.AUTO_EXECUTE,
            applied=True,
        )
        assert entry.action == "auto_applied"
        assert entry.applied is True
        assert entry.category == EmailCategory.RECEIPT

    def test_queued_entry(self):
        entry = EmailAuditEntry(
            timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc),
            action="queued_for_review",
            account_email="user@example.com",
            uid="701",
            subject="Unknown sender",
            from_address="unknown@random.net",
            category=EmailCategory.OTHER,
            email_action=EmailAction(action_type=EmailActionType.KEEP),
            gate_action=GateAction.QUEUE_FOR_REVIEW,
            applied=False,
        )
        assert entry.action == "queued_for_review"
        assert entry.applied is False
