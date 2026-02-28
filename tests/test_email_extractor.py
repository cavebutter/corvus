"""Tests for corvus.executors.email_extractor — data extraction from classified emails."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvus.executors.email_extractor import EXTRACTABLE_CATEGORIES, extract_email_data
from corvus.schemas.email import (
    ActionItem,
    EmailCategory,
    EmailClassification,
    EmailExtractionResult,
    EmailMessage,
    InvoiceData,
)


# --- Helpers ---


def _make_email(**overrides) -> EmailMessage:
    defaults = dict(
        uid="200",
        account_email="user@example.com",
        from_address="billing@vendor.com",
        from_name="Vendor Billing",
        to=["user@example.com"],
        subject="Invoice #12345",
        date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        body_text="Your invoice total is $85.50. Due by 2024-07-01.",
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


def _make_classification(
    category: EmailCategory = EmailCategory.INVOICE,
    confidence: float = 0.9,
) -> EmailClassification:
    return EmailClassification(
        category=category,
        confidence=confidence,
        reasoning="Test reasoning",
        suggested_action="move_to_receipts",
        summary="Invoice from vendor",
    )


def _make_mock_ollama(result: EmailExtractionResult) -> AsyncMock:
    """Create a mock OllamaClient that returns the given extraction result."""
    mock_ollama = AsyncMock()
    mock_raw = MagicMock()
    mock_raw.done = True
    mock_ollama.generate_structured = AsyncMock(return_value=(result, mock_raw))
    return mock_ollama


# --- EXTRACTABLE_CATEGORIES ---


class TestExtractableCategories:
    def test_contains_expected_categories(self):
        assert EmailCategory.RECEIPT in EXTRACTABLE_CATEGORIES
        assert EmailCategory.INVOICE in EXTRACTABLE_CATEGORIES
        assert EmailCategory.ACTION_REQUIRED in EXTRACTABLE_CATEGORIES

    def test_does_not_contain_non_extractable(self):
        assert EmailCategory.SPAM not in EXTRACTABLE_CATEGORIES
        assert EmailCategory.NEWSLETTER not in EXTRACTABLE_CATEGORIES
        assert EmailCategory.PERSONAL not in EXTRACTABLE_CATEGORIES
        assert EmailCategory.IMPORTANT not in EXTRACTABLE_CATEGORIES
        assert EmailCategory.OTHER not in EXTRACTABLE_CATEGORIES
        assert EmailCategory.PACKAGE_NOTICE not in EXTRACTABLE_CATEGORIES

    def test_count(self):
        assert len(EXTRACTABLE_CATEGORIES) == 3


# --- extract_email_data ---


class TestExtractEmailData:
    async def test_returns_extraction_with_invoice(self):
        email = _make_email()
        classification = _make_classification(category=EmailCategory.INVOICE)
        extraction = EmailExtractionResult(
            invoice=InvoiceData(
                vendor="Vendor Corp",
                amount=85.50,
                currency="USD",
                due_date="2024-07-01",
                invoice_number="12345",
                confidence=0.92,
            ),
            key_dates=["2024-07-01"],
        )
        mock_ollama = _make_mock_ollama(extraction)

        result, raw = await extract_email_data(
            email, classification, ollama=mock_ollama, model="test-model",
        )

        assert result.invoice is not None
        assert result.invoice.vendor == "Vendor Corp"
        assert result.invoice.amount == 85.50
        assert result.invoice.due_date == "2024-07-01"
        assert "2024-07-01" in result.key_dates

    async def test_returns_action_items_for_action_required(self):
        email = _make_email(
            subject="Action needed: review contract",
            body_text="Please review and sign the contract by Friday.",
        )
        classification = _make_classification(category=EmailCategory.ACTION_REQUIRED)
        extraction = EmailExtractionResult(
            action_items=[
                ActionItem(
                    description="Review and sign the contract",
                    deadline="2024-06-07",
                    priority="high",
                ),
            ],
        )
        mock_ollama = _make_mock_ollama(extraction)

        result, _ = await extract_email_data(
            email, classification, ollama=mock_ollama, model="test-model",
        )

        assert len(result.action_items) == 1
        assert result.action_items[0].description == "Review and sign the contract"
        assert result.action_items[0].priority == "high"

    async def test_raises_for_non_extractable_category(self):
        email = _make_email()
        classification = _make_classification(category=EmailCategory.SPAM)

        mock_ollama = AsyncMock()

        with pytest.raises(ValueError, match="not applicable"):
            await extract_email_data(
                email, classification, ollama=mock_ollama, model="test-model",
            )

        # The LLM should never have been called
        mock_ollama.generate_structured.assert_not_called()

    async def test_receipt_category_is_extractable(self):
        email = _make_email(subject="Your Amazon receipt")
        classification = _make_classification(category=EmailCategory.RECEIPT)
        extraction = EmailExtractionResult(
            invoice=InvoiceData(vendor="Amazon", amount=29.99, confidence=0.88),
        )
        mock_ollama = _make_mock_ollama(extraction)

        result, _ = await extract_email_data(
            email, classification, ollama=mock_ollama, model="test-model",
        )

        assert result.invoice is not None
        assert result.invoice.vendor == "Amazon"
        mock_ollama.generate_structured.assert_awaited_once()
