"""Tests for the email triage and summary pipelines (corvus/orchestrator/email_pipelines.py).

All external dependencies (IMAP, Ollama, classifier, extractor, router) are
mocked so that tests run without a mail server or LLM.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvus.orchestrator.email_pipelines import run_email_summary, run_email_triage
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    ActionItem,
    EmailAccountConfig,
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailExtractionResult,
    EmailMessage,
    EmailSummaryResult,
    EmailTriageResult,
    EmailTriageTask,
    InvoiceData,
)


# --- Fixtures ---


@pytest.fixture
def account_config():
    return EmailAccountConfig(
        name="Test Account",
        server="imap.test.com",
        email="test@test.com",
        password="secret",
        folders={
            "inbox": "INBOX",
            "processed": "Corvus/Processed",
            "receipts": "Corvus/Receipts",
        },
    )


def _make_email(uid: str = "100", subject: str = "Test Email", from_addr: str = "sender@example.com") -> EmailMessage:
    """Build a minimal EmailMessage for testing."""
    return EmailMessage(
        uid=uid,
        account_email="test@test.com",
        from_address=from_addr,
        subject=subject,
        date=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        body_text="Hello, this is a test email.",
    )


def _make_triage_task(
    uid: str = "100",
    category: EmailCategory = EmailCategory.SPAM,
    action_type: EmailActionType = EmailActionType.DELETE,
    confidence: float = 0.92,
) -> EmailTriageTask:
    """Build an EmailTriageTask as the classifier would return."""
    return EmailTriageTask(
        uid=uid,
        account_email="test@test.com",
        subject="Test Email",
        from_address="sender@example.com",
        classification=EmailClassification(
            category=category,
            confidence=confidence,
            reasoning="Automated classification",
            suggested_action="delete",
            summary="A test email summary",
        ),
        proposed_action=EmailAction(action_type=action_type),
        overall_confidence=confidence,
        gate_action=GateAction.AUTO_EXECUTE,
    )


# --- Mock helpers ---

# The pipeline functions use local imports (inside the function body), so
# patches must target the *source* module rather than the pipeline module.
_CLASSIFY_PATH = "corvus.executors.email_classifier.classify_email"
_EXTRACT_PATH = "corvus.executors.email_extractor.extract_email_data"
_EXTRACTABLE_PATH = "corvus.executors.email_extractor.EXTRACTABLE_CATEGORIES"
_ROUTE_PATH = "corvus.router.email.route_email_action"
_EXECUTE_PATH = "corvus.router.email.execute_email_action"
_IMAP_PATH = "corvus.orchestrator.email_pipelines.ImapClient"


def _patch_imap_client(messages: list[EmailMessage] | None = None):
    """Return a patch that replaces ImapClient with an async-context-manager mock.

    The mock's ``fetch_messages`` returns *messages* (default: empty list).
    """
    mock_imap = AsyncMock()
    mock_imap.fetch_messages = AsyncMock(return_value=messages or [])
    mock_imap.ensure_folders = AsyncMock()

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_imap)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    return patch(_IMAP_PATH, mock_cls), mock_imap


# --- run_email_triage tests ---


class TestRunEmailTriageEmpty:
    """Empty inbox returns zero counts."""

    async def test_empty_inbox(self, account_config, tmp_path):
        imap_patch, _mock_imap = _patch_imap_client(messages=[])

        with imap_patch:
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
            )

        assert isinstance(result, EmailTriageResult)
        assert result.processed == 0
        assert result.auto_acted == 0
        assert result.queued == 0
        assert result.errors == 0
        assert result.categories == {}


class TestRunEmailTriageProcessing:
    """Triage processes messages and returns correct counts."""

    async def test_processes_messages_and_counts(self, account_config, tmp_path):
        emails = [_make_email(uid="1"), _make_email(uid="2")]
        task1 = _make_triage_task(uid="1")
        task2 = _make_triage_task(uid="2", category=EmailCategory.NEWSLETTER)

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(side_effect=[
                    (task1, MagicMock()),
                    (task2, MagicMock()),
                ]),
            ),
            patch(
                _ROUTE_PATH,
                new=AsyncMock(return_value=False),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                force_queue=True,
            )

        assert result.processed == 2
        assert result.queued == 2
        assert "spam" in result.categories
        assert "newsletter" in result.categories

    async def test_handles_errors_gracefully(self, account_config, tmp_path):
        """If classify_email raises, errors count should increment."""
        emails = [_make_email(uid="1"), _make_email(uid="2")]

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(side_effect=RuntimeError("LLM fail")),
            ),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
            )

        assert result.errors == 2
        assert result.processed == 0

    async def test_calls_extractor_for_receipt(self, account_config, tmp_path):
        """Emails classified as RECEIPT trigger data extraction."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1", category=EmailCategory.RECEIPT)

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        extraction_result = EmailExtractionResult(
            invoice=InvoiceData(vendor="TestCo", amount=42.50, confidence=0.9),
        )

        mock_extract = AsyncMock(return_value=(extraction_result, MagicMock()))

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _EXTRACT_PATH,
                new=mock_extract,
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset({EmailCategory.RECEIPT, EmailCategory.INVOICE, EmailCategory.ACTION_REQUIRED}),
            ),
            patch(
                _ROUTE_PATH,
                new=AsyncMock(return_value=False),
            ),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
            )

        assert result.processed == 1
        mock_extract.assert_awaited_once()

    async def test_force_queue_queues_all(self, account_config, tmp_path):
        """force_queue=True causes route to return False (queued)."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1")

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        mock_route = AsyncMock(return_value=False)

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _ROUTE_PATH,
                new=mock_route,
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                force_queue=True,
            )

        assert result.queued == 1
        assert result.auto_acted == 0
        # Verify force_queue was forwarded
        call_kwargs = mock_route.call_args.kwargs
        assert call_kwargs["force_queue"] is True

    async def test_on_progress_callback(self, account_config, tmp_path):
        """The on_progress callback is invoked during processing."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1")

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)
        progress_calls: list[str] = []

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _ROUTE_PATH,
                new=AsyncMock(return_value=False),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                on_progress=progress_calls.append,
            )

        assert len(progress_calls) > 0
        # Should include fetching and done messages
        assert any("Fetching" in msg for msg in progress_calls)
        assert any("Done" in msg for msg in progress_calls)


# --- run_email_summary tests ---


class TestRunEmailSummaryEmpty:
    """Empty inbox returns a 'no unread' summary."""

    async def test_empty_inbox(self, account_config):
        imap_patch, _mock_imap = _patch_imap_client(messages=[])

        with imap_patch:
            result = await run_email_summary(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
            )

        assert isinstance(result, EmailSummaryResult)
        assert "No unread" in result.summary or "no unread" in result.summary.lower()
        assert result.total_unread == 0


class TestRunEmailSummaryProcessing:
    """Summary classifies emails and generates a summary."""

    async def test_classifies_and_summarizes(self, account_config):
        emails = [_make_email(uid="1"), _make_email(uid="2")]
        task1 = _make_triage_task(uid="1", category=EmailCategory.IMPORTANT)
        task2 = _make_triage_task(uid="2", category=EmailCategory.NEWSLETTER)

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Here is your inbox summary.", MagicMock()))

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(side_effect=[
                    (task1, MagicMock()),
                    (task2, MagicMock()),
                ]),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            result = await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
            )

        assert result.total_unread == 2
        assert "important" in result.by_category
        assert "newsletter" in result.by_category
        assert result.summary == "Here is your inbox summary."
        # Important emails tracked
        assert len(result.important_subjects) == 1

    async def test_extracts_action_items(self, account_config):
        """ACTION_REQUIRED emails trigger extraction, and action_items appear in the result."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1", category=EmailCategory.ACTION_REQUIRED)

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        extraction = EmailExtractionResult(
            action_items=[ActionItem(description="Reply to Bob", deadline="2025-07-01")],
        )

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary text.", MagicMock()))

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _EXTRACT_PATH,
                new=AsyncMock(return_value=(extraction, MagicMock())),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset({EmailCategory.ACTION_REQUIRED}),
            ),
        ):
            result = await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
            )

        assert len(result.action_items) == 1
        assert result.action_items[0].description == "Reply to Bob"
        assert result.action_items[0].deadline == "2025-07-01"

    async def test_on_progress_callback(self, account_config):
        """The on_progress callback is invoked during summarization."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1", category=EmailCategory.OTHER)

        imap_patch, _mock_imap = _patch_imap_client(messages=emails)

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary.", MagicMock()))

        progress_calls: list[str] = []

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
                on_progress=progress_calls.append,
            )

        assert len(progress_calls) > 0
        assert any("Fetching" in msg for msg in progress_calls)

    async def test_account_config_folders_used(self, account_config):
        """The pipeline uses the account_config folders to select the inbox."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1", category=EmailCategory.OTHER)

        imap_patch, mock_imap = _patch_imap_client(messages=emails)

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary.", MagicMock()))

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _EXTRACTABLE_PATH,
                new=frozenset(),
            ),
        ):
            await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
            )

        # ImapClient.fetch_messages should have been called with the inbox folder
        mock_imap.fetch_messages.assert_awaited_once()
        call_args = mock_imap.fetch_messages.call_args
        assert call_args[0][0] == "INBOX"


# --- Sender list integration tests ---


def _write_sender_lists(tmp_path, data: dict) -> str:
    """Write a sender_lists.json to tmp_path and return the path string."""
    import json

    path = tmp_path / "sender_lists.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestTriageSenderListBlacklist:
    """Blacklisted senders skip LLM and are auto-deleted."""

    async def test_blacklisted_sender_skips_llm(self, account_config, tmp_path):
        blacklisted_email = _make_email(uid="1", from_addr="spam@scammer.com")
        emails = [blacklisted_email]

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "black": {"action": "delete", "addresses": ["spam@scammer.com"]},
            },
            "priority": ["black"],
        })

        imap_patch, mock_imap = _patch_imap_client(messages=emails)
        mock_classify = AsyncMock()
        mock_execute = AsyncMock()

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=mock_classify),
            patch(_EXECUTE_PATH, new=mock_execute),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                sender_lists_path=sender_lists_path,
                force_queue=True,
            )

        # LLM should NOT have been called
        mock_classify.assert_not_awaited()
        # Action should have been executed
        mock_execute.assert_awaited_once()
        assert result.processed == 1
        assert result.auto_acted == 1
        assert result.queued == 0

    async def test_blacklisted_sender_bypasses_force_queue(self, account_config, tmp_path):
        """Sender list matches bypass force_queue — they're explicit user rules."""
        emails = [_make_email(uid="1", from_addr="spam@scammer.com")]

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "black": {"action": "delete", "addresses": ["spam@scammer.com"]},
            },
            "priority": ["black"],
        })

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_execute = AsyncMock()

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=AsyncMock()),
            patch(_EXECUTE_PATH, new=mock_execute),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                sender_lists_path=sender_lists_path,
                force_queue=True,  # Even with force_queue, sender list acts
            )

        assert result.auto_acted == 1
        assert result.queued == 0


class TestTriageSenderListVendor:
    """Vendor senders skip LLM and are auto-moved."""

    async def test_vendor_sender_auto_moved(self, account_config, tmp_path):
        emails = [_make_email(uid="1", from_addr="deals@amazon.com")]

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "vendor": {
                    "action": "move",
                    "folder_key": "approved_ads",
                    "cleanup_days": 14,
                    "addresses": ["deals@amazon.com"],
                },
            },
            "priority": ["vendor"],
        })

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_classify = AsyncMock()
        mock_execute = AsyncMock()

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=mock_classify),
            patch(_EXECUTE_PATH, new=mock_execute),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                sender_lists_path=sender_lists_path,
            )

        mock_classify.assert_not_awaited()
        mock_execute.assert_awaited_once()
        assert result.auto_acted == 1


class TestTriageSenderListWhitelist:
    """Whitelisted senders still get classified but force KEEP + queue."""

    async def test_whitelisted_sender_still_classified(self, account_config, tmp_path):
        emails = [_make_email(uid="1", from_addr="alice@example.com")]
        task = _make_triage_task(uid="1", category=EmailCategory.PERSONAL)

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "white": {"action": "keep", "addresses": ["alice@example.com"]},
            },
            "priority": ["white"],
        })

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_classify = AsyncMock(return_value=(task, MagicMock()))
        mock_route = AsyncMock(return_value=False)

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=mock_classify),
            patch(_ROUTE_PATH, new=mock_route),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                sender_lists_path=sender_lists_path,
                force_queue=True,
            )

        # LLM SHOULD have been called (white list still classifies)
        mock_classify.assert_awaited_once()
        assert result.processed == 1
        assert result.queued == 1

    async def test_whitelisted_sender_forces_keep(self, account_config, tmp_path):
        """Even if LLM suggests DELETE, whitelist forces KEEP."""
        emails = [_make_email(uid="1", from_addr="alice@example.com")]
        task = _make_triage_task(
            uid="1",
            category=EmailCategory.SPAM,
            action_type=EmailActionType.DELETE,
        )

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "white": {"action": "keep", "addresses": ["alice@example.com"]},
            },
            "priority": ["white"],
        })

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_classify = AsyncMock(return_value=(task, MagicMock()))
        mock_route = AsyncMock(return_value=False)

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=mock_classify),
            patch(_ROUTE_PATH, new=mock_route),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                sender_lists_path=sender_lists_path,
            )

        # Verify the task passed to route has KEEP action
        routed_task = mock_route.call_args.args[0]
        assert routed_task.proposed_action.action_type == EmailActionType.KEEP
        assert routed_task.sender_list == "white"


class TestTriageNoSenderLists:
    """Without sender_lists_path, triage works as before."""

    async def test_no_sender_lists_falls_through(self, account_config, tmp_path):
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1")

        imap_patch, _mock = _patch_imap_client(messages=emails)

        with (
            imap_patch,
            patch(
                _CLASSIFY_PATH,
                new=AsyncMock(return_value=(task, MagicMock())),
            ),
            patch(
                _ROUTE_PATH,
                new=AsyncMock(return_value=False),
            ),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=str(tmp_path / "audit.jsonl"),
                # No sender_lists_path
            )

        assert result.processed == 1


class TestSummarySenderListWhitelist:
    """Whitelisted senders always appear in important_subjects and get extraction."""

    async def test_whitelisted_sender_in_important_subjects(self, account_config, tmp_path):
        """Whitelisted sender always added to important_subjects, even if category is OTHER."""
        emails = [_make_email(uid="1", from_addr="alice@example.com")]
        # Category OTHER would normally NOT be in important_subjects
        task = _make_triage_task(uid="1", category=EmailCategory.OTHER)

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "white": {"action": "keep", "addresses": ["alice@example.com"]},
            },
            "priority": ["white"],
        })

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary.", MagicMock()))

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=AsyncMock(return_value=(task, MagicMock()))),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
                sender_lists_path=sender_lists_path,
            )

        assert len(result.important_subjects) == 1
        assert result.important_subjects[0] == "Test Email"

    async def test_whitelisted_sender_gets_extraction(self, account_config, tmp_path):
        """Whitelisted sender always triggers extraction, even if category isn't extractable."""
        emails = [_make_email(uid="1", from_addr="alice@example.com")]
        task = _make_triage_task(uid="1", category=EmailCategory.OTHER)

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "white": {"action": "keep", "addresses": ["alice@example.com"]},
            },
            "priority": ["white"],
        })

        extraction = EmailExtractionResult(
            action_items=[ActionItem(description="Call Alice back")],
        )

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary.", MagicMock()))
        mock_extract = AsyncMock(return_value=(extraction, MagicMock()))

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=AsyncMock(return_value=(task, MagicMock()))),
            patch(_EXTRACT_PATH, new=mock_extract),
            patch(_EXTRACTABLE_PATH, new=frozenset()),  # empty — normally no extraction
        ):
            result = await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
                sender_lists_path=sender_lists_path,
            )

        mock_extract.assert_awaited_once()
        assert len(result.action_items) == 1
        assert result.action_items[0].description == "Call Alice back"

    async def test_no_sender_lists_summary_unchanged(self, account_config):
        """Without sender_lists_path, summary behaves as before."""
        emails = [_make_email(uid="1")]
        task = _make_triage_task(uid="1", category=EmailCategory.OTHER)

        imap_patch, _mock = _patch_imap_client(messages=emails)
        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=("Summary.", MagicMock()))

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=AsyncMock(return_value=(task, MagicMock()))),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            result = await run_email_summary(
                account_config=account_config,
                ollama=mock_ollama,
                model="test-model",
                # No sender_lists_path
            )

        # OTHER category should NOT be in important_subjects
        assert len(result.important_subjects) == 0


class TestTriageSenderListAudit:
    """Sender list actions are logged with sender_list_applied audit action."""

    async def test_sender_list_audit_entry(self, account_config, tmp_path):
        from corvus.schemas.email import EmailAuditEntry

        emails = [_make_email(uid="1", from_addr="spam@scammer.com")]

        sender_lists_path = _write_sender_lists(tmp_path, {
            "lists": {
                "black": {"action": "delete", "addresses": ["spam@scammer.com"]},
            },
            "priority": ["black"],
        })

        audit_path = str(tmp_path / "audit.jsonl")
        imap_patch, _mock = _patch_imap_client(messages=emails)

        with (
            imap_patch,
            patch(_CLASSIFY_PATH, new=AsyncMock()),
            patch(_EXECUTE_PATH, new=AsyncMock()),
            patch(_EXTRACTABLE_PATH, new=frozenset()),
        ):
            await run_email_triage(
                account_config=account_config,
                ollama=AsyncMock(),
                model="test-model",
                review_db_path=str(tmp_path / "review.db"),
                audit_log_path=audit_path,
                sender_lists_path=sender_lists_path,
            )

        # Read the audit log and verify the entry
        from corvus.audit.email_logger import EmailAuditLog

        audit_log = EmailAuditLog(audit_path)
        entries = audit_log.read_entries()
        assert len(entries) == 1
        assert entries[0].action == "sender_list_applied"
        assert entries[0].applied is True
