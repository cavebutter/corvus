"""Tests for the email CLI commands (corvus email ...).

Uses Click's CliRunner so no real IMAP/Ollama connections are required.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from corvus.cli import cli


# --- Fixtures ---


@pytest.fixture
def runner():
    return CliRunner()


SAMPLE_ACCOUNTS = [
    {
        "name": "Personal",
        "server": "imap.example.com",
        "email": "me@example.com",
        "password": "secret",
        "folders": {"inbox": "INBOX", "processed": "Corvus/Processed"},
    },
    {
        "name": "Work",
        "server": "imap.work.com",
        "email": "me@work.com",
        "password": "secret2",
        "is_gmail": True,
        "folders": {"inbox": "INBOX", "processed": "Corvus/Processed"},
    },
]


# --- corvus email accounts ---


class TestEmailAccounts:
    def test_no_config_shows_message(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: [])
        result = runner.invoke(cli, ["email", "accounts"])
        assert result.exit_code == 0
        assert "No email accounts configured" in result.output

    def test_with_accounts_shows_them(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: SAMPLE_ACCOUNTS)
        result = runner.invoke(cli, ["email", "accounts"])
        assert result.exit_code == 0
        assert "me@example.com" in result.output
        assert "me@work.com" in result.output
        assert "Personal" in result.output
        assert "Work" in result.output
        assert "[Gmail]" in result.output

    def test_shows_folder_info(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: SAMPLE_ACCOUNTS)
        result = runner.invoke(cli, ["email", "accounts"])
        assert "Folders:" in result.output


# --- corvus email status ---


class TestEmailStatus:
    def test_shows_statistics(self, runner):
        """email status should display pending/auto/queued/approved/rejected counts."""
        # EmailAuditLog and EmailReviewQueue are imported locally inside
        # the email_status command, so we patch at the source module level.
        mock_queue_instance = MagicMock()
        mock_queue_instance.count_pending.return_value = 3
        mock_queue_instance.__enter__ = MagicMock(return_value=mock_queue_instance)
        mock_queue_instance.__exit__ = MagicMock(return_value=False)

        mock_audit_instance = MagicMock()
        mock_audit_instance.read_entries.return_value = []

        with (
            patch(
                "corvus.audit.email_logger.EmailAuditLog",
                return_value=mock_audit_instance,
            ),
            patch(
                "corvus.queue.email_review.EmailReviewQueue",
                return_value=mock_queue_instance,
            ),
        ):
            result = runner.invoke(cli, ["email", "status"])

        assert result.exit_code == 0
        assert "Pending review:" in result.output
        assert "3" in result.output
        assert "Auto-applied:" in result.output


# --- corvus email triage ---


class TestEmailTriage:
    def test_no_config_exits_with_error(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: [])
        result = runner.invoke(cli, ["email", "triage"])
        assert result.exit_code != 0
        assert "No email accounts configured" in result.output

    def test_unknown_account_exits_with_error(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: SAMPLE_ACCOUNTS)
        result = runner.invoke(cli, ["email", "triage", "--account", "nonexistent@x.com"])
        assert result.exit_code != 0
        assert "No account found matching" in result.output


# --- corvus email summary ---


class TestEmailSummary:
    def test_no_config_exits_with_error(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: [])
        result = runner.invoke(cli, ["email", "summary"])
        assert result.exit_code != 0
        assert "No email accounts configured" in result.output


# --- corvus email review ---


class TestEmailReview:
    def test_no_pending_shows_message(self, runner):
        """When the review queue is empty, print 'No pending' and exit cleanly."""
        mock_queue_instance = MagicMock()
        mock_queue_instance.list_pending.return_value = []
        mock_queue_instance.__enter__ = MagicMock(return_value=mock_queue_instance)
        mock_queue_instance.__exit__ = MagicMock(return_value=False)

        with patch(
            "corvus.queue.email_review.EmailReviewQueue",
            return_value=mock_queue_instance,
        ):
            result = runner.invoke(cli, ["email", "review"])

        assert result.exit_code == 0
        assert "No pending" in result.output
