"""Tests for the email CLI commands (corvus email ...).

Uses Click's CliRunner so no real IMAP/Ollama connections are required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from corvus.cli import cli


# --- Fixtures ---


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sender_lists_path(tmp_path):
    """Write sample sender lists to a temp file and return the path string."""
    data = {
        "lists": {
            "white": {
                "description": "Known humans.",
                "action": "keep",
                "addresses": ["alice@example.com"],
            },
            "black": {
                "description": "Blacklisted.",
                "action": "delete",
                "addresses": ["spam@scammer.com"],
            },
        },
        "priority": ["white", "black"],
    }
    path = tmp_path / "sender_lists.json"
    path.write_text(json.dumps(data))
    return str(path)


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


# --- corvus email lists ---


class TestEmailLists:
    def test_shows_lists(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "lists"])
        assert result.exit_code == 0
        assert "white" in result.output
        assert "black" in result.output
        assert "alice@example.com" in result.output
        assert "spam@scammer.com" in result.output

    def test_empty_lists(self, runner, tmp_path, monkeypatch):
        path = str(tmp_path / "empty.json")
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", path)
        result = runner.invoke(cli, ["email", "lists"])
        assert result.exit_code == 0
        assert "No sender lists configured" in result.output


# --- corvus email list-add ---


class TestEmailListAdd:
    def test_add_to_list(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "list-add", "black", "evil@bad.com"])
        assert result.exit_code == 0
        assert "Added" in result.output
        assert "evil@bad.com" in result.output

        # Verify it was persisted
        data = json.loads(open(sender_lists_path).read())
        assert "evil@bad.com" in data["lists"]["black"]["addresses"]

    def test_add_duplicate(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "list-add", "white", "alice@example.com"])
        assert result.exit_code == 0
        assert "already in" in result.output


# --- corvus email list-remove ---


class TestEmailListRemove:
    def test_remove_from_list(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "list-remove", "white", "alice@example.com"])
        assert result.exit_code == 0
        assert "Removed" in result.output

        # Verify it was persisted
        data = json.loads(open(sender_lists_path).read())
        assert "alice@example.com" not in data["lists"]["white"]["addresses"]

    def test_remove_nonexistent(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "list-remove", "white", "nobody@x.com"])
        assert result.exit_code == 0
        assert "not found" in result.output


# --- corvus email rationalize ---


class TestEmailRationalize:
    def test_clean_lists(self, runner, sender_lists_path, monkeypatch):
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", sender_lists_path)
        result = runner.invoke(cli, ["email", "rationalize"])
        assert result.exit_code == 0
        assert "clean" in result.output.lower() or "no changes" in result.output.lower()

    def test_dedup(self, runner, tmp_path, monkeypatch):
        data = {
            "lists": {
                "black": {
                    "action": "delete",
                    "addresses": ["dupe@spam.com", "DUPE@SPAM.COM"],
                },
            },
            "priority": ["black"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", str(path))

        result = runner.invoke(cli, ["email", "rationalize"])
        assert result.exit_code == 0
        assert "1 change" in result.output
        assert "duplicate" in result.output.lower()


# --- corvus email cleanup ---


class TestEmailCleanup:
    def test_no_cleanup_lists(self, runner, tmp_path, monkeypatch):
        """No lists with cleanup_days shows informative message."""
        data = {
            "lists": {
                "white": {"action": "keep", "addresses": ["a@b.com"]},
            },
            "priority": ["white"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr("corvus.cli.EMAIL_SENDER_LISTS_PATH", str(path))
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: SAMPLE_ACCOUNTS)

        result = runner.invoke(cli, ["email", "cleanup"])
        assert result.exit_code == 0
        assert "No sender lists have cleanup_days" in result.output

    def test_no_config_exits_with_error(self, runner, monkeypatch):
        monkeypatch.setattr("corvus.cli.load_email_accounts", lambda: [])
        result = runner.invoke(cli, ["email", "cleanup"])
        assert result.exit_code != 0
        assert "No email accounts configured" in result.output
