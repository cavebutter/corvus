"""Tests for corvus.integrations.imap — IMAP client with mocked MailBox."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvus.integrations.imap import ImapClient, _parse_envelope, _parse_message
from corvus.schemas.email import EmailAccountConfig


# --- Mock helpers ---


class MockAttachment:
    """Minimal mock for an imap-tools attachment."""

    def __init__(self, filename: str = "file.pdf"):
        self.filename = filename


class MockMailMessage:
    """Minimal mock for an imap-tools MailMessage."""

    def __init__(
        self,
        uid="123",
        from_="Test Sender <test@example.com>",
        to=("recipient@example.com",),
        subject="Test Subject",
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        flags=("\\Seen",),
        size=1024,
        text="Hello body",
        html="",
        attachments=(),
    ):
        self.uid = uid
        self.from_ = from_
        self.to = to
        self.subject = subject
        self.date = date
        self.flags = flags
        self.size = size
        self.text = text
        self.html = html
        self.attachments = attachments


def _make_config(**overrides) -> EmailAccountConfig:
    defaults = dict(
        name="test",
        server="imap.example.com",
        email="user@example.com",
        password="secret",
        folders={"inbox": "INBOX", "receipts": "Corvus/Receipts"},
    )
    defaults.update(overrides)
    return EmailAccountConfig(**defaults)


# --- _parse_envelope ---


class TestParseEnvelope:
    def test_basic_envelope(self):
        msg = MockMailMessage()
        env = _parse_envelope(msg, "user@example.com")

        assert env.uid == "123"
        assert env.account_email == "user@example.com"
        assert env.from_address == "test@example.com"
        assert env.from_name == "Test Sender"
        assert env.to == ["recipient@example.com"]
        assert env.subject == "Test Subject"
        assert env.flags == ["\\Seen"]
        assert env.size_bytes == 1024

    def test_no_subject(self):
        msg = MockMailMessage(subject="")
        env = _parse_envelope(msg, "user@example.com")
        assert env.subject == "(no subject)"

    def test_no_from_name(self):
        msg = MockMailMessage(from_="noreply@system.com")
        env = _parse_envelope(msg, "user@example.com")
        assert env.from_address == "noreply@system.com"
        assert env.from_name == ""

    def test_multiple_recipients(self):
        msg = MockMailMessage(to=("a@example.com", "b@example.com", "c@example.com"))
        env = _parse_envelope(msg, "user@example.com")
        assert len(env.to) == 3


# --- _parse_message ---


class TestParseMessage:
    def test_basic_message(self):
        msg = MockMailMessage(text="Plain body", html="<p>HTML body</p>")
        email = _parse_message(msg, "user@example.com")

        assert email.uid == "123"
        assert email.body_text == "Plain body"
        assert email.body_html == "<p>HTML body</p>"
        assert email.has_attachments is False
        assert email.attachment_names == []

    def test_message_with_attachments(self):
        attachments = [MockAttachment("doc.pdf"), MockAttachment("image.png")]
        msg = MockMailMessage(attachments=attachments)
        email = _parse_message(msg, "user@example.com")

        assert email.has_attachments is True
        assert email.attachment_names == ["doc.pdf", "image.png"]

    def test_empty_body(self):
        msg = MockMailMessage(text="", html="")
        email = _parse_message(msg, "user@example.com")
        assert email.body_text == ""
        assert email.body_html == ""

    def test_attachment_without_filename_excluded(self):
        """Attachments with no filename should not appear in attachment_names."""
        attachments = [MockAttachment("report.pdf"), MockAttachment("")]
        msg = MockMailMessage(attachments=attachments)
        email = _parse_message(msg, "user@example.com")
        # The empty-filename attachment is filtered out
        assert email.attachment_names == ["report.pdf"]
        # But has_attachments is true because there are 2 attachment objects
        assert email.has_attachments is True


# --- ImapClient ---


class TestImapClientMoveNoOp:
    """Empty UID lists should be no-ops (no IMAP calls made)."""

    async def test_move_empty_uids(self):
        config = _make_config()
        client = ImapClient(config)
        # No mailbox connected — would fail if it tried to call anything
        await client.move([], "SomeFolder")

    async def test_delete_empty_uids(self):
        config = _make_config()
        client = ImapClient(config)
        await client.delete([])

    async def test_flag_empty_uids(self):
        config = _make_config()
        client = ImapClient(config)
        await client.flag([], "\\Flagged")

    async def test_mark_read_empty_uids(self):
        config = _make_config()
        client = ImapClient(config)
        await client.mark_read([])


class TestImapClientGmailMove:
    """Gmail accounts should use COPY + DELETE instead of MOVE."""

    async def test_gmail_move_uses_copy_and_delete(self):
        config = _make_config(is_gmail=True)
        client = ImapClient(config)

        mock_mailbox = MagicMock()
        mock_mailbox.copy = MagicMock()
        mock_mailbox.delete = MagicMock()
        client._mailbox = mock_mailbox

        await client.move(["uid1", "uid2"], "Archive")

        mock_mailbox.copy.assert_called_once_with(["uid1", "uid2"], "Archive")
        mock_mailbox.delete.assert_called_once_with(["uid1", "uid2"])

    async def test_standard_move_uses_mailbox_move(self):
        config = _make_config(is_gmail=False)
        client = ImapClient(config)

        mock_mailbox = MagicMock()
        mock_mailbox.move = MagicMock()
        client._mailbox = mock_mailbox

        await client.move(["uid1"], "Processed")

        mock_mailbox.move.assert_called_once_with(["uid1"], "Processed")


class TestImapClientEnsureFolders:
    """ensure_folders should only create folders that don't already exist."""

    async def test_creates_missing_folders(self):
        config = _make_config()
        client = ImapClient(config)

        mock_folder = MagicMock()
        # Simulate existing folders: list returns folder objects with .name attribute
        existing_1 = MagicMock()
        existing_1.name = "INBOX"
        existing_2 = MagicMock()
        existing_2.name = "Corvus/Receipts"
        mock_folder.list.return_value = [existing_1, existing_2]
        mock_folder.create = MagicMock()

        mock_mailbox = MagicMock()
        mock_mailbox.folder = mock_folder
        client._mailbox = mock_mailbox

        await client.ensure_folders(["INBOX", "Corvus/Receipts", "Corvus/Processed"])

        # Only the missing folder should be created
        mock_folder.create.assert_called_once_with("Corvus/Processed")

    async def test_no_creation_when_all_exist(self):
        config = _make_config()
        client = ImapClient(config)

        mock_folder = MagicMock()
        existing = MagicMock()
        existing.name = "INBOX"
        mock_folder.list.return_value = [existing]
        mock_folder.create = MagicMock()

        mock_mailbox = MagicMock()
        mock_mailbox.folder = mock_folder
        client._mailbox = mock_mailbox

        await client.ensure_folders(["INBOX"])

        mock_folder.create.assert_not_called()


class TestImapClientFetchUidsOlderThan:
    """fetch_uids_older_than uses server-side date filtering."""

    async def test_returns_uids_for_old_messages(self):
        config = _make_config()
        client = ImapClient(config)

        old_msg1 = MockMailMessage(uid="10")
        old_msg2 = MockMailMessage(uid="20")

        mock_folder = MagicMock()
        mock_mailbox = MagicMock()
        mock_mailbox.folder = mock_folder
        mock_mailbox.fetch = MagicMock(return_value=[old_msg1, old_msg2])
        client._mailbox = mock_mailbox

        uids = await client.fetch_uids_older_than("Corvus/Ads", days=14)

        assert uids == ["10", "20"]
        mock_folder.set.assert_called_once_with("Corvus/Ads")
        mock_mailbox.fetch.assert_called_once()

    async def test_returns_empty_for_no_matches(self):
        config = _make_config()
        client = ImapClient(config)

        mock_folder = MagicMock()
        mock_mailbox = MagicMock()
        mock_mailbox.folder = mock_folder
        mock_mailbox.fetch = MagicMock(return_value=[])
        client._mailbox = mock_mailbox

        uids = await client.fetch_uids_older_than("INBOX", days=30)

        assert uids == []


class TestImapClientMailboxProperty:
    """Accessing .mailbox before connecting should raise RuntimeError."""

    def test_mailbox_not_connected_raises(self):
        config = _make_config()
        client = ImapClient(config)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.mailbox
