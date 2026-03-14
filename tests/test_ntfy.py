"""Tests for the Ntfy push notification integration."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner

_DUMMY_REQUEST = httpx.Request("POST", "https://ntfy.example.com/test")

from corvus.integrations.ntfy import (
    PRIORITY_DEFAULT,
    PRIORITY_HIGH,
    notify_digest_ready,
    notify_pipeline_error,
    notify_tag_pipeline_complete,
    notify_triage_complete,
    send,
)


@pytest.mark.asyncio
async def test_send_success(monkeypatch):
    """Successful POST returns True."""
    captured = {}

    async def mock_post(self, url, *, content, headers):
        captured["url"] = str(url)
        captured["content"] = content
        captured["headers"] = headers
        return httpx.Response(200, request=_DUMMY_REQUEST)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await send(
        server="https://ntfy.example.com",
        topic="corvus-test",
        message="Hello world",
        title="Test",
        tags=["check"],
    )
    assert result is True
    assert captured["url"] == "https://ntfy.example.com/corvus-test"
    assert captured["content"] == "Hello world"
    assert captured["headers"]["Title"] == "Test"
    assert captured["headers"]["Tags"] == "check"


@pytest.mark.asyncio
async def test_send_with_priority_and_click(monkeypatch):
    """Headers include priority and click URL when set."""
    captured = {}

    async def mock_post(self, url, *, content, headers):
        captured["headers"] = headers
        return httpx.Response(200, request=_DUMMY_REQUEST)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await send(
        server="https://ntfy.example.com",
        topic="test",
        message="msg",
        priority=PRIORITY_HIGH,
        click_url="https://corvus.local/review.html",
    )
    assert captured["headers"]["Priority"] == PRIORITY_HIGH
    assert captured["headers"]["Click"] == "https://corvus.local/review.html"


@pytest.mark.asyncio
async def test_send_default_priority_omitted(monkeypatch):
    """Default priority is not sent as a header."""
    captured = {}

    async def mock_post(self, url, *, content, headers):
        captured["headers"] = headers
        return httpx.Response(200, request=_DUMMY_REQUEST)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await send(
        server="https://ntfy.example.com",
        topic="test",
        message="msg",
    )
    assert "Priority" not in captured["headers"]


@pytest.mark.asyncio
async def test_send_failure_returns_false(monkeypatch):
    """HTTP error returns False (does not raise)."""

    async def mock_post(self, url, *, content, headers):
        return httpx.Response(500, request=_DUMMY_REQUEST)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await send(
        server="https://ntfy.example.com",
        topic="test",
        message="msg",
    )
    assert result is False


@pytest.mark.asyncio
async def test_send_network_error_returns_false(monkeypatch):
    """Network exception returns False (does not raise)."""

    async def mock_post(self, url, *, content, headers):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await send(
        server="https://ntfy.example.com",
        topic="test",
        message="msg",
    )
    assert result is False


@pytest.mark.asyncio
async def test_send_no_config_returns_false():
    """Empty server or topic returns False immediately."""
    assert await send(server="", topic="test", message="msg") is False
    assert await send(server="https://ntfy.sh", topic="", message="msg") is False


@pytest.mark.asyncio
async def test_send_strips_trailing_slash(monkeypatch):
    """Trailing slash on server URL is normalized."""
    captured = {}

    async def mock_post(self, url, *, content, headers):
        captured["url"] = str(url)
        return httpx.Response(200, request=_DUMMY_REQUEST)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await send(server="https://ntfy.sh/", topic="my-topic", message="msg")
    assert captured["url"] == "https://ntfy.sh/my-topic"


# ── Pipeline helper tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_triage_skips_when_nothing_queued(monkeypatch):
    """No notification when queued=0 and errors=0."""
    result = await notify_triage_complete(
        account_email="test@example.com",
        processed=5,
        queued=0,
        auto_acted=5,
        errors=0,
    )
    assert result is False


@pytest.mark.asyncio
async def test_notify_triage_sends_when_queued(monkeypatch):
    """Sends notification when items are queued for review."""
    sent = {}

    async def mock_notify(message, *, title=None, priority=PRIORITY_DEFAULT, tags=None, click_url=None):
        sent["message"] = message
        sent["title"] = title
        sent["priority"] = priority
        sent["tags"] = tags
        return True

    monkeypatch.setattr("corvus.integrations.ntfy.notify", mock_notify)

    result = await notify_triage_complete(
        account_email="jay@example.com",
        processed=10,
        queued=3,
        auto_acted=7,
        errors=0,
    )
    assert result is True
    assert "3 queued for review" in sent["message"]
    assert "jay@example.com" in sent["message"]
    assert sent["priority"] == PRIORITY_HIGH


@pytest.mark.asyncio
async def test_notify_triage_sends_on_errors(monkeypatch):
    """Sends notification when there are errors even if nothing queued."""
    sent = {}

    async def mock_notify(message, **kwargs):
        sent["message"] = message
        sent["tags"] = kwargs.get("tags")
        return True

    monkeypatch.setattr("corvus.integrations.ntfy.notify", mock_notify)

    result = await notify_triage_complete(
        account_email="test@example.com",
        processed=5,
        queued=0,
        auto_acted=5,
        errors=2,
    )
    assert result is True
    assert "2 error(s)" in sent["message"]
    assert "warning" in sent["tags"]


@pytest.mark.asyncio
async def test_notify_tag_pipeline_skips_when_nothing_queued():
    """No notification when queued=0 and errors=0."""
    result = await notify_tag_pipeline_complete(
        processed=10, queued=0, auto_applied=10, errors=0,
    )
    assert result is False


@pytest.mark.asyncio
async def test_notify_tag_pipeline_sends_when_queued(monkeypatch):
    """Sends notification when documents are queued for review."""
    sent = {}

    async def mock_notify(message, **kwargs):
        sent["message"] = message
        return True

    monkeypatch.setattr("corvus.integrations.ntfy.notify", mock_notify)

    result = await notify_tag_pipeline_complete(
        processed=5, queued=2, auto_applied=3, errors=0,
    )
    assert result is True
    assert "2 queued for review" in sent["message"]


@pytest.mark.asyncio
async def test_notify_digest_ready(monkeypatch):
    """Sends digest content as notification."""
    sent = {}

    async def mock_notify(message, **kwargs):
        sent["message"] = message
        sent["title"] = kwargs.get("title")
        return True

    monkeypatch.setattr("corvus.integrations.ntfy.notify", mock_notify)

    await notify_digest_ready("Summary: 5 processed, 2 pending")
    assert "5 processed" in sent["message"]
    assert sent["title"] == "Daily digest"


@pytest.mark.asyncio
async def test_notify_pipeline_error(monkeypatch):
    """Sends error notification with high priority."""
    sent = {}

    async def mock_notify(message, **kwargs):
        sent["message"] = message
        sent["priority"] = kwargs.get("priority")
        return True

    monkeypatch.setattr("corvus.integrations.ntfy.notify", mock_notify)

    await notify_pipeline_error("email_triage", "IMAP connection failed")
    assert "email_triage" in sent["message"]
    assert "IMAP connection failed" in sent["message"]
    assert sent["priority"] == PRIORITY_HIGH


# ── CLI tests ────────────────────────────────────────────────────────


class TestNotifyCLI:
    """Tests for the `corvus notify` command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_notify_no_config(self, runner, monkeypatch):
        """Errors when NTFY_SERVER/NTFY_TOPIC not set."""
        from corvus.cli import cli

        monkeypatch.setattr("corvus.cli.NTFY_SERVER", "")
        monkeypatch.setattr("corvus.cli.NTFY_TOPIC", "")
        result = runner.invoke(cli, ["notify"])
        assert result.exit_code == 1
        assert "NTFY_SERVER" in result.output

    def test_notify_success(self, runner, monkeypatch):
        """Sends test notification successfully."""
        from corvus.cli import cli

        monkeypatch.setattr("corvus.cli.NTFY_SERVER", "https://ntfy.example.com")
        monkeypatch.setattr("corvus.cli.NTFY_TOPIC", "corvus-test")

        mock = AsyncMock(return_value=True)
        with patch("corvus.integrations.ntfy.notify", mock):
            result = runner.invoke(cli, ["notify", "Hello from test"])

        assert result.exit_code == 0
        assert "Sent successfully" in result.output

    def test_notify_failure(self, runner, monkeypatch):
        """Reports failure when ntfy returns error."""
        from corvus.cli import cli

        monkeypatch.setattr("corvus.cli.NTFY_SERVER", "https://ntfy.example.com")
        monkeypatch.setattr("corvus.cli.NTFY_TOPIC", "corvus-test")

        mock = AsyncMock(return_value=False)
        with patch("corvus.integrations.ntfy.notify", mock):
            result = runner.invoke(cli, ["notify"])

        assert result.exit_code == 1
        assert "Failed" in result.output


class TestDigestNotify:
    """Tests for the `corvus digest --notify` flag."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_digest_notify_sends(self, runner, monkeypatch, tmp_path):
        """--notify flag sends digest via ntfy."""
        from corvus.cli import cli

        monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
        monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

        mock = AsyncMock(return_value=True)
        with patch("corvus.integrations.ntfy.notify_digest_ready", mock):
            result = runner.invoke(cli, ["digest", "--notify"])

        assert result.exit_code == 0
        assert mock.called
        # The digest text should be passed to notify
        call_arg = mock.call_args[0][0]
        assert "Corvus Daily Digest" in call_arg

    def test_digest_without_notify_does_not_send(self, runner, monkeypatch, tmp_path):
        """Without --notify, no notification is sent."""
        from corvus.cli import cli

        monkeypatch.setattr("corvus.cli.QUEUE_DB_PATH", str(tmp_path / "queue.db"))
        monkeypatch.setattr("corvus.cli.AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))

        mock = AsyncMock(return_value=True)
        with patch("corvus.integrations.ntfy.notify_digest_ready", mock):
            result = runner.invoke(cli, ["digest"])

        assert result.exit_code == 0
        assert not mock.called
