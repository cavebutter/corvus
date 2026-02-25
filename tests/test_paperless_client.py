"""Tests for the Paperless-ngx API client."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from corvus.config import PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.integrations.paperless import (
    MAX_RETRIES,
    PaperlessClient,
    _retry_on_disconnect,
)

# ------------------------------------------------------------------
# Unit tests for _retry_on_disconnect (no live Paperless needed)
# ------------------------------------------------------------------


async def test_retry_on_remote_protocol_error():
    """First call raises RemoteProtocolError, second succeeds."""
    mock_fn = AsyncMock(side_effect=[httpx.RemoteProtocolError("peer closed"), "ok"])

    with patch("corvus.integrations.paperless.asyncio.sleep", new_callable=AsyncMock):
        result = await _retry_on_disconnect(mock_fn, "arg1", key="val")

    assert result == "ok"
    assert mock_fn.call_count == 2
    mock_fn.assert_called_with("arg1", key="val")


async def test_retry_exhausted_raises():
    """All calls raise — final RemoteProtocolError is re-raised."""
    mock_fn = AsyncMock(
        side_effect=[httpx.RemoteProtocolError("drop")] * (MAX_RETRIES + 1)
    )

    with (
        patch("corvus.integrations.paperless.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(httpx.RemoteProtocolError),
    ):
        await _retry_on_disconnect(mock_fn)

    assert mock_fn.call_count == MAX_RETRIES + 1


# ------------------------------------------------------------------
# Integration tests (require live Paperless)
# ------------------------------------------------------------------

_skip_no_paperless = pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)


@pytest.fixture()
async def client():
    async with PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as c:
        yield c


@_skip_no_paperless
async def test_list_documents(client: PaperlessClient):
    docs, count = await client.list_documents(page_size=5)
    assert isinstance(count, int)
    assert count >= 0
    for doc in docs:
        assert doc.id > 0
        assert isinstance(doc.title, str)


@_skip_no_paperless
async def test_list_tags(client: PaperlessClient):
    tags = await client.list_tags()
    assert isinstance(tags, list)
    for tag in tags:
        assert tag.id > 0
        assert isinstance(tag.name, str)


@_skip_no_paperless
async def test_list_correspondents(client: PaperlessClient):
    correspondents = await client.list_correspondents()
    assert isinstance(correspondents, list)


@_skip_no_paperless
async def test_list_document_types(client: PaperlessClient):
    doc_types = await client.list_document_types()
    assert isinstance(doc_types, list)
