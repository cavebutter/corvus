"""Tests for the Paperless-ngx API client."""

import pytest

from corvus.config import PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.integrations.paperless import PaperlessClient

pytestmark = pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)


@pytest.fixture()
async def client():
    async with PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as c:
        yield c


async def test_list_documents(client: PaperlessClient):
    docs, count = await client.list_documents(page_size=5)
    assert isinstance(count, int)
    assert count >= 0
    for doc in docs:
        assert doc.id > 0
        assert isinstance(doc.title, str)


async def test_list_tags(client: PaperlessClient):
    tags = await client.list_tags()
    assert isinstance(tags, list)
    for tag in tags:
        assert tag.id > 0
        assert isinstance(tag.name, str)


async def test_list_correspondents(client: PaperlessClient):
    correspondents = await client.list_correspondents()
    assert isinstance(correspondents, list)


async def test_list_document_types(client: PaperlessClient):
    doc_types = await client.list_document_types()
    assert isinstance(doc_types, list)
