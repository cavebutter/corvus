"""Tests for the Ollama API client."""

import pytest

from corvus.config import OLLAMA_BASE_URL
from corvus.integrations.ollama import OllamaClient
from corvus.schemas.document_tagging import DocumentTaggingResult


@pytest.fixture()
async def client():
    async with OllamaClient(OLLAMA_BASE_URL) as c:
        yield c


async def test_list_models(client: OllamaClient):
    models = await client.list_models()
    assert isinstance(models, list)
    # At least one model should be pulled
    assert len(models) > 0
    assert "name" in models[0]


@pytest.mark.slow
async def test_structured_output(client: OllamaClient):
    """Test structured generation against a real model.

    Requires a model to be available on the Ollama server.
    Uses the first available model.
    """
    models = await client.list_models()
    if not models:
        pytest.skip("No models available on Ollama server")

    model_name = models[0]["name"]

    result, raw = await client.generate_structured(
        model=model_name,
        schema_class=DocumentTaggingResult,
        system="You are a document classifier. Analyze the document and suggest tags.",
        prompt=(
            "Document title: AT&T Wireless Invoice\n"
            "Content: Account number 1234. Billing period Jan 1-31 2026. "
            "Total due: $85.50. Payment due by Feb 15, 2026."
        ),
        keep_alive="0",
    )

    assert isinstance(result, DocumentTaggingResult)
    assert len(result.suggested_tags) > 0
    assert all(0.0 <= t.confidence <= 1.0 for t in result.suggested_tags)
    assert isinstance(result.reasoning, str)
    assert raw.done is True
    assert raw.eval_count > 0
