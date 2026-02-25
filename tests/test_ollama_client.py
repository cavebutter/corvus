"""Tests for the Ollama API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from corvus.config import OLLAMA_BASE_URL
from corvus.integrations.ollama import OllamaClient
from corvus.schemas.document_tagging import DocumentTaggingResult


# ------------------------------------------------------------------
# Unit tests (no Ollama server needed)
# ------------------------------------------------------------------


class TestChatMessageBuilding:
    """Test that chat() builds the messages payload correctly."""

    async def test_chat_with_messages_builds_correct_payload(self):
        """Messages are inserted between system prompt and user message."""
        client = OllamaClient("http://localhost:11434")

        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "model": "test",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await client.chat(
                model="test-model",
                system="You are helpful.",
                prompt="current question",
                messages=history,
            )

        payload = mock_post.call_args.kwargs["json"]
        msgs = payload["messages"]
        assert len(msgs) == 4
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1] == {"role": "user", "content": "previous question"}
        assert msgs[2] == {"role": "assistant", "content": "previous answer"}
        assert msgs[3] == {"role": "user", "content": "current question"}

        await client.close()

    async def test_chat_without_messages_backwards_compatible(self):
        """Without messages param, payload has system + user only (2 messages)."""
        client = OllamaClient("http://localhost:11434")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "model": "test",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await client.chat(
                model="test-model",
                system="You are helpful.",
                prompt="hello",
            )

        payload = mock_post.call_args.kwargs["json"]
        msgs = payload["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

        await client.close()


# ------------------------------------------------------------------
# Integration tests (require Ollama server)
# ------------------------------------------------------------------


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
