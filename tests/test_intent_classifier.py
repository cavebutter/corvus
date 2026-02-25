"""Tests for the intent classifier executor."""

from unittest.mock import AsyncMock

import pytest

from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.integrations.ollama import OllamaClient, OllamaResponse, pick_instruct_model
from corvus.planner.intent_classifier import SYSTEM_PROMPT, classify_intent
from corvus.schemas.orchestrator import Intent, IntentClassification

# ------------------------------------------------------------------
# System prompt tests
# ------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_all_intents(self):
        for intent in Intent:
            assert intent.value in SYSTEM_PROMPT

    def test_contains_classification_rules(self):
        assert "ambiguous" in SYSTEM_PROMPT.lower()
        assert "fetch_document" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT.lower()


# ------------------------------------------------------------------
# Mocked LLM tests
# ------------------------------------------------------------------


def _mock_classification(intent: Intent, confidence: float = 0.9, **kwargs):
    return IntentClassification(
        intent=intent,
        confidence=confidence,
        reasoning="Test reasoning",
        **kwargs,
    )


def _mock_raw():
    return OllamaResponse(
        model="gemma3",
        message={"role": "assistant", "content": "{}"},
        done=True,
        eval_count=50,
    )


class TestClassifyIntent:
    async def test_returns_classification_and_response(self):
        """classify_intent returns (IntentClassification, OllamaResponse)."""
        mock_classification = _mock_classification(Intent.FETCH_DOCUMENT, fetch_query="AT&T invoice")
        mock_raw = _mock_raw()

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, mock_raw)

        result, raw = await classify_intent(
            "find my AT&T invoice",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.intent == Intent.FETCH_DOCUMENT
        assert result.fetch_query == "AT&T invoice"
        assert result.confidence == 0.9
        assert raw.done is True

    async def test_tag_documents_intent(self):
        """Classifies tagging requests."""
        mock_classification = _mock_classification(Intent.TAG_DOCUMENTS, tag_limit=5)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        result, _raw = await classify_intent(
            "tag 5 documents",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.intent == Intent.TAG_DOCUMENTS
        assert result.tag_limit == 5

    async def test_review_queue_intent(self):
        """Classifies review queue requests."""
        mock_classification = _mock_classification(Intent.REVIEW_QUEUE)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        result, _raw = await classify_intent(
            "review pending items",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.intent == Intent.REVIEW_QUEUE

    async def test_show_digest_intent(self):
        """Classifies digest requests with hours param."""
        mock_classification = _mock_classification(
            Intent.SHOW_DIGEST, digest_hours=48,
        )
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        result, _raw = await classify_intent(
            "what happened in the last 2 days",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.intent == Intent.SHOW_DIGEST
        assert result.digest_hours == 48

    async def test_general_chat_intent(self):
        """Classifies general conversation."""
        mock_classification = _mock_classification(Intent.GENERAL_CHAT)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        result, _raw = await classify_intent(
            "hello, how are you?",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.intent == Intent.GENERAL_CHAT

    async def test_low_confidence(self):
        """Low confidence classification is passed through."""
        mock_classification = _mock_classification(Intent.FETCH_DOCUMENT, confidence=0.4)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        result, _raw = await classify_intent(
            "something vague",
            ollama=mock_ollama,
            model="gemma3",
        )

        assert result.confidence == 0.4

    async def test_passes_keep_alive(self):
        """keep_alive is forwarded to the Ollama client."""
        mock_classification = _mock_classification(Intent.GENERAL_CHAT)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        await classify_intent(
            "hello",
            ollama=mock_ollama,
            model="gemma3",
            keep_alive="10m",
        )

        call_kwargs = mock_ollama.generate_structured.call_args.kwargs
        assert call_kwargs["keep_alive"] == "10m"

    async def test_uses_intent_classification_schema(self):
        """generate_structured is called with IntentClassification schema."""
        mock_classification = _mock_classification(Intent.GENERAL_CHAT)
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_classification, _mock_raw())

        await classify_intent(
            "hello",
            ollama=mock_ollama,
            model="gemma3",
        )

        call_kwargs = mock_ollama.generate_structured.call_args.kwargs
        assert call_kwargs["schema_class"] is IntentClassification
        assert call_kwargs["model"] == "gemma3"


# ------------------------------------------------------------------
# Integration test (requires Ollama)
# ------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)
async def test_classify_intent_live():
    """End-to-end: classify a real query via Ollama."""
    async with OllamaClient(OLLAMA_BASE_URL) as ollama:
        models = await ollama.list_models()
        model_name = pick_instruct_model(models)
        if not model_name:
            pytest.skip("No models available on Ollama server")

        result, raw = await classify_intent(
            "find my latest invoice from AT&T",
            ollama=ollama,
            model=model_name,
            keep_alive="0",
        )

        assert isinstance(result, IntentClassification)
        assert result.intent == Intent.FETCH_DOCUMENT
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert raw.done is True
        assert raw.eval_count > 0
