"""Tests for the query interpreter executor."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.executors.query_interpreter import (
    _build_system_prompt,
    interpret_query,
)
from corvus.integrations.ollama import OllamaClient, OllamaResponse, pick_instruct_model
from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.document_retrieval import QueryInterpretation
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocumentType,
    PaperlessTag,
)

# --- Test fixtures ---

SAMPLE_TAGS = [
    PaperlessTag(id=1, name="invoice", slug="invoice"),
    PaperlessTag(id=2, name="utility-bill", slug="utility-bill"),
    PaperlessTag(id=3, name="tax-return", slug="tax-return"),
]

SAMPLE_CORRESPONDENTS = [
    PaperlessCorrespondent(id=10, name="AT&T", slug="att"),
    PaperlessCorrespondent(id=11, name="Comcast", slug="comcast"),
]

SAMPLE_DOC_TYPES = [
    PaperlessDocumentType(id=20, name="Invoice", slug="invoice"),
    PaperlessDocumentType(id=21, name="Statement", slug="statement"),
]


# --- Unit tests: prompt construction ---


class TestBuildSystemPrompt:
    def test_includes_all_names(self):
        prompt = _build_system_prompt(SAMPLE_TAGS, SAMPLE_CORRESPONDENTS, SAMPLE_DOC_TYPES)
        assert "invoice" in prompt
        assert "utility-bill" in prompt
        assert "AT&T" in prompt
        assert "Comcast" in prompt
        assert "Invoice" in prompt
        assert "Statement" in prompt

    def test_includes_today_date(self):
        prompt = _build_system_prompt([], [], [])
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert today in prompt

    def test_empty_lists(self):
        prompt = _build_system_prompt([], [], [])
        assert "(none yet)" in prompt

    def test_single_items(self):
        tags = [PaperlessTag(id=1, name="receipt", slug="receipt")]
        correspondents = [PaperlessCorrespondent(id=1, name="Amazon", slug="amazon")]
        doc_types = [PaperlessDocumentType(id=1, name="Receipt", slug="receipt")]
        prompt = _build_system_prompt(tags, correspondents, doc_types)
        assert "receipt" in prompt
        assert "Amazon" in prompt
        assert "Receipt" in prompt


# --- Unit tests: mocked LLM calls ---


class TestInterpretQuery:
    async def test_returns_interpretation_and_response(self):
        """interpret_query returns (QueryInterpretation, OllamaResponse)."""
        mock_interpretation = QueryInterpretation(
            correspondent_name="AT&T",
            document_type_name="Invoice",
            tag_names=[],
            text_search=None,
            sort_order="newest",
            confidence=0.9,
            reasoning="User wants the most recent AT&T invoice.",
        )
        mock_raw = OllamaResponse(
            model="gemma3",
            message={"role": "assistant", "content": "{}"},
            done=True,
            eval_count=50,
        )

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_interpretation, mock_raw)

        result, raw = await interpret_query(
            "most recent invoice from AT&T",
            ollama=mock_ollama,
            model="gemma3",
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert result.correspondent_name == "AT&T"
        assert result.document_type_name == "Invoice"
        assert result.confidence == 0.9
        assert raw.done is True

    async def test_passes_keep_alive(self):
        """keep_alive is forwarded to the Ollama client."""
        mock_interpretation = QueryInterpretation(
            confidence=0.8,
            reasoning="test",
        )
        mock_raw = OllamaResponse(
            model="gemma3",
            message={"role": "assistant", "content": "{}"},
            done=True,
        )

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_interpretation, mock_raw)

        await interpret_query(
            "any document",
            ollama=mock_ollama,
            model="gemma3",
            tags=[],
            correspondents=[],
            document_types=[],
            keep_alive="10m",
        )

        call_kwargs = mock_ollama.generate_structured.call_args.kwargs
        assert call_kwargs["keep_alive"] == "10m"

    async def test_system_prompt_sent_to_ollama(self):
        """The system prompt includes available metadata."""
        mock_interpretation = QueryInterpretation(
            confidence=0.8,
            reasoning="test",
        )
        mock_raw = OllamaResponse(
            model="gemma3",
            message={"role": "assistant", "content": "{}"},
            done=True,
        )

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_interpretation, mock_raw)

        await interpret_query(
            "invoice from AT&T",
            ollama=mock_ollama,
            model="gemma3",
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        call_kwargs = mock_ollama.generate_structured.call_args.kwargs
        assert "AT&T" in call_kwargs["system"]
        assert "invoice" in call_kwargs["system"]


# --- Integration test (requires Ollama + Paperless) ---


@pytest.mark.slow
@pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)
async def test_interpret_query_live():
    """End-to-end: interpret a real query via Ollama."""
    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL) as ollama,
    ):
        models = await ollama.list_models()
        model_name = pick_instruct_model(models)
        if not model_name:
            pytest.skip("No models available on Ollama server")

        tags = await paperless.list_tags()
        correspondents = await paperless.list_correspondents()
        doc_types = await paperless.list_document_types()

        result, raw = await interpret_query(
            "most recent invoice",
            ollama=ollama,
            model=model_name,
            tags=tags,
            correspondents=correspondents,
            document_types=doc_types,
            keep_alive="0",
        )

        assert isinstance(result, QueryInterpretation)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert raw.done is True
        assert raw.eval_count > 0
