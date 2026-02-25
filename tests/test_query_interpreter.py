"""Tests for the query interpreter executor."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.executors.query_interpreter import (
    _build_system_prompt,
    _has_search_fields,
    _strip_today_only_date_range,
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


# --- Unit tests: _has_search_fields helper ---


class TestHasSearchFields:
    def test_all_empty(self):
        interp = QueryInterpretation(confidence=0.9, reasoning="test")
        assert _has_search_fields(interp) is False

    def test_text_search_only(self):
        interp = QueryInterpretation(
            text_search="mortgage", confidence=0.9, reasoning="test"
        )
        assert _has_search_fields(interp) is True

    def test_correspondent_only(self):
        interp = QueryInterpretation(
            correspondent_name="AT&T", confidence=0.9, reasoning="test"
        )
        assert _has_search_fields(interp) is True

    def test_document_type_only(self):
        interp = QueryInterpretation(
            document_type_name="Invoice", confidence=0.9, reasoning="test"
        )
        assert _has_search_fields(interp) is True

    def test_tags_only(self):
        interp = QueryInterpretation(
            tag_names=["tax-return"], confidence=0.9, reasoning="test"
        )
        assert _has_search_fields(interp) is True

    def test_empty_tag_list(self):
        interp = QueryInterpretation(
            tag_names=[], confidence=0.9, reasoning="test"
        )
        assert _has_search_fields(interp) is False


# --- Unit tests: empty-field fallback ---


class TestEmptyFieldFallback:
    async def test_empty_fields_injects_query_as_text_search(self):
        """When LLM returns all empty search fields, original query becomes text_search."""
        mock_interpretation = QueryInterpretation(
            confidence=0.90,
            reasoning="Looking for mortgage statement.",
        )
        mock_raw = OllamaResponse(
            model="gemma3",
            message={"role": "assistant", "content": "{}"},
            done=True,
        )

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_interpretation, mock_raw)

        result, _ = await interpret_query(
            "find latest mortgage statement",
            ollama=mock_ollama,
            model="gemma3",
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert result.text_search == "find latest mortgage statement"

    async def test_partial_fields_no_injection(self):
        """When LLM populates at least one field, text_search is not overwritten."""
        mock_interpretation = QueryInterpretation(
            correspondent_name="AT&T",
            text_search=None,
            confidence=0.85,
            reasoning="Found correspondent.",
        )
        mock_raw = OllamaResponse(
            model="gemma3",
            message={"role": "assistant", "content": "{}"},
            done=True,
        )

        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.generate_structured.return_value = (mock_interpretation, mock_raw)

        result, _ = await interpret_query(
            "AT&T bill",
            ollama=mock_ollama,
            model="gemma3",
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert result.text_search is None
        assert result.correspondent_name == "AT&T"


# --- Unit tests: prompt includes few-shot examples ---


class TestPromptExamples:
    def test_prompt_includes_examples_section(self):
        prompt = _build_system_prompt(SAMPLE_TAGS, SAMPLE_CORRESPONDENTS, SAMPLE_DOC_TYPES)
        assert "## Examples" in prompt
        assert "mortgage statement" in prompt
        assert "AT&T invoice" in prompt
        assert "tax 2022" in prompt


# --- Unit tests: today-only date range stripping ---


class TestStripTodayOnlyDateRange:
    def test_today_only_date_range_stripped(self):
        """When LLM returns start=today, end=today, both become None."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        interp = QueryInterpretation(
            text_search="mortgage statement",
            date_range_start=today,
            date_range_end=today,
            sort_order="newest",
            confidence=0.9,
            reasoning="Most recent mortgage statement",
        )
        _strip_today_only_date_range(interp)
        assert interp.date_range_start is None
        assert interp.date_range_end is None

    def test_explicit_date_range_preserved(self):
        """A real date range (not today-only) is preserved."""
        interp = QueryInterpretation(
            text_search="tax 2022",
            date_range_start="2022-01-01",
            date_range_end="2022-12-31",
            confidence=0.9,
            reasoning="Tax documents for 2022",
        )
        _strip_today_only_date_range(interp)
        assert interp.date_range_start == "2022-01-01"
        assert interp.date_range_end == "2022-12-31"

    def test_no_date_range_is_noop(self):
        """When no date range is set, nothing changes."""
        interp = QueryInterpretation(
            text_search="mortgage",
            confidence=0.9,
            reasoning="test",
        )
        _strip_today_only_date_range(interp)
        assert interp.date_range_start is None
        assert interp.date_range_end is None

    def test_partial_today_range_preserved(self):
        """If only start or end is today (not both), preserve both."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        interp = QueryInterpretation(
            text_search="docs",
            date_range_start="2025-01-01",
            date_range_end=today,
            confidence=0.8,
            reasoning="Docs since 2025",
        )
        _strip_today_only_date_range(interp)
        assert interp.date_range_start == "2025-01-01"
        assert interp.date_range_end == today


# --- Unit tests: prompt includes date range rule ---


class TestPromptDateRangeRule:
    def test_prompt_includes_date_range_rule(self):
        """System prompt contains the 'do NOT set date ranges' rule."""
        prompt = _build_system_prompt(SAMPLE_TAGS, SAMPLE_CORRESPONDENTS, SAMPLE_DOC_TYPES)
        assert "do NOT set date ranges" in prompt


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
