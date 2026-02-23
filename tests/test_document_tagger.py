"""Tests for the document tagger executor."""

import pytest

from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.executors.document_tagger import (
    _compute_overall_confidence,
    _determine_gate_action,
    tag_document,
)
from corvus.integrations.ollama import OllamaClient, pick_instruct_model
from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    GateAction,
    TagSuggestion,
)

# --- Unit tests (no external services) ---


class TestComputeOverallConfidence:
    def test_single_tag(self):
        result = DocumentTaggingResult(
            suggested_tags=[TagSuggestion(tag_name="invoice", confidence=0.9)],
            reasoning="test",
        )
        assert _compute_overall_confidence(result) == pytest.approx(0.9)

    def test_multiple_tags(self):
        result = DocumentTaggingResult(
            suggested_tags=[
                TagSuggestion(tag_name="invoice", confidence=0.8),
                TagSuggestion(tag_name="att", confidence=0.6),
            ],
            reasoning="test",
        )
        assert _compute_overall_confidence(result) == pytest.approx(0.7)

    def test_with_correspondent_and_type(self):
        result = DocumentTaggingResult(
            suggested_tags=[TagSuggestion(tag_name="invoice", confidence=0.9)],
            suggested_correspondent="AT&T",
            correspondent_confidence=0.8,
            suggested_document_type="Invoice",
            document_type_confidence=0.7,
            reasoning="test",
        )
        # (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert _compute_overall_confidence(result) == pytest.approx(0.8)

    def test_empty_suggestions(self):
        result = DocumentTaggingResult(suggested_tags=[], reasoning="nothing found")
        assert _compute_overall_confidence(result) == 0.0

    def test_correspondent_without_type(self):
        result = DocumentTaggingResult(
            suggested_tags=[TagSuggestion(tag_name="bill", confidence=0.85)],
            suggested_correspondent="Comcast",
            correspondent_confidence=0.75,
            reasoning="test",
        )
        # (0.85 + 0.75) / 2 = 0.8
        assert _compute_overall_confidence(result) == pytest.approx(0.8)


class TestDetermineGateAction:
    def test_high_confidence(self):
        assert _determine_gate_action(0.95) == GateAction.AUTO_EXECUTE
        assert _determine_gate_action(0.9) == GateAction.AUTO_EXECUTE

    def test_medium_confidence(self):
        assert _determine_gate_action(0.85) == GateAction.FLAG_IN_DIGEST
        assert _determine_gate_action(0.7) == GateAction.FLAG_IN_DIGEST

    def test_low_confidence(self):
        assert _determine_gate_action(0.69) == GateAction.QUEUE_FOR_REVIEW
        assert _determine_gate_action(0.0) == GateAction.QUEUE_FOR_REVIEW


# --- Integration test (requires Ollama + Paperless) ---


@pytest.mark.slow
@pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)
async def test_tag_document_live():
    """End-to-end: fetch a real document from Paperless, tag it via Ollama."""
    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL) as ollama,
    ):
        # Get available model (prefer instruct models)
        models = await ollama.list_models()
        model_name = pick_instruct_model(models)
        if not model_name:
            pytest.skip("No models available on Ollama server")

        # Fetch a real document
        docs, _count = await paperless.list_documents(page_size=1)
        if not docs:
            pytest.skip("No documents in Paperless instance")
        doc = docs[0]

        # Fetch existing metadata for context
        tags = await paperless.list_tags()
        correspondents = await paperless.list_correspondents()
        doc_types = await paperless.list_document_types()

        task, raw = await tag_document(
            doc,
            ollama=ollama,
            model=model_name,
            tags=tags,
            correspondents=correspondents,
            document_types=doc_types,
            keep_alive="0",
        )

        # Validate the task structure (executor correctness, not model quality)
        assert task.document_id == doc.id
        assert task.document_title == doc.title
        assert len(task.content_snippet) > 0
        assert 0.0 <= task.overall_confidence <= 1.0
        assert task.gate_action in GateAction
        assert isinstance(task.result, DocumentTaggingResult)
        assert isinstance(task.result.reasoning, str)

        # Validate the raw Ollama response
        assert raw.done is True
        assert raw.eval_count > 0
