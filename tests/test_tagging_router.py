"""Tests for the document tagging router."""

import pytest

from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.executors.document_tagger import tag_document
from corvus.integrations.ollama import OllamaClient, pick_instruct_model
from corvus.integrations.paperless import PaperlessClient
from corvus.router.tagging import (
    RoutingResult,
    resolve_and_route,
    resolve_correspondent,
    resolve_document_type,
    resolve_tag,
)
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
    TagSuggestion,
)
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


def _make_task(
    *,
    tag_names: list[tuple[str, float]] | None = None,
    correspondent: str | None = None,
    corr_confidence: float = 0.0,
    doc_type: str | None = None,
    dtype_confidence: float = 0.0,
    overall_confidence: float = 0.85,
    gate_action: GateAction = GateAction.FLAG_IN_DIGEST,
) -> DocumentTaggingTask:
    """Build a DocumentTaggingTask for testing."""
    suggestions = [
        TagSuggestion(tag_name=name, confidence=conf)
        for name, conf in (tag_names or [("invoice", 0.9)])
    ]
    return DocumentTaggingTask(
        document_id=42,
        document_title="Test Invoice",
        content_snippet="Some invoice content...",
        result=DocumentTaggingResult(
            suggested_tags=suggestions,
            suggested_correspondent=correspondent,
            suggested_document_type=doc_type,
            correspondent_confidence=corr_confidence,
            document_type_confidence=dtype_confidence,
            reasoning="Test reasoning",
        ),
        overall_confidence=overall_confidence,
        gate_action=gate_action,
    )


# --- Unit tests: name resolution ---


class TestResolveTag:
    def test_exact_match(self):
        assert resolve_tag("invoice", SAMPLE_TAGS) == 1

    def test_case_insensitive(self):
        assert resolve_tag("Invoice", SAMPLE_TAGS) == 1
        assert resolve_tag("UTILITY-BILL", SAMPLE_TAGS) == 2

    def test_whitespace_stripped(self):
        assert resolve_tag("  invoice  ", SAMPLE_TAGS) == 1

    def test_no_match(self):
        assert resolve_tag("receipt", SAMPLE_TAGS) is None

    def test_empty_list(self):
        assert resolve_tag("invoice", []) is None


class TestResolveCorrespondent:
    def test_exact_match(self):
        assert resolve_correspondent("AT&T", SAMPLE_CORRESPONDENTS) == 10

    def test_case_insensitive(self):
        assert resolve_correspondent("at&t", SAMPLE_CORRESPONDENTS) == 10
        assert resolve_correspondent("comcast", SAMPLE_CORRESPONDENTS) == 11

    def test_no_match(self):
        assert resolve_correspondent("Verizon", SAMPLE_CORRESPONDENTS) is None


class TestResolveDocumentType:
    def test_exact_match(self):
        assert resolve_document_type("Invoice", SAMPLE_DOC_TYPES) == 20

    def test_case_insensitive(self):
        assert resolve_document_type("invoice", SAMPLE_DOC_TYPES) == 20
        assert resolve_document_type("STATEMENT", SAMPLE_DOC_TYPES) == 21

    def test_no_match(self):
        assert resolve_document_type("Receipt", SAMPLE_DOC_TYPES) is None


# --- Unit tests: gate logic ---


class TestForceQueue:
    """Verify that force_queue overrides the gate action."""

    async def test_force_queue_overrides_auto_execute(self):
        """Even with high confidence, force_queue should queue for review."""
        task = _make_task(
            overall_confidence=0.95,
            gate_action=GateAction.AUTO_EXECUTE,
        )

        # We need a mock-ish paperless client. Since force_queue=True,
        # it should never call paperless.update_document.
        # Use a real client pointed at a dummy URL — it won't be called.
        async with PaperlessClient("http://localhost:0", "fake") as paperless:
            result = await resolve_and_route(
                task,
                paperless=paperless,
                tags=SAMPLE_TAGS,
                correspondents=SAMPLE_CORRESPONDENTS,
                document_types=SAMPLE_DOC_TYPES,
                existing_doc_tag_ids=[5, 6],
                force_queue=True,
            )

        assert result.effective_action == GateAction.QUEUE_FOR_REVIEW
        assert result.applied is False

    async def test_force_queue_still_resolves_names(self):
        """Names should be resolved even when queuing."""
        task = _make_task(
            tag_names=[("invoice", 0.9), ("utility-bill", 0.8)],
            correspondent="AT&T",
            corr_confidence=0.9,
        )

        async with PaperlessClient("http://localhost:0", "fake") as paperless:
            result = await resolve_and_route(
                task,
                paperless=paperless,
                tags=SAMPLE_TAGS,
                correspondents=SAMPLE_CORRESPONDENTS,
                document_types=SAMPLE_DOC_TYPES,
                existing_doc_tag_ids=[],
                force_queue=True,
            )

        assert result.proposed_update.add_tag_ids == [1, 2]
        assert result.proposed_update.set_correspondent_id == 10
        assert result.applied is False

    async def test_unresolved_names_excluded_when_queuing(self):
        """Tags that don't match existing ones are excluded (not created) when queuing."""
        task = _make_task(
            tag_names=[("invoice", 0.9), ("new-tag", 0.7)],
        )

        async with PaperlessClient("http://localhost:0", "fake") as paperless:
            result = await resolve_and_route(
                task,
                paperless=paperless,
                tags=SAMPLE_TAGS,
                correspondents=SAMPLE_CORRESPONDENTS,
                document_types=SAMPLE_DOC_TYPES,
                existing_doc_tag_ids=[],
                force_queue=True,
            )

        # Only "invoice" resolves; "new-tag" is unresolved and excluded
        assert result.proposed_update.add_tag_ids == [1]
        assert result.applied is False


# --- Integration test (requires Ollama + Paperless) ---


@pytest.mark.slow
@pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)
async def test_resolve_and_route_live():
    """Full pipeline: fetch doc, tag via LLM, route with force_queue=True."""
    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL) as ollama,
    ):
        models = await ollama.list_models()
        model_name = pick_instruct_model(models)
        if not model_name:
            pytest.skip("No models available on Ollama server")

        docs, _count = await paperless.list_documents(page_size=1)
        if not docs:
            pytest.skip("No documents in Paperless instance")
        doc = docs[0]

        tags = await paperless.list_tags()
        correspondents = await paperless.list_correspondents()
        doc_types = await paperless.list_document_types()

        # Step 1: Executor tags the document
        task, _raw = await tag_document(
            doc,
            ollama=ollama,
            model=model_name,
            tags=tags,
            correspondents=correspondents,
            document_types=doc_types,
            keep_alive="0",
        )

        # Step 2: Router resolves and routes (force_queue — no writes to Paperless)
        result = await resolve_and_route(
            task,
            paperless=paperless,
            tags=tags,
            correspondents=correspondents,
            document_types=doc_types,
            existing_doc_tag_ids=doc.tags,
            force_queue=True,
        )

        assert isinstance(result, RoutingResult)
        assert result.applied is False
        assert result.effective_action == GateAction.QUEUE_FOR_REVIEW
        assert result.proposed_update.document_id == doc.id
        assert result.task.document_id == doc.id
