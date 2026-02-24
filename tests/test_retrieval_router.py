"""Tests for the document retrieval router."""

from unittest.mock import AsyncMock

from corvus.router.retrieval import (
    build_filter_params,
    resolve_and_search,
    resolve_search_params,
)
from corvus.schemas.document_retrieval import QueryInterpretation, ResolvedSearchParams
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocument,
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


def _make_interpretation(**kwargs) -> QueryInterpretation:
    defaults = {
        "confidence": 0.9,
        "reasoning": "Test interpretation",
        "sort_order": "newest",
    }
    defaults.update(kwargs)
    return QueryInterpretation(**defaults)


def _make_document(doc_id: int = 1, title: str = "Test Doc") -> PaperlessDocument:
    return PaperlessDocument(
        id=doc_id,
        title=title,
        content="Some content",
        tags=[],
        created="2025-03-15T00:00:00Z",
        added="2025-03-15T00:00:00Z",
    )


# --- Unit tests: resolve_search_params ---

_RESOLVE_KWARGS = dict(
    tags=SAMPLE_TAGS,
    correspondents=SAMPLE_CORRESPONDENTS,
    document_types=SAMPLE_DOC_TYPES,
)


class TestResolveSearchParams:
    def test_resolves_correspondent(self):
        interp = _make_interpretation(correspondent_name="AT&T")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.correspondent_id == 10
        assert params.warnings == []

    def test_resolves_document_type(self):
        interp = _make_interpretation(document_type_name="Invoice")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.document_type_id == 20
        assert params.warnings == []

    def test_resolves_tags(self):
        interp = _make_interpretation(tag_names=["invoice", "utility-bill"])
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.tag_ids == [1, 2]
        assert params.warnings == []

    def test_unresolved_correspondent_warns(self):
        interp = _make_interpretation(correspondent_name="Verizon")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.correspondent_id is None
        assert len(params.warnings) == 1
        assert "Verizon" in params.warnings[0]
        assert "added to text search" in params.warnings[0]
        assert params.text_search == "Verizon"

    def test_unresolved_document_type_warns(self):
        interp = _make_interpretation(document_type_name="Receipt")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.document_type_id is None
        assert len(params.warnings) == 1
        assert "Receipt" in params.warnings[0]
        assert "added to text search" in params.warnings[0]
        assert params.text_search == "Receipt"

    def test_unresolved_tag_warns(self):
        interp = _make_interpretation(tag_names=["invoice", "nonexistent"])
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.tag_ids == [1]
        assert len(params.warnings) == 1
        assert "nonexistent" in params.warnings[0]
        assert "added to text search" in params.warnings[0]
        assert params.text_search == "nonexistent"

    def test_multiple_unresolved_warnings(self):
        interp = _make_interpretation(
            correspondent_name="Unknown Corp",
            document_type_name="Unknown Type",
            tag_names=["bad-tag"],
        )
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert len(params.warnings) == 3

    def test_unresolved_names_merged_with_text_search(self):
        interp = _make_interpretation(
            text_search="death certificate",
            correspondent_name="Hilda",
        )
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.correspondent_id is None
        assert params.text_search == "death certificate Hilda"
        assert len(params.warnings) == 1

    def test_only_unresolved_names_become_text_search(self):
        interp = _make_interpretation(correspondent_name="Hilda")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.correspondent_id is None
        assert params.text_search == "Hilda"

    def test_sort_newest(self):
        interp = _make_interpretation(sort_order="newest")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.ordering == "-created"

    def test_sort_oldest(self):
        interp = _make_interpretation(sort_order="oldest")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.ordering == "created"

    def test_sort_relevance(self):
        interp = _make_interpretation(sort_order="relevance")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.ordering is None

    def test_text_search_passthrough(self):
        interp = _make_interpretation(text_search="wireless bill")
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.text_search == "wireless bill"

    def test_date_range_passthrough(self):
        interp = _make_interpretation(
            date_range_start="2024-01-01",
            date_range_end="2024-12-31",
        )
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.date_range_start == "2024-01-01"
        assert params.date_range_end == "2024-12-31"

    def test_empty_interpretation(self):
        interp = _make_interpretation()
        params = resolve_search_params(interp, **_RESOLVE_KWARGS)
        assert params.correspondent_id is None
        assert params.document_type_id is None
        assert params.tag_ids == []
        assert params.text_search is None
        assert params.warnings == []


# --- Unit tests: build_filter_params ---


class TestBuildFilterParams:
    def test_text_search(self):
        params = ResolvedSearchParams(text_search="wireless bill")
        assert build_filter_params(params) == {"query": "wireless bill"}

    def test_correspondent_id(self):
        params = ResolvedSearchParams(correspondent_id=10)
        assert build_filter_params(params) == {"correspondent__id": 10}

    def test_document_type_id(self):
        params = ResolvedSearchParams(document_type_id=20)
        assert build_filter_params(params) == {"document_type__id": 20}

    def test_tag_ids(self):
        params = ResolvedSearchParams(tag_ids=[1, 2, 3])
        assert build_filter_params(params) == {"tags__id__all": "1,2,3"}

    def test_date_range(self):
        params = ResolvedSearchParams(
            date_range_start="2024-01-01",
            date_range_end="2024-12-31",
        )
        result = build_filter_params(params)
        assert result["created__date__gte"] == "2024-01-01"
        assert result["created__date__lte"] == "2024-12-31"

    def test_combined_filters(self):
        params = ResolvedSearchParams(
            text_search="wireless",
            correspondent_id=10,
            document_type_id=20,
            tag_ids=[1],
            date_range_start="2024-01-01",
        )
        result = build_filter_params(params)
        assert result == {
            "query": "wireless",
            "correspondent__id": 10,
            "document_type__id": 20,
            "tags__id__all": "1",
            "created__date__gte": "2024-01-01",
        }

    def test_empty_params(self):
        params = ResolvedSearchParams()
        assert build_filter_params(params) == {}

    def test_none_values_excluded(self):
        params = ResolvedSearchParams(
            text_search=None,
            correspondent_id=None,
            document_type_id=None,
            tag_ids=[],
        )
        assert build_filter_params(params) == {}


# --- Unit tests: resolve_and_search (mocked Paperless) ---


class TestResolveAndSearch:
    async def test_search_with_results(self):
        interp = _make_interpretation(
            correspondent_name="AT&T",
            document_type_name="Invoice",
        )
        docs = [_make_document(1, "AT&T Invoice"), _make_document(2, "AT&T Statement")]

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = (docs, 2)

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.correspondent_id == 10
        assert params.document_type_id == 20
        assert len(results) == 2
        assert total == 2

        # Verify the Paperless API was called with correct filters
        call_kwargs = mock_paperless.list_documents.call_args.kwargs
        assert call_kwargs["filter_params"]["correspondent__id"] == 10
        assert call_kwargs["filter_params"]["document_type__id"] == 20
        assert call_kwargs["ordering"] == "-created"

    async def test_search_no_results(self):
        interp = _make_interpretation(correspondent_name="AT&T")

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        _params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert results == []
        assert total == 0

    async def test_relevance_ordering(self):
        interp = _make_interpretation(sort_order="relevance", text_search="wireless")

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        # Relevance ordering falls back to -created
        call_kwargs = mock_paperless.list_documents.call_args.kwargs
        assert call_kwargs["ordering"] == "-created"

    async def test_warnings_propagated(self):
        interp = _make_interpretation(correspondent_name="Unknown Corp")

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        params, _, _ = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert len(params.warnings) == 1
        assert "Unknown Corp" in params.warnings[0]

    async def test_page_size_forwarded(self):
        interp = _make_interpretation()

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
            page_size=10,
        )

        call_kwargs = mock_paperless.list_documents.call_args.kwargs
        assert call_kwargs["page_size"] == 10

    async def test_fallback_per_tag_on_zero_results(self):
        """Tags (AND) + text → 0 results → per-tag fallback finds results."""
        interp = _make_interpretation(
            tag_names=["invoice", "utility-bill"],
            text_search="fy2022 taxes",
        )
        tag_docs = [_make_document(10, "Tax Return 2022")]

        mock_paperless = AsyncMock()
        # Call 1: full structured (tags AND + text) → 0
        # Call 2: per-tag fallback (tag_id=1) → results
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            (tag_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert results[0].title == "Tax Return 2022"
        assert mock_paperless.list_documents.call_count == 2

        # Verify per-tag call used single tag filter
        tag_call_kwargs = mock_paperless.list_documents.call_args_list[1].kwargs
        assert tag_call_kwargs["filter_params"] == {"tags__id__all": "1"}

    async def test_fallback_per_tag_tries_second_tag(self):
        """First tag returns 0, second tag finds results."""
        interp = _make_interpretation(
            tag_names=["invoice", "utility-bill"],
            text_search="some query",
        )
        tag_docs = [_make_document(11, "Utility Bill")]

        mock_paperless = AsyncMock()
        # Call 1: full structured → 0
        # Call 2: per-tag tag_id=1 → 0
        # Call 3: per-tag tag_id=2 → results
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            ([], 0),
            (tag_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert mock_paperless.list_documents.call_count == 3

        # Verify third call used second tag
        third_call_kwargs = mock_paperless.list_documents.call_args_list[2].kwargs
        assert third_call_kwargs["filter_params"] == {"tags__id__all": "2"}

    async def test_fallback_to_text_after_tags_exhausted(self):
        """Per-tag fallback returns 0 for all tags → text-only fallback finds results."""
        interp = _make_interpretation(
            document_type_name="Statement",
            tag_names=["invoice"],
            text_search="trust transfer Howard",
        )
        fallback_docs = [_make_document(5, "Trust Transfer Howard")]

        mock_paperless = AsyncMock()
        # Call 1: full structured (doc_type + tag + text) → 0
        # Call 2: per-tag (tag_id=1) → 0
        # Call 3: text-only → results
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            ([], 0),
            (fallback_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert results[0].title == "Trust Transfer Howard"
        assert mock_paperless.list_documents.call_count == 3

        # Verify third call used text-only filters
        text_call_kwargs = mock_paperless.list_documents.call_args_list[2].kwargs
        assert text_call_kwargs["filter_params"] == {"query": "trust transfer Howard"}

    async def test_fallback_text_only_no_tags(self):
        """Doc type + text → 0 results, no tags → skips per-tag, text-only fallback."""
        interp = _make_interpretation(
            document_type_name="Statement",
            text_search="trust transfer Howard",
        )
        fallback_docs = [_make_document(5, "Trust Transfer Howard")]

        mock_paperless = AsyncMock()
        # Call 1: full structured (doc_type + text) → 0
        # Call 2: text-only → results (per-tag skipped, no tags)
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            (fallback_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert mock_paperless.list_documents.call_count == 2

    async def test_no_fallback_when_results_found(self):
        """Structured filters + results → no fallback."""
        interp = _make_interpretation(
            correspondent_name="AT&T",
            text_search="wireless bill",
        )
        docs = [_make_document(1, "AT&T Wireless Bill")]

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = (docs, 1)

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is False
        assert total == 1
        assert mock_paperless.list_documents.call_count == 1

    async def test_no_fallback_without_text_or_tags(self):
        """Structured filters (no tags, no text) + 0 results → no fallback."""
        interp = _make_interpretation(
            correspondent_name="AT&T",
            document_type_name="Invoice",
        )

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is False
        assert total == 0
        assert mock_paperless.list_documents.call_count == 1

    async def test_fallback_title_content_for_text_only_query(self):
        """Text-only search → 0 → title_content fallback finds results."""
        interp = _make_interpretation(text_search="trust transfer")
        tc_docs = [_make_document(27, "Trust Transfer Howard")]

        mock_paperless = AsyncMock()
        # Call 1: full-text query → 0
        # Call 2: title_content → results
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            (tc_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert results[0].title == "Trust Transfer Howard"
        assert mock_paperless.list_documents.call_count == 2

        # Verify second call used title_content filter
        tc_call_kwargs = mock_paperless.list_documents.call_args_list[1].kwargs
        assert tc_call_kwargs["filter_params"] == {"title_content": "trust transfer"}

    async def test_fallback_title_content_after_structured_cascade(self):
        """All structured fallbacks fail → title_content finds results."""
        interp = _make_interpretation(
            document_type_name="Statement",
            tag_names=["invoice"],
            text_search="trust transfer Howard",
        )
        tc_docs = [_make_document(27, "Trust Transfer Howard")]

        mock_paperless = AsyncMock()
        # Call 1: full structured → 0
        # Call 2: per-tag (tag_id=1) → 0
        # Call 3: text-only → 0
        # Call 4: title_content → results
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            ([], 0),
            ([], 0),
            (tc_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert mock_paperless.list_documents.call_count == 4

        # Verify last call used title_content
        tc_call_kwargs = mock_paperless.list_documents.call_args_list[3].kwargs
        assert tc_call_kwargs["filter_params"] == {
            "title_content": "trust transfer Howard",
        }

    async def test_title_content_skipped_when_earlier_fallback_succeeds(self):
        """Per-tag fallback finds results → title_content not tried."""
        interp = _make_interpretation(
            tag_names=["invoice"],
            text_search="some query",
        )
        tag_docs = [_make_document(1, "Invoice Doc")]

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.side_effect = [
            ([], 0),
            (tag_docs, 1),
        ]

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is True
        assert total == 1
        assert mock_paperless.list_documents.call_count == 2

    async def test_all_fallbacks_exhausted(self):
        """All fallbacks return 0 → used_fallback stays False."""
        interp = _make_interpretation(text_search="nonexistent document")

        mock_paperless = AsyncMock()
        mock_paperless.list_documents.return_value = ([], 0)

        params, results, total = await resolve_and_search(
            interp,
            paperless=mock_paperless,
            tags=SAMPLE_TAGS,
            correspondents=SAMPLE_CORRESPONDENTS,
            document_types=SAMPLE_DOC_TYPES,
        )

        assert params.used_fallback is False
        assert total == 0
        # Call 1: full-text query, Call 2: title_content
        assert mock_paperless.list_documents.call_count == 2
