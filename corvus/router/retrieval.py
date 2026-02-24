"""Deterministic router for the document retrieval pipeline.

Receives a QueryInterpretation from the LLM executor, resolves names
to Paperless IDs, builds filter params, and searches via the Paperless API.

No LLM calls — pure Python logic.
"""

import logging

from corvus.integrations.paperless import PaperlessClient
from corvus.router.tagging import (
    resolve_correspondent,
    resolve_document_type,
    resolve_tag,
)
from corvus.schemas.document_retrieval import QueryInterpretation, ResolvedSearchParams
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocument,
    PaperlessDocumentType,
    PaperlessTag,
)

logger = logging.getLogger(__name__)


def resolve_search_params(
    interpretation: QueryInterpretation,
    *,
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
) -> ResolvedSearchParams:
    """Resolve LLM-suggested names to Paperless IDs and build search params.

    Args:
        interpretation: Structured query interpretation from the LLM.
        tags: All existing Paperless tags.
        correspondents: All existing Paperless correspondents.
        document_types: All existing Paperless document types.

    Returns:
        ResolvedSearchParams with IDs and any warnings about unresolved names.
    """
    warnings: list[str] = []
    text_parts: list[str] = []
    if interpretation.text_search:
        text_parts.append(interpretation.text_search)

    # Resolve correspondent
    correspondent_id: int | None = None
    if interpretation.correspondent_name:
        correspondent_id = resolve_correspondent(interpretation.correspondent_name, correspondents)
        if correspondent_id is None:
            text_parts.append(interpretation.correspondent_name)
            warnings.append(
                f"Correspondent not found (added to text search): '{interpretation.correspondent_name}'"
            )

    # Resolve document type
    document_type_id: int | None = None
    if interpretation.document_type_name:
        document_type_id = resolve_document_type(interpretation.document_type_name, document_types)
        if document_type_id is None:
            text_parts.append(interpretation.document_type_name)
            warnings.append(
                f"Document type not found (added to text search): '{interpretation.document_type_name}'"
            )

    # Resolve tags
    tag_ids: list[int] = []
    for tag_name in interpretation.tag_names:
        tag_id = resolve_tag(tag_name, tags)
        if tag_id is not None:
            tag_ids.append(tag_id)
        else:
            text_parts.append(tag_name)
            warnings.append(f"Tag not found (added to text search): '{tag_name}'")

    # Map sort order
    ordering: str | None = None
    if interpretation.sort_order == "newest":
        ordering = "-created"
    elif interpretation.sort_order == "oldest":
        ordering = "created"
    # "relevance" → None (use Paperless default/relevance ranking)

    text_search = " ".join(text_parts) if text_parts else None

    return ResolvedSearchParams(
        text_search=text_search,
        correspondent_id=correspondent_id,
        document_type_id=document_type_id,
        tag_ids=tag_ids,
        date_range_start=interpretation.date_range_start,
        date_range_end=interpretation.date_range_end,
        ordering=ordering,
        warnings=warnings,
    )


def build_filter_params(params: ResolvedSearchParams) -> dict:
    """Convert ResolvedSearchParams into a dict for PaperlessClient.list_documents().

    Args:
        params: Resolved search parameters with Paperless IDs.

    Returns:
        Dict of query parameters for the Paperless API.
    """
    filters: dict = {}

    if params.text_search:
        filters["query"] = params.text_search

    if params.correspondent_id is not None:
        filters["correspondent__id"] = params.correspondent_id

    if params.document_type_id is not None:
        filters["document_type__id"] = params.document_type_id

    if params.tag_ids:
        filters["tags__id__all"] = ",".join(str(tid) for tid in params.tag_ids)

    if params.date_range_start:
        filters["created__date__gte"] = params.date_range_start

    if params.date_range_end:
        filters["created__date__lte"] = params.date_range_end

    return filters


async def resolve_and_search(
    interpretation: QueryInterpretation,
    *,
    paperless: PaperlessClient,
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
    page_size: int = 25,
) -> tuple[ResolvedSearchParams, list[PaperlessDocument], int]:
    """Resolve names, build filters, and search Paperless.

    Args:
        interpretation: Structured query interpretation from the LLM.
        paperless: An open PaperlessClient.
        tags: All existing Paperless tags.
        correspondents: All existing Paperless correspondents.
        document_types: All existing Paperless document types.
        page_size: Number of results to fetch.

    Returns:
        Tuple of (ResolvedSearchParams, list of matching documents, total count).
    """
    params = resolve_search_params(
        interpretation,
        tags=tags,
        correspondents=correspondents,
        document_types=document_types,
    )

    filter_params = build_filter_params(params)
    ordering = params.ordering or "-created"

    logger.info("Searching Paperless with filters: %s ordering=%s", filter_params, ordering)

    docs, total = await paperless.list_documents(
        page_size=page_size,
        ordering=ordering,
        filter_params=filter_params,
    )

    logger.info("Search returned %d documents (total=%d)", len(docs), total)

    # Fallback cascade when structured filters produce 0 results
    has_structured_filters = (
        params.correspondent_id is not None
        or params.document_type_id is not None
        or params.tag_ids
    )
    if total == 0 and has_structured_filters:
        # Fallback 1: per-tag search (broadens AND → individual tag lookups)
        if params.tag_ids:
            for tag_id in params.tag_ids:
                tag_filters = {"tags__id__all": str(tag_id)}
                docs, total = await paperless.list_documents(
                    page_size=page_size,
                    ordering=ordering,
                    filter_params=tag_filters,
                )
                if total > 0:
                    params.used_fallback = True
                    logger.info(
                        "Tag fallback (tag_id=%d) returned %d documents (total=%d)",
                        tag_id, len(docs), total,
                    )
                    break

        # Fallback 2: text-only search (drop all structured filters)
        if total == 0 and params.text_search:
            logger.info("Falling back to text-only search")
            fallback_filters = build_filter_params(
                ResolvedSearchParams(text_search=params.text_search)
            )
            docs, total = await paperless.list_documents(
                page_size=page_size,
                ordering=ordering,
                filter_params=fallback_filters,
            )
            if total > 0:
                params.used_fallback = True
            logger.info("Text fallback returned %d documents (total=%d)", len(docs), total)

    # Fallback 3: title/content substring search (last resort, works for any query)
    if total == 0 and params.text_search:
        logger.info("Falling back to title/content substring search")
        docs, total = await paperless.list_documents(
            page_size=page_size,
            ordering=ordering,
            filter_params={"title_content": params.text_search},
        )
        if total > 0:
            params.used_fallback = True
        logger.info(
            "Title/content fallback returned %d documents (total=%d)", len(docs), total,
        )

    return params, docs, total
