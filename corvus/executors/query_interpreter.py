"""Query interpreter executor — parses natural language into search params via LLM.

Stateless: receives all inputs (query, available metadata, clients),
returns a QueryInterpretation. No side effects.
"""

import logging
from datetime import UTC, datetime

from corvus.integrations.ollama import OllamaClient, OllamaResponse
from corvus.schemas.document_retrieval import QueryInterpretation
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocumentType,
    PaperlessTag,
)

logger = logging.getLogger(__name__)


def _has_search_fields(interp: QueryInterpretation) -> bool:
    """Check if the interpretation has any usable search fields."""
    return bool(
        interp.text_search
        or interp.correspondent_name
        or interp.document_type_name
        or interp.tag_names
    )


def _strip_today_only_date_range(interp: QueryInterpretation) -> None:
    """Strip date range when both start and end equal today.

    This is almost certainly a misinterpretation of "latest"/"most recent"
    — the LLM set today's date instead of leaving dates null.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if interp.date_range_start == today and interp.date_range_end == today:
        logger.warning(
            "Stripping today-only date range (likely 'most recent' misinterpretation)"
        )
        interp.date_range_start = None
        interp.date_range_end = None

SYSTEM_PROMPT = """\
You are a document search assistant for a personal document management system.

Your job is to interpret a natural language search query and extract structured \
search parameters. The user wants to find a specific document or set of documents.

## Rules
1. Only extract information that is explicitly mentioned or clearly implied in the query.
2. Reuse existing correspondent, document type, and tag names when they match. \
Use the EXACT name from the available lists below.
3. If the query mentions a company/person, set correspondent_name.
4. If the query mentions a document category (invoice, receipt, statement, etc.), \
set document_type_name.
5. Use text_search for keywords that don't map to correspondent/type/tag.
6. Always include the most important search terms in text_search, even if you also \
extract them as correspondent/type/tag. The text_search field is used for full-text \
search and should capture the core intent of the query.
7. Convert relative dates to absolute ISO dates (YYYY-MM-DD). Today is {today}.
8. Default sort_order to "newest" unless the query implies otherwise.
9. Assign a confidence score (0.0-1.0). Use lower scores when the query is vague.
10. Provide brief reasoning explaining your interpretation.
11. "most recent", "latest", or "newest" means sort_order: "newest" — do NOT set \
date ranges for these terms. Only set date ranges when the query mentions a specific \
date or time period (e.g. "from last month", "in 2022", "since January").

## Available Correspondents
{correspondents}

## Available Document Types
{document_types}

## Available Tags
{tags}

## Examples

Query: "latest mortgage statement"
→ text_search: "mortgage statement", document_type_name: "Statement", \
sort_order: "newest", date_range_start: null, date_range_end: null

Query: "AT&T invoice from last month"
→ text_search: "AT&T invoice", correspondent_name: "AT&T", document_type_name: "Invoice"

Query: "tax documents for 2022"
→ text_search: "tax 2022", tag_names: ["tax-return"], date_range_start: "2022-01-01", \
date_range_end: "2022-12-31"
"""

USER_PROMPT = """\
Find documents matching this query:

{query}
"""


def _build_system_prompt(
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
) -> str:
    tag_list = ", ".join(t.name for t in tags) if tags else "(none yet)"
    corr_list = ", ".join(c.name for c in correspondents) if correspondents else "(none yet)"
    dtype_list = ", ".join(d.name for d in document_types) if document_types else "(none yet)"

    return SYSTEM_PROMPT.format(
        today=datetime.now(UTC).strftime("%Y-%m-%d"),
        tags=tag_list,
        correspondents=corr_list,
        document_types=dtype_list,
    )


async def interpret_query(
    query: str,
    *,
    ollama: OllamaClient,
    model: str,
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
    keep_alive: str | None = None,
) -> tuple[QueryInterpretation, OllamaResponse]:
    """Interpret a natural language search query using the LLM.

    Args:
        query: Natural language query (e.g. "most recent invoice from AT&T").
        ollama: An open OllamaClient instance.
        model: Ollama model name to use for inference.
        tags: All existing Paperless tags (for context).
        correspondents: All existing Paperless correspondents (for context).
        document_types: All existing Paperless document types (for context).
        keep_alive: Ollama keep_alive parameter.

    Returns:
        Tuple of (QueryInterpretation, OllamaResponse).
    """
    system = _build_system_prompt(tags, correspondents, document_types)
    prompt = USER_PROMPT.format(query=query)

    logger.info("Interpreting query: %s", query)

    interpretation, raw = await ollama.generate_structured(
        model=model,
        schema_class=QueryInterpretation,
        system=system,
        prompt=prompt,
        keep_alive=keep_alive,
    )

    if not _has_search_fields(interpretation):
        logger.warning(
            "LLM returned empty search fields (confidence=%.2f); "
            "injecting original query as text_search fallback",
            interpretation.confidence,
        )
        interpretation.text_search = query

    _strip_today_only_date_range(interpretation)

    logger.info(
        "Query interpreted: correspondent=%s type=%s tags=%s text=%s confidence=%.2f",
        interpretation.correspondent_name,
        interpretation.document_type_name,
        interpretation.tag_names,
        interpretation.text_search,
        interpretation.confidence,
    )

    return interpretation, raw
