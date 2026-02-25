"""Reusable pipeline handler functions for the Corvus orchestrator.

Each handler encapsulates a complete pipeline and returns a typed result.
CLI commands and the orchestrator router both call these handlers.
Progress output is delegated via an optional callback.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from corvus.integrations.ollama import OllamaClient
from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.orchestrator import (
    DigestResult,
    FetchPipelineResult,
    StatusResult,
    TagPipelineResult,
    WebSearchResult,
    WebSearchSource,
)

logger = logging.getLogger(__name__)


async def run_tag_pipeline(
    *,
    paperless: PaperlessClient,
    ollama: OllamaClient,
    model: str,
    limit: int = 0,
    include_tagged: bool = False,
    keep_alive: str = "5m",
    force_queue: bool = True,
    queue_db_path: str,
    audit_log_path: str,
    on_progress: Callable[[str], None] | None = None,
) -> TagPipelineResult:
    """Run the document tagging pipeline.

    Fetches documents from Paperless, tags each via LLM, and routes
    through the confidence gate.

    Args:
        paperless: An open PaperlessClient.
        ollama: An open OllamaClient.
        model: Ollama model name.
        limit: Max documents to process (0 = all).
        include_tagged: Include already-tagged documents.
        keep_alive: Ollama keep_alive duration.
        force_queue: Queue all items for review (initial safe posture).
        queue_db_path: Path to the review queue SQLite database.
        audit_log_path: Path to the audit log file.
        on_progress: Optional callback for progress messages.

    Returns:
        TagPipelineResult with counts and details.
    """
    from corvus.audit.logger import AuditLog
    from corvus.executors.document_tagger import tag_document
    from corvus.queue.review import ReviewQueue
    from corvus.router.tagging import resolve_and_route
    from corvus.schemas.document_tagging import GateAction

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    audit_log = AuditLog(audit_log_path)

    # Fetch metadata for LLM context
    tags = await paperless.list_tags()
    correspondents = await paperless.list_correspondents()
    doc_types = await paperless.list_document_types()

    # Fetch documents page by page
    filter_params = None if include_tagged else {"tags__isnull": True}
    page = 1
    page_size = 25
    processed = 0
    queued = 0
    auto_applied = 0
    errors = 0
    details: list[str] = []
    total_count: int | None = None

    with ReviewQueue(queue_db_path) as review_queue:
        while True:
            docs, count = await paperless.list_documents(
                page=page,
                page_size=page_size,
                filter_params=filter_params,
            )
            if total_count is None:
                total_count = count
                if total_count == 0:
                    _emit("No documents to process.")
                    return TagPipelineResult(
                        processed=0, queued=0, auto_applied=0, errors=0,
                    )
                effective_limit = limit if limit > 0 else total_count
                _emit(f"Found {total_count} document(s). Processing up to {effective_limit}.")

            if not docs:
                break

            for doc in docs:
                if limit > 0 and (processed + errors) >= limit:
                    break

                try:
                    _emit(f"\n[{processed + errors + 1}] {doc.title} (id={doc.id})")

                    task, _raw = await tag_document(
                        doc,
                        ollama=ollama,
                        model=model,
                        tags=tags,
                        correspondents=correspondents,
                        document_types=doc_types,
                        keep_alive=keep_alive,
                    )

                    routing_result = await resolve_and_route(
                        task,
                        paperless=paperless,
                        tags=tags,
                        correspondents=correspondents,
                        document_types=doc_types,
                        existing_doc_tag_ids=doc.tags,
                        force_queue=force_queue,
                    )

                    # Queue or audit based on routing
                    if routing_result.effective_action == GateAction.QUEUE_FOR_REVIEW:
                        review_queue.add(routing_result.task, routing_result.proposed_update)
                        audit_log.log_queued_for_review(
                            routing_result.task, routing_result.proposed_update
                        )
                        queued += 1
                    elif routing_result.applied:
                        audit_log.log_auto_applied(
                            routing_result.task, routing_result.proposed_update
                        )
                        auto_applied += 1

                    # Summary line
                    tag_names = [t.tag_name for t in task.result.suggested_tags]
                    detail = (
                        f"confidence={task.overall_confidence:.0%} "
                        f"gate={routing_result.effective_action.value} "
                        f"tags={tag_names}"
                    )
                    _emit(f"  {detail}")
                    details.append(f"[{doc.id}] {doc.title}: {detail}")

                    if task.result.suggested_correspondent:
                        _emit(f"  correspondent={task.result.suggested_correspondent}")
                    if task.result.suggested_document_type:
                        _emit(f"  type={task.result.suggested_document_type}")

                    processed += 1

                except Exception:
                    errors += 1
                    logger.exception("Error processing document %d: %s", doc.id, doc.title)
                    _emit("  ERROR: Failed to process (see log for details)")

            if limit > 0 and (processed + errors) >= limit:
                break
            # Stop if we've fetched all available pages
            if len(docs) < page_size:
                break
            page += 1

    _emit(f"\nDone. Processed: {processed}, Errors: {errors}")

    return TagPipelineResult(
        processed=processed,
        queued=queued,
        auto_applied=auto_applied,
        errors=errors,
        details=details,
    )


async def run_fetch_pipeline(
    *,
    paperless: PaperlessClient,
    ollama: OllamaClient,
    model: str,
    query: str,
    keep_alive: str = "5m",
    on_progress: Callable[[str], None] | None = None,
) -> FetchPipelineResult:
    """Run the document fetch/retrieval pipeline.

    Interprets a natural language query via LLM, resolves search params,
    and returns matching documents. Stops before interactive selection —
    UI concerns stay in the caller.

    Args:
        paperless: An open PaperlessClient.
        ollama: An open OllamaClient.
        model: Ollama model name.
        query: Natural language search query.
        keep_alive: Ollama keep_alive duration.
        on_progress: Optional callback for progress messages.

    Returns:
        FetchPipelineResult with document list and warnings.
    """
    from corvus.executors.query_interpreter import interpret_query
    from corvus.router.retrieval import resolve_and_search

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Fetch metadata for LLM context
    tags = await paperless.list_tags()
    correspondents = await paperless.list_correspondents()
    doc_types = await paperless.list_document_types()

    # Step 1: Interpret query via LLM
    _emit(f'Interpreting: "{query}"')
    interpretation, _raw = await interpret_query(
        query,
        ollama=ollama,
        model=model,
        tags=tags,
        correspondents=correspondents,
        document_types=doc_types,
        keep_alive=keep_alive,
    )

    _emit(f"  Confidence: {interpretation.confidence:.0%}")
    _emit(f"  Reasoning: {interpretation.reasoning}")
    if interpretation.correspondent_name:
        _emit(f"  Correspondent: {interpretation.correspondent_name}")
    if interpretation.document_type_name:
        _emit(f"  Document type: {interpretation.document_type_name}")
    if interpretation.tag_names:
        _emit(f"  Tags: {interpretation.tag_names}")
    if interpretation.text_search:
        _emit(f"  Text search: {interpretation.text_search}")

    # Step 2: Resolve and search
    params, docs, total = await resolve_and_search(
        interpretation,
        paperless=paperless,
        tags=tags,
        correspondents=correspondents,
        document_types=doc_types,
    )

    warnings = list(params.warnings)
    for warning in warnings:
        _emit(f"  Warning: {warning}")

    if params.used_fallback:
        _emit("  Note: Structured filters returned no results; showing results from relaxed search.")

    # Build lookup dicts from already-fetched metadata
    tag_map = {t.id: t.name for t in tags}
    corr_map = {c.id: c.name for c in correspondents}
    dtype_map = {d.id: d.name for d in doc_types}

    doc_dicts = [
        {
            "id": d.id,
            "title": d.title,
            "created": d.created,
            "correspondent": corr_map.get(d.correspondent) if d.correspondent else None,
            "document_type": dtype_map.get(d.document_type) if d.document_type else None,
            "tags": [tag_map.get(tid, f"#{tid}") for tid in d.tags] if d.tags else [],
        }
        for d in docs
    ]

    return FetchPipelineResult(
        documents_found=total,
        documents=doc_dicts,
        warnings=warnings,
        used_fallback=params.used_fallback,
        interpretation_confidence=interpretation.confidence,
    )


def run_status_pipeline(
    *,
    queue_db_path: str,
    audit_log_path: str,
) -> StatusResult:
    """Run the status pipeline (pure Python, no LLM).

    Args:
        queue_db_path: Path to the review queue SQLite database.
        audit_log_path: Path to the audit log file.

    Returns:
        StatusResult with pending/processed/reviewed counts.
    """
    from corvus.audit.logger import AuditLog
    from corvus.queue.review import ReviewQueue

    audit_log = AuditLog(audit_log_path)
    with ReviewQueue(queue_db_path) as review_queue:
        pending_count = review_queue.count_pending()

    since = datetime.now(UTC) - timedelta(hours=24)
    entries = audit_log.read_entries(since=since)

    processed = sum(1 for e in entries if e.action in ("auto_applied", "queued_for_review"))
    reviewed = sum(1 for e in entries if e.action in ("review_approved", "review_rejected"))

    return StatusResult(
        pending_count=pending_count,
        processed_24h=processed,
        reviewed_24h=reviewed,
    )


def run_digest_pipeline(
    *,
    queue_db_path: str,
    audit_log_path: str,
    hours: int = 24,
) -> DigestResult:
    """Run the digest pipeline (pure Python, no LLM).

    Args:
        queue_db_path: Path to the review queue SQLite database.
        audit_log_path: Path to the audit log file.
        hours: Lookback period in hours.

    Returns:
        DigestResult with rendered text.
    """
    from corvus.audit.logger import AuditLog
    from corvus.digest.daily import generate_digest, render_text
    from corvus.queue.review import ReviewQueue

    audit_log = AuditLog(audit_log_path)
    with ReviewQueue(queue_db_path) as review_queue:
        d = generate_digest(audit_log, review_queue, hours=hours)

    return DigestResult(rendered_text=render_text(d))


async def run_search_pipeline(
    *,
    ollama: OllamaClient,
    model: str,
    query: str,
    keep_alive: str = "5m",
    max_results: int = 5,
    fetch_pages: int = 2,
    page_max_chars: int = 8000,
    fetch_timeout: int = 10,
    on_progress: Callable[[str], None] | None = None,
) -> WebSearchResult:
    """Run the web search pipeline.

    Searches DuckDuckGo, optionally fetches page content for richer context,
    then uses an LLM to summarize the results with source citations.

    Args:
        ollama: An open OllamaClient instance.
        model: Ollama model name.
        query: Search query string.
        keep_alive: Ollama keep_alive duration.
        max_results: Maximum number of search results.
        fetch_pages: Number of top results to fetch page content for (0 to skip).
        page_max_chars: Maximum characters of extracted text per page.
        fetch_timeout: HTTP timeout in seconds for page fetches.
        on_progress: Optional callback for progress messages.

    Returns:
        WebSearchResult with summary, sources, and query.
    """
    from corvus.integrations.search import SearchError, fetch_page_content, web_search

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _emit(f'Searching: "{query}"')

    # Step 1: Web search
    try:
        results = await web_search(query, max_results=max_results)
    except SearchError:
        logger.exception("Web search failed for query: %s", query)
        return await _search_fallback_chat(
            ollama=ollama, model=model, query=query, keep_alive=keep_alive,
        )

    if not results:
        _emit("No search results found, falling back to LLM.")
        return await _search_fallback_chat(
            ollama=ollama, model=model, query=query, keep_alive=keep_alive,
        )

    # Step 1.5: Fetch page content for top results
    if fetch_pages > 0:
        _emit("Fetching page content...")
        results = await fetch_page_content(
            results,
            max_pages=fetch_pages,
            max_chars_per_page=page_max_chars,
            timeout=fetch_timeout,
        )
        pages_fetched = sum(1 for r in results if r.page_content)
        if pages_fetched:
            _emit(f"Fetched content from {pages_fetched} page(s).")

    # Step 2: Build context for LLM summarization
    context_lines = []
    for i, r in enumerate(results, 1):
        context_lines.append(f"[{i}] {r.title}")
        context_lines.append(f"    URL: {r.url}")
        context_lines.append(f"    Snippet: {r.snippet}")
        if r.page_content:
            context_lines.append(f"    Page content:\n{r.page_content}")
    context = "\n".join(context_lines)

    _emit(f"Found {len(results)} result(s), summarizing...")

    system_prompt = (
        "You are Corvus, a helpful AI assistant. Your job is to answer the user's "
        "question using ONLY the search results provided below. Some results include "
        "extracted page content in addition to snippets.\n\n"
        "Rules:\n"
        "- Extract and state specific facts, numbers, and data from the results.\n"
        "- Prefer data from page content over snippets when both are available.\n"
        "- Give a direct answer first, then add context if needed.\n"
        "- Cite sources by number in square brackets (e.g. [1], [2]).\n"
        "- If the results don't contain enough data to fully answer, say what you "
        "found and note what's missing — do NOT tell the user to visit the websites.\n"
        "- Never invent data not present in the results.\n"
        "- Be concise: 2-4 sentences max."
    )
    user_prompt = f"Question: {query}\n\nSearch results:\n{context}"

    text, _raw = await ollama.chat(
        model=model,
        system=system_prompt,
        prompt=user_prompt,
        keep_alive=keep_alive,
        temperature=0.3,
    )

    sources = [
        WebSearchSource(title=r.title, url=r.url, snippet=r.snippet)
        for r in results
    ]

    return WebSearchResult(summary=text, sources=sources, query=query)


async def _search_fallback_chat(
    *,
    ollama: OllamaClient,
    model: str,
    query: str,
    keep_alive: str,
) -> WebSearchResult:
    """Fallback when web search is unavailable — LLM-only answer with disclaimer.

    Returns:
        WebSearchResult with LLM answer and no sources.
    """
    system_prompt = (
        "You are Corvus, a helpful AI assistant. Web search is currently unavailable. "
        "Answer the user's question to the best of your ability based on your training "
        "data. Start your response with a brief note that search results were unavailable."
    )

    text, _raw = await ollama.chat(
        model=model,
        system=system_prompt,
        prompt=query,
        keep_alive=keep_alive,
    )

    return WebSearchResult(summary=text, sources=[], query=query)
