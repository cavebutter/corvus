"""CLI entry point for the Corvus document tagging pipeline.

Commands:
    corvus tag      — batch-tag documents via LLM
    corvus review   — interactively review pending items
    corvus digest   — show activity digest
    corvus status   — quick overview of queue and recent activity
"""

import asyncio
import logging
import sys

import click

from corvus.config import (
    AUDIT_LOG_PATH,
    OLLAMA_BASE_URL,
    PAPERLESS_API_TOKEN,
    PAPERLESS_BASE_URL,
    QUEUE_DB_PATH,
)

logger = logging.getLogger("corvus")


def _validate_config() -> None:
    """Fail loudly if required config is missing."""
    missing = []
    if not PAPERLESS_BASE_URL:
        missing.append("PAPERLESS_BASE_URL")
    if not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder":
        missing.append("PAPERLESS_API_TOKEN")
    if missing:
        click.echo(f"Error: Missing required config: {', '.join(missing)}", err=True)
        click.echo("Set these in secrets/internal.env or via SOPS.", err=True)
        sys.exit(1)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Corvus — local document tagging pipeline for Paperless-ngx."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# corvus tag
# ------------------------------------------------------------------


@cli.command()
@click.option("--limit", "-n", default=0, show_default=True, help="Max documents to process (0=all).")
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option("--all", "include_tagged", is_flag=True, help="Include already-tagged documents.")
@click.option("--keep-alive", default="5m", show_default=True, help="Ollama keep_alive duration.")
@click.option(
    "--force-queue/--no-force-queue",
    default=True,
    show_default=True,
    help="Queue all items for review (initial safe posture).",
)
def tag(limit: int, model: str | None, include_tagged: bool, keep_alive: str, force_queue: bool) -> None:
    """Batch-tag documents from Paperless-ngx via LLM."""
    _validate_config()
    asyncio.run(_tag_async(limit, model, include_tagged, keep_alive, force_queue))


async def _tag_async(
    limit: int,
    model: str | None,
    include_tagged: bool,
    keep_alive: str,
    force_queue: bool,
) -> None:
    from corvus.audit.logger import AuditLog
    from corvus.executors.document_tagger import tag_document
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.queue.review import ReviewQueue
    from corvus.router.tagging import resolve_and_route
    from corvus.schemas.document_tagging import GateAction

    audit_log = AuditLog(AUDIT_LOG_PATH)

    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
    ):
        # --- Resolve model ---
        if model is None:
            model = await ollama.pick_instruct_model()
            if model is None:
                click.echo("Error: No models available on Ollama server.", err=True)
                sys.exit(1)
            click.echo(f"Auto-selected model: {model}")

        # --- Fetch metadata for LLM context ---
        tags = await paperless.list_tags()
        correspondents = await paperless.list_correspondents()
        doc_types = await paperless.list_document_types()

        # --- Fetch documents page by page ---
        filter_params = None if include_tagged else {"tags__isnull": True}
        page = 1
        page_size = 25
        processed = 0
        errors = 0
        total_count: int | None = None

        with ReviewQueue(QUEUE_DB_PATH) as review_queue:
            while True:
                docs, count = await paperless.list_documents(
                    page=page,
                    page_size=page_size,
                    filter_params=filter_params,
                )
                if total_count is None:
                    total_count = count
                    if total_count == 0:
                        click.echo("No documents to process.")
                        return
                    effective_limit = limit if limit > 0 else total_count
                    click.echo(f"Found {total_count} document(s). Processing up to {effective_limit}.")

                if not docs:
                    break

                for doc in docs:
                    if limit > 0 and (processed + errors) >= limit:
                        break

                    try:
                        click.echo(f"\n[{processed + errors + 1}] {doc.title} (id={doc.id})")

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
                        elif routing_result.applied:
                            audit_log.log_auto_applied(
                                routing_result.task, routing_result.proposed_update
                            )

                        # Summary line
                        tag_names = [t.tag_name for t in task.result.suggested_tags]
                        click.echo(
                            f"  confidence={task.overall_confidence:.0%} "
                            f"gate={routing_result.effective_action.value} "
                            f"tags={tag_names}"
                        )
                        if task.result.suggested_correspondent:
                            click.echo(f"  correspondent={task.result.suggested_correspondent}")
                        if task.result.suggested_document_type:
                            click.echo(f"  type={task.result.suggested_document_type}")

                        processed += 1

                    except Exception:
                        errors += 1
                        logger.exception("Error processing document %d: %s", doc.id, doc.title)
                        click.echo(f"  ERROR: Failed to process (see log for details)", err=True)

                if limit > 0 and (processed + errors) >= limit:
                    break
                page += 1

        # --- Summary ---
        click.echo(f"\nDone. Processed: {processed}, Errors: {errors}")


# ------------------------------------------------------------------
# corvus review
# ------------------------------------------------------------------


@cli.command()
def review() -> None:
    """Interactively review pending items in the queue."""
    _validate_config()
    asyncio.run(_review_async())


async def _review_async() -> None:
    from corvus.audit.logger import AuditLog
    from corvus.integrations.paperless import PaperlessClient
    from corvus.queue.review import ReviewQueue
    from corvus.router.tagging import apply_approved_update

    audit_log = AuditLog(AUDIT_LOG_PATH)

    with ReviewQueue(QUEUE_DB_PATH) as review_queue:
        pending = review_queue.list_pending()
        if not pending:
            click.echo("No pending items to review.")
            return

        click.echo(f"Found {len(pending)} pending item(s).\n")

        approved = 0
        rejected = 0
        skipped = 0

        async with PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless:
            for i, item in enumerate(pending, 1):
                task = item.task
                click.echo(f"--- [{i}/{len(pending)}] Document {task.document_id}: {task.document_title} ---")
                click.echo(f"  Confidence: {task.overall_confidence:.0%}")
                click.echo(f"  Tags: {[t.tag_name for t in task.result.suggested_tags]}")
                if task.result.suggested_correspondent:
                    click.echo(f"  Correspondent: {task.result.suggested_correspondent}")
                if task.result.suggested_document_type:
                    click.echo(f"  Type: {task.result.suggested_document_type}")
                click.echo(f"  Reasoning: {task.result.reasoning}")
                snippet = task.content_snippet
                if snippet:
                    click.echo(f"  Snippet: {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

                choice = click.prompt(
                    "  Action",
                    type=click.Choice(["a", "r", "s", "q"], case_sensitive=False),
                    prompt_suffix=" [a]pprove / [r]eject / [s]kip / [q]uit: ",
                )

                if choice == "a":
                    try:
                        result = await apply_approved_update(item, paperless=paperless)
                        review_queue.approve(item.id, notes="Approved via CLI")
                        audit_log.log_review_approved(item.task, result.proposed_update)
                        click.echo("  -> Approved and applied to Paperless.")
                        approved += 1
                    except Exception:
                        logger.exception("Error applying update for document %d", task.document_id)
                        click.echo("  -> ERROR: Failed to apply. Item remains pending.", err=True)
                elif choice == "r":
                    notes = click.prompt("  Rejection notes (optional)", default="", show_default=False)
                    review_queue.reject(item.id, notes=notes or None)
                    audit_log.log_review_rejected(item.task, item.proposed_update)
                    click.echo("  -> Rejected.")
                    rejected += 1
                elif choice == "s":
                    click.echo("  -> Skipped.")
                    skipped += 1
                elif choice == "q":
                    click.echo("  -> Quitting review.")
                    break

        click.echo(f"\nReview complete. Approved: {approved}, Rejected: {rejected}, Skipped: {skipped}")


# ------------------------------------------------------------------
# corvus digest
# ------------------------------------------------------------------


@cli.command()
@click.option("--hours", default=24, show_default=True, help="Lookback period in hours.")
def digest(hours: int) -> None:
    """Show activity digest for the recent period."""
    from corvus.audit.logger import AuditLog
    from corvus.digest.daily import generate_digest, render_text
    from corvus.queue.review import ReviewQueue

    audit_log = AuditLog(AUDIT_LOG_PATH)
    with ReviewQueue(QUEUE_DB_PATH) as review_queue:
        d = generate_digest(audit_log, review_queue, hours=hours)
        click.echo(render_text(d))


# ------------------------------------------------------------------
# corvus status
# ------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Quick overview of queue and recent activity."""
    from datetime import UTC, datetime, timedelta

    from corvus.audit.logger import AuditLog
    from corvus.queue.review import ReviewQueue

    audit_log = AuditLog(AUDIT_LOG_PATH)
    with ReviewQueue(QUEUE_DB_PATH) as review_queue:
        pending_count = review_queue.count_pending()

        since = datetime.now(UTC) - timedelta(hours=24)
        entries = audit_log.read_entries(since=since)

        processed = sum(1 for e in entries if e.action in ("auto_applied", "queued_for_review"))
        reviewed = sum(1 for e in entries if e.action in ("review_approved", "review_rejected"))

        click.echo("Corvus Status")
        click.echo(f"  Pending review:     {pending_count}")
        click.echo(f"  Processed (24h):    {processed}")
        click.echo(f"  Reviewed (24h):     {reviewed}")
