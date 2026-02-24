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
    WATCHDOG_AUDIT_LOG_PATH,
    WATCHDOG_CONSUME_DIR,
    WATCHDOG_FILE_PATTERNS,
    WATCHDOG_HASH_DB_PATH,
    WATCHDOG_SCAN_DIR,
    WATCHDOG_TRANSFER_METHOD,
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
                    type=click.Choice(["a", "e", "r", "s", "q"], case_sensitive=False),
                    prompt_suffix=" [a]pprove / [e]dit / [r]eject / [s]kip / [q]uit: ",
                )

                if choice in ("a", "e"):
                    extra_tag_names: list[str] = []
                    if choice == "e":
                        raw = click.prompt(
                            "  Add tags (comma-separated)", default="", show_default=False
                        )
                        extra_tag_names = [
                            t.strip() for t in raw.split(",") if t.strip()
                        ]
                        if not extra_tag_names:
                            click.echo("  (no tags entered, approving as-is)")

                    try:
                        result = await apply_approved_update(
                            item,
                            paperless=paperless,
                            extra_tag_names=extra_tag_names or None,
                        )
                        if extra_tag_names:
                            notes = f"Modified via CLI; added tags: {extra_tag_names}"
                            review_queue.modify(item.id, notes=notes)
                        else:
                            notes = "Approved via CLI"
                            review_queue.approve(item.id, notes=notes)
                        audit_log.log_review_approved(item.task, result.proposed_update)
                        if extra_tag_names:
                            click.echo(
                                f"  -> Approved with extra tags {extra_tag_names} and applied to Paperless."
                            )
                        else:
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


# ------------------------------------------------------------------
# corvus watch
# ------------------------------------------------------------------


def _validate_watchdog_config(
    scan_dir: str, method: str, consume_dir: str
) -> None:
    """Fail loudly if watchdog config is invalid."""
    from pathlib import Path

    if not scan_dir:
        click.echo("Error: --scan-dir is required (or set WATCHDOG_SCAN_DIR).", err=True)
        sys.exit(1)
    if not Path(scan_dir).is_dir():
        click.echo(f"Error: Scan directory does not exist: {scan_dir}", err=True)
        sys.exit(1)
    if method == "move":
        if not consume_dir:
            click.echo(
                "Error: --consume-dir is required for move method (or set WATCHDOG_CONSUME_DIR).",
                err=True,
            )
            sys.exit(1)
        if not Path(consume_dir).is_dir():
            click.echo(f"Error: Consume directory does not exist: {consume_dir}", err=True)
            sys.exit(1)
    elif method == "upload":
        missing_creds = (
            not PAPERLESS_BASE_URL
            or not PAPERLESS_API_TOKEN
            or PAPERLESS_API_TOKEN == "placeholder"
        )
        if missing_creds:
            click.echo(
                "Error: PAPERLESS_BASE_URL and PAPERLESS_API_TOKEN required for upload method.",
                err=True,
            )
            sys.exit(1)


@cli.command()
@click.option(
    "--scan-dir",
    default=WATCHDOG_SCAN_DIR,
    show_default=True,
    help="Directory to watch for new files.",
)
@click.option(
    "--method",
    type=click.Choice(["move", "upload"], case_sensitive=False),
    default=WATCHDOG_TRANSFER_METHOD,
    show_default=True,
    help="Transfer method: move to consume dir or upload via API.",
)
@click.option(
    "--consume-dir",
    default=WATCHDOG_CONSUME_DIR,
    show_default=True,
    help="Paperless consume directory (required for move method).",
)
@click.option(
    "--patterns",
    default=WATCHDOG_FILE_PATTERNS,
    show_default=True,
    help="Comma-separated file patterns to watch (e.g. '*.pdf,*.png').",
)
@click.option("--once", is_flag=True, help="Scan existing files and exit (no continuous watch).")
def watch(scan_dir: str, method: str, consume_dir: str, patterns: str, once: bool) -> None:
    """Watch a scan folder and transfer new files to Paperless-ngx."""
    _validate_watchdog_config(scan_dir, method, consume_dir)
    asyncio.run(_watch_async(scan_dir, method, consume_dir, patterns, once))


async def _watch_async(
    scan_dir: str, method: str, consume_dir: str, patterns: str, once: bool
) -> None:
    from pathlib import Path

    from corvus.schemas.watchdog import TransferMethod
    from corvus.watchdog.audit import WatchdogAuditLog
    from corvus.watchdog.hash_store import HashStore
    from corvus.watchdog.watcher import scan_existing, watch_folder

    scan_path = Path(scan_dir)
    file_patterns = [p.strip() for p in patterns.split(",") if p.strip()]
    transfer_method = TransferMethod(method)
    consume_path = Path(consume_dir) if consume_dir else None

    audit_log = WatchdogAuditLog(WATCHDOG_AUDIT_LOG_PATH)

    with HashStore(WATCHDOG_HASH_DB_PATH) as hash_store:
        paperless_client = None
        try:
            if transfer_method == TransferMethod.UPLOAD:
                from corvus.integrations.paperless import PaperlessClient

                paperless_client = PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN)

            kwargs = dict(
                method=transfer_method,
                hash_store=hash_store,
                audit_log=audit_log,
                file_patterns=file_patterns,
                consume_dir=consume_path,
                paperless_client=paperless_client,
            )

            if once:
                click.echo(f"Scanning {scan_path} (once mode)…")
                results = await scan_existing(scan_path, **kwargs)
                success = sum(1 for r in results if r.transfer_status.value == "success")
                dupes = sum(1 for r in results if r.transfer_status.value == "duplicate")
                errors = sum(1 for r in results if r.transfer_status.value == "error")
                click.echo(
                    f"Done. Files: {len(results)}, "
                    f"Transferred: {success}, Duplicates: {dupes}, Errors: {errors}"
                )
            else:
                click.echo(f"Watching {scan_path} for new files (Ctrl+C to stop)…")
                click.echo(f"  Method: {transfer_method.value}")
                click.echo(f"  Patterns: {file_patterns}")
                await watch_folder(scan_path, **kwargs)

        finally:
            if paperless_client:
                await paperless_client.close()
