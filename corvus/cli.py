"""CLI entry point for the Corvus document management system.

Commands:
    corvus tag      — batch-tag documents via LLM
    corvus review   — interactively review pending items
    corvus digest   — show activity digest
    corvus status   — quick overview of queue and recent activity
    corvus watch    — watch a scan folder and transfer files to Paperless
    corvus fetch    — retrieve a document via natural language
    corvus ask      — single natural language query to the orchestrator
    corvus chat     — interactive REPL with the orchestrator
"""

import asyncio
import logging
import sys

import click
import httpx

from corvus.config import (
    AUDIT_LOG_PATH,
    CHAT_MODEL,
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
    """Corvus — local AI agent system for Paperless-ngx."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _format_doc_line(i: int, doc: dict, width: int = 2) -> str:
    """Format a document result as two display lines.

    Line 1: index, date, title, id
    Line 2: correspondent | document_type | tags (indented)
    """
    line1 = f"  {i:>{width}}. [{doc['created'][:10]}] {doc['title']}  (id={doc['id']})"
    parts = []
    parts.append(doc.get("correspondent") or "\u2014")
    parts.append(doc.get("document_type") or "\u2014")
    if doc.get("tags"):
        parts.append(", ".join(doc["tags"]))
    line2 = f"  {' ' * width}  {' | '.join(parts)}"
    return f"{line1}\n{line2}"


async def _resolve_model(ollama, model: str | None) -> str:
    """Auto-detect model if not specified. Exits on failure."""
    if model is not None:
        return model
    resolved = await ollama.pick_instruct_model()
    if resolved is None:
        click.echo("Error: No models available on Ollama server.", err=True)
        sys.exit(1)
    click.echo(f"Auto-selected model: {resolved}")
    return resolved


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
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.pipelines import run_tag_pipeline

    try:
        async with (
            PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
            OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
        ):
            model = await _resolve_model(ollama, model)

            await run_tag_pipeline(
                paperless=paperless,
                ollama=ollama,
                model=model,
                limit=limit,
                include_tagged=include_tagged,
                keep_alive=keep_alive,
                force_queue=force_queue,
                queue_db_path=QUEUE_DB_PATH,
                audit_log_path=AUDIT_LOG_PATH,
                on_progress=click.echo,
            )
    except httpx.RemoteProtocolError:
        click.echo("\nError: Lost connection to Paperless. Check that the server is running.", err=True)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        click.echo(f"\nError: Paperless returned {exc.response.status_code}.", err=True)
        sys.exit(1)


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

    try:
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
    except httpx.RemoteProtocolError:
        click.echo("\nError: Lost connection to Paperless. Check that the server is running.", err=True)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        click.echo(f"\nError: Paperless returned {exc.response.status_code}.", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# corvus digest
# ------------------------------------------------------------------


@cli.command()
@click.option("--hours", default=24, show_default=True, help="Lookback period in hours.")
def digest(hours: int) -> None:
    """Show activity digest for the recent period."""
    from corvus.orchestrator.pipelines import run_digest_pipeline

    result = run_digest_pipeline(
        queue_db_path=QUEUE_DB_PATH,
        audit_log_path=AUDIT_LOG_PATH,
        hours=hours,
    )
    click.echo(result.rendered_text)


# ------------------------------------------------------------------
# corvus status
# ------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Quick overview of queue and recent activity."""
    from corvus.orchestrator.pipelines import run_status_pipeline

    result = run_status_pipeline(
        queue_db_path=QUEUE_DB_PATH,
        audit_log_path=AUDIT_LOG_PATH,
    )
    click.echo("Corvus Status")
    click.echo(f"  Pending review:     {result.pending_count}")
    click.echo(f"  Processed (24h):    {result.processed_24h}")
    click.echo(f"  Reviewed (24h):     {result.reviewed_24h}")


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

        except httpx.RemoteProtocolError:
            click.echo("\nError: Lost connection to Paperless. Check that the server is running.", err=True)
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            click.echo(f"\nError: Paperless returned {exc.response.status_code}.", err=True)
            sys.exit(1)
        finally:
            if paperless_client:
                await paperless_client.close()


# ------------------------------------------------------------------
# corvus fetch
# ------------------------------------------------------------------


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option(
    "--method",
    type=click.Choice(["browser", "download"], case_sensitive=False),
    default="browser",
    show_default=True,
    help="Delivery method: open in browser or download file.",
)
@click.option(
    "--download-dir",
    default=None,
    help="Download directory (defaults to ~/Downloads).",
)
@click.option("--keep-alive", default="5m", show_default=True, help="Ollama keep_alive duration.")
def fetch(
    query: tuple[str, ...],
    model: str | None,
    method: str,
    download_dir: str | None,
    keep_alive: str,
) -> None:
    """Retrieve a document from Paperless-ngx via natural language."""
    _validate_config()
    query_str = " ".join(query)
    if not query_str.strip():
        click.echo("Error: Query cannot be empty.", err=True)
        sys.exit(1)
    asyncio.run(_fetch_async(query_str, model, method, download_dir, keep_alive))


async def _fetch_async(
    query: str,
    model: str | None,
    method: str,
    download_dir: str | None,
    keep_alive: str,
) -> None:
    import webbrowser
    from pathlib import Path

    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.pipelines import run_fetch_pipeline
    from corvus.schemas.document_retrieval import DeliveryMethod

    delivery = DeliveryMethod(method)

    try:
        async with (
            PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
            OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
        ):
            model = await _resolve_model(ollama, model)

            result = await run_fetch_pipeline(
                paperless=paperless,
                ollama=ollama,
                model=model,
                query=query,
                keep_alive=keep_alive,
                on_progress=click.echo,
            )

            # Low confidence check
            if result.interpretation_confidence < 0.5 and not click.confirm(
                "\nLow confidence. Continue?", default=False
            ):
                click.echo("Aborted.")
                return

            # Display results and deliver
            if result.documents_found == 0:
                click.echo("\nNo documents found.")
                return

            click.echo(f"\nFound {result.documents_found} document(s).")
            docs = result.documents

            if len(docs) == 1 and result.documents_found == 1:
                doc = docs[0]
                click.echo(_format_doc_line(1, doc))
                selected = doc
            else:
                display_count = min(len(docs), 10)
                for i, doc in enumerate(docs[:display_count], 1):
                    click.echo(_format_doc_line(i, doc))

                if result.documents_found > display_count:
                    click.echo(f"  ... and {result.documents_found - display_count} more. Consider refining your query.")

                raw_choice = click.prompt(
                    "\nSelect", prompt_suffix=f" [1-{display_count}, q to quit]: "
                )
                if raw_choice.strip().lower() == "q":
                    click.echo("Aborted.")
                    return

                try:
                    idx = int(raw_choice) - 1
                    if idx < 0 or idx >= display_count:
                        raise ValueError
                    selected = docs[idx]
                except (ValueError, IndexError):
                    click.echo("Invalid selection.", err=True)
                    sys.exit(1)

            # Deliver
            if delivery == DeliveryMethod.BROWSER:
                url = paperless.get_document_url(selected["id"])
                click.echo(f"Opening: {url}")
                webbrowser.open(url)
            else:
                dest_dir = Path(download_dir) if download_dir else Path.home() / "Downloads"
                dest_dir.mkdir(parents=True, exist_ok=True)
                path = await paperless.download_document(selected["id"], dest_dir)
                click.echo(f"Downloaded: {path}")
    except httpx.RemoteProtocolError:
        click.echo("\nError: Lost connection to Paperless. Check that the server is running.", err=True)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        click.echo(f"\nError: Paperless returned {exc.response.status_code}.", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# corvus ask
# ------------------------------------------------------------------


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option("--keep-alive", default="5m", show_default=True, help="Ollama keep_alive duration.")
def ask(query: tuple[str, ...], model: str | None, keep_alive: str) -> None:
    """Ask Corvus a natural language question."""
    _validate_config()
    query_str = " ".join(query)
    if not query_str.strip():
        click.echo("Error: Query cannot be empty.", err=True)
        sys.exit(1)
    asyncio.run(_ask_async(query_str, model, keep_alive))


async def _ask_async(query: str, model: str | None, keep_alive: str) -> None:
    import webbrowser

    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.router import dispatch
    from corvus.planner.intent_classifier import classify_intent
    from corvus.schemas.orchestrator import OrchestratorAction

    chat_model_resolved = CHAT_MODEL if CHAT_MODEL else None

    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
    ):
        model = await _resolve_model(ollama, model)
        if chat_model_resolved:
            click.echo(f"Chat model: {CHAT_MODEL}")

        # Step 1: Classify intent
        classification, _raw = await classify_intent(
            query, ollama=ollama, model=model, keep_alive=keep_alive,
        )
        click.echo(f"  Intent: {classification.intent.value} ({classification.confidence:.0%})")

        # Step 2: Dispatch via orchestrator router
        try:
            response = await dispatch(
                classification,
                user_input=query,
                paperless=paperless,
                ollama=ollama,
                model=model,
                keep_alive=keep_alive,
                queue_db_path=QUEUE_DB_PATH,
                audit_log_path=AUDIT_LOG_PATH,
                on_progress=click.echo,
                chat_model=chat_model_resolved,
            )
        except Exception:
            logger.exception("Error dispatching intent %s", classification.intent.value)
            click.echo("\nError: Failed to complete request. See log for details.", err=True)
            sys.exit(1)

        # Step 3: Render response
        _render_orchestrator_response(response)

        # Handle interactive fetch selection if needed
        if (
            response.action == OrchestratorAction.DISPATCHED
            and response.result is not None
        ):
            from corvus.schemas.orchestrator import FetchPipelineResult

            if isinstance(response.result, FetchPipelineResult) and response.result.documents:
                docs = response.result.documents
                total = response.result.documents_found
                if len(docs) == 1 and total == 1:
                    doc = docs[0]
                    url = paperless.get_document_url(doc["id"])
                    click.echo(f"Opening: {url}")
                    webbrowser.open(url)
                elif docs:
                    display_count = min(len(docs), 10)
                    for i, doc in enumerate(docs[:display_count], 1):
                        click.echo(_format_doc_line(i, doc))
                    if total > display_count:
                        click.echo(f"  ... and {total - display_count} more.")

                    raw_choice = click.prompt(
                        "\nSelect", prompt_suffix=f" [1-{display_count}, q to quit]: "
                    )
                    if raw_choice.strip().lower() == "q":
                        click.echo("Aborted.")
                        return

                    try:
                        idx = int(raw_choice) - 1
                        if idx < 0 or idx >= display_count:
                            raise ValueError
                        selected = docs[idx]
                    except (ValueError, IndexError):
                        click.echo("Invalid selection.", err=True)
                        return

                    url = paperless.get_document_url(selected["id"])
                    click.echo(f"Opening: {url}")
                    webbrowser.open(url)


def _render_orchestrator_response(response) -> None:
    """Render an OrchestratorResponse to CLI output."""
    from corvus.schemas.orchestrator import (
        DigestResult,
        FetchPipelineResult,
        OrchestratorAction,
        StatusResult,
        TagPipelineResult,
        WebSearchResult,
    )

    if response.action == OrchestratorAction.NEEDS_CLARIFICATION:
        click.echo(f"\n{response.message}")
        if response.clarification_prompt:
            click.echo(f"  Hint: {response.clarification_prompt}")
        return

    if response.action == OrchestratorAction.INTERACTIVE_REQUIRED:
        click.echo(f"\n{response.message}")
        return

    if response.action == OrchestratorAction.CHAT_RESPONSE:
        click.echo(f"\n{response.message}")
        return

    if response.action == OrchestratorAction.DISPATCHED and response.result is not None:
        result = response.result
        if isinstance(result, TagPipelineResult):
            click.echo(
                f"\nTagging complete. Processed: {result.processed}, "
                f"Queued: {result.queued}, Auto-applied: {result.auto_applied}, "
                f"Errors: {result.errors}"
            )
        elif isinstance(result, FetchPipelineResult):
            click.echo(f"\nFound {result.documents_found} document(s).")
        elif isinstance(result, StatusResult):
            click.echo("Corvus Status")
            click.echo(f"  Pending review:     {result.pending_count}")
            click.echo(f"  Processed (24h):    {result.processed_24h}")
            click.echo(f"  Reviewed (24h):     {result.reviewed_24h}")
        elif isinstance(result, DigestResult):
            click.echo(result.rendered_text)
        elif isinstance(result, WebSearchResult):
            click.echo(f"\n{result.summary}")
            if result.sources:
                click.echo("\nSources:")
                for i, src in enumerate(result.sources, 1):
                    click.echo(f"  [{i}] {src.title}")
                    click.echo(f"      {src.url}")


# ------------------------------------------------------------------
# corvus chat
# ------------------------------------------------------------------


@cli.command()
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option("--keep-alive", default="10m", show_default=True, help="Ollama keep_alive duration.")
def chat(model: str | None, keep_alive: str) -> None:
    """Interactive REPL with the Corvus orchestrator."""
    _validate_config()
    asyncio.run(_chat_async(model, keep_alive))


async def _chat_async(model: str | None, keep_alive: str) -> None:
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.history import ConversationHistory, summarize_response
    from corvus.orchestrator.router import dispatch
    from corvus.planner.intent_classifier import classify_intent
    from corvus.schemas.orchestrator import FetchPipelineResult, OrchestratorAction

    chat_model_resolved = CHAT_MODEL if CHAT_MODEL else None

    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
    ):
        model = await _resolve_model(ollama, model)
        if chat_model_resolved:
            click.echo(f"Chat model: {CHAT_MODEL}")

        click.echo("Corvus interactive mode. Type 'quit' to leave.\n")

        history = ConversationHistory(max_turns=20)

        while True:
            try:
                user_input = click.prompt("You", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye.")
                break

            if user_input.strip().lower() in ("quit", "exit", "q"):
                click.echo("Goodbye.")
                break

            if not user_input.strip():
                continue

            try:
                conversation_context = history.get_recent_context(max_turns=5)

                classification, _raw = await classify_intent(
                    user_input, ollama=ollama, model=model, keep_alive=keep_alive,
                    conversation_context=conversation_context or None,
                )
                click.echo(f"  Intent: {classification.intent.value} ({classification.confidence:.0%})")

                response = await dispatch(
                    classification,
                    user_input=user_input,
                    paperless=paperless,
                    ollama=ollama,
                    model=model,
                    keep_alive=keep_alive,
                    queue_db_path=QUEUE_DB_PATH,
                    audit_log_path=AUDIT_LOG_PATH,
                    on_progress=click.echo,
                    chat_model=chat_model_resolved,
                    conversation_history=history.get_messages(),
                )

                _render_orchestrator_response(response)

                # Handle fetch results with interactive selection
                if (
                    response.action == OrchestratorAction.DISPATCHED
                    and isinstance(response.result, FetchPipelineResult)
                    and response.result.documents
                ):
                    import webbrowser

                    docs = response.result.documents
                    total = response.result.documents_found

                    if len(docs) == 1 and total == 1:
                        doc = docs[0]
                        click.echo(_format_doc_line(1, doc, width=1))
                        url = paperless.get_document_url(doc["id"])
                        click.echo(f"Opening: {url}")
                        webbrowser.open(url)
                    else:
                        display_count = min(len(docs), 10)
                        for i, doc in enumerate(docs[:display_count], 1):
                            click.echo(_format_doc_line(i, doc, width=1))
                        if total > display_count:
                            click.echo(f"  ... and {total - display_count} more.")

                        raw_choice = click.prompt(
                            "\nSelect",
                            prompt_suffix=f" [1-{display_count}, s to skip]: ",
                            default="s",
                        )
                        if raw_choice.strip().lower() != "s":
                            try:
                                idx = int(raw_choice) - 1
                                if idx < 0 or idx >= display_count:
                                    raise ValueError
                                selected = docs[idx]
                                url = paperless.get_document_url(selected["id"])
                                click.echo(f"Opening: {url}")
                                webbrowser.open(url)
                            except (ValueError, IndexError):
                                click.echo("Invalid selection.", err=True)

                # Update conversation history
                history.add_user_message(user_input)
                history.add_assistant_message(summarize_response(response))

            except Exception:
                logger.exception("Error processing input")
                click.echo("  Error processing your request. See log for details.")

            click.echo()
