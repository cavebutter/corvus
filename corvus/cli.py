"""CLI entry point for the Corvus document management system.

Commands:
    corvus tag             — batch-tag documents via LLM
    corvus review          — interactively review pending items
    corvus digest          — show activity digest
    corvus status          — quick overview of queue and recent activity
    corvus watch           — watch a scan folder and transfer files to Paperless
    corvus fetch           — retrieve a document via natural language
    corvus ask             — single natural language query to the orchestrator
    corvus chat            — interactive REPL with the orchestrator
    corvus voice           — voice assistant (wake word, STT, TTS)
    corvus email triage    — classify and triage unread emails
    corvus email review    — review queued email actions
    corvus email summary   — summarize unread emails
    corvus email status    — show email pipeline statistics
    corvus email accounts  — list configured email accounts
"""

import asyncio
import logging
import sys

import click
import httpx

from corvus.config import (
    AUDIT_LOG_PATH,
    CHAT_MODEL,
    CONVERSATION_DB_PATH,
    EMAIL_AUDIT_LOG_PATH,
    EMAIL_BATCH_SIZE,
    EMAIL_REVIEW_DB_PATH,
    OLLAMA_BASE_URL,
    PAPERLESS_API_TOKEN,
    PAPERLESS_BASE_URL,
    QUEUE_DB_PATH,
    VOICE_MAX_LISTEN_DURATION,
    VOICE_SILENCE_DURATION,
    VOICE_STT_BEAM_SIZE,
    VOICE_STT_COMPUTE_TYPE,
    VOICE_STT_DEVICE,
    VOICE_STT_MODEL,
    VOICE_TTS_LANG_CODE,
    VOICE_TTS_SPEED,
    VOICE_TTS_VOICE,
    VOICE_WAKEWORD_MODEL_PATH,
    VOICE_WAKEWORD_THRESHOLD,
    WATCHDOG_AUDIT_LOG_PATH,
    WATCHDOG_CONSUME_DIR,
    WATCHDOG_FILE_PATTERNS,
    WATCHDOG_HASH_DB_PATH,
    WATCHDOG_SCAN_DIR,
    WATCHDOG_TRANSFER_METHOD,
    load_email_accounts,
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
@click.option("--new", "start_new", is_flag=True, help="Start a fresh conversation.")
@click.option("--list", "list_convs", is_flag=True, help="Show recent conversations and exit.")
@click.option("--resume", "resume_id", default=None, help="Resume a specific conversation (accepts ID prefix).")
def chat(
    model: str | None,
    keep_alive: str,
    start_new: bool,
    list_convs: bool,
    resume_id: str | None,
) -> None:
    """Interactive REPL with the Corvus orchestrator."""
    if list_convs:
        _list_conversations()
        return
    _validate_config()
    asyncio.run(_chat_async(model, keep_alive, start_new, resume_id))


def _list_conversations() -> None:
    """Display recent conversations and exit."""
    from corvus.orchestrator.conversation_store import ConversationStore

    with ConversationStore(CONVERSATION_DB_PATH) as store:
        convs = store.list_conversations(limit=20)
        if not convs:
            click.echo("No conversations yet.")
            return
        click.echo("Recent conversations:\n")
        for i, conv in enumerate(convs, 1):
            short_id = conv["id"][:8]
            updated = conv["updated_at"][:16].replace("T", " ")
            count = conv["message_count"]
            title = conv["title"]
            click.echo(f"  {i:>2}. [{updated}] {title}  ({count} msgs, id={short_id})")


async def _chat_async(
    model: str | None,
    keep_alive: str,
    start_new: bool,
    resume_id: str | None,
) -> None:
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.conversation_store import ConversationStore
    from corvus.orchestrator.history import ConversationHistory, summarize_response
    from corvus.orchestrator.router import dispatch
    from corvus.planner.intent_classifier import classify_intent
    from corvus.schemas.orchestrator import FetchPipelineResult, OrchestratorAction

    chat_model_resolved = CHAT_MODEL if CHAT_MODEL else None

    store = ConversationStore(CONVERSATION_DB_PATH)
    try:
        async with (
            PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
            OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
        ):
            model = await _resolve_model(ollama, model)
            if chat_model_resolved:
                click.echo(f"Chat model: {CHAT_MODEL}")

            # Determine conversation mode: resume or new
            history: ConversationHistory
            if resume_id:
                conv = store.get_conversation(resume_id)
                if conv is None:
                    click.echo(f"Error: No conversation found matching '{resume_id}'.", err=True)
                    return
                history = ConversationHistory.from_store(store, conv["id"])
                click.echo(f"Resuming: {conv['title']}")
            elif start_new:
                history = ConversationHistory(max_turns=20, store=store)
                click.echo("Starting new conversation.")
            else:
                # Default: resume most recent, or start new
                recent_id = store.get_most_recent()
                if recent_id:
                    conv = store.get_conversation(recent_id)
                    history = ConversationHistory.from_store(store, recent_id)
                    click.echo(f"Resuming: {conv['title']}")
                else:
                    history = ConversationHistory(max_turns=20, store=store)
                    click.echo("Starting new conversation.")

            click.echo("Corvus interactive mode. Type 'quit' to leave.\n")

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
                    # Deferred creation: create conversation on first real message
                    if history.conversation_id is None:
                        conv_id = store.create(user_input)
                        history.set_persistence(store, conv_id)

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
    finally:
        store.close()


# ------------------------------------------------------------------
# corvus voice
# ------------------------------------------------------------------


def _check_voice_deps() -> None:
    """Check that voice optional dependencies are installed."""
    missing = []
    for module, package in [
        ("faster_whisper", "faster-whisper"),
        ("kokoro", "kokoro"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        click.echo(
            f"Error: Missing voice dependencies: {', '.join(missing)}\n"
            f"Install them with: pip install corvus[voice]",
            err=True,
        )
        sys.exit(1)


def _list_voices() -> None:
    """List available Kokoro TTS voices."""
    try:
        from huggingface_hub import list_repo_tree

        files = list(list_repo_tree("hexgrad/Kokoro-82M", path_in_repo="voices"))
        voices = sorted(
            f.rfilename.replace("voices/", "").replace(".pt", "")
            for f in files
            if f.rfilename.endswith(".pt")
        )
        prefixes = {
            "af": "American Female",
            "am": "American Male",
            "bf": "British Female",
            "bm": "British Male",
        }
        groups: dict[str, list[str]] = {}
        for v in voices:
            groups.setdefault(v[:2], []).append(v)
        click.echo(f"Available Kokoro voices ({len(voices)} total):\n")
        for prefix, items in sorted(groups.items()):
            label = prefixes.get(prefix, prefix)
            click.echo(f"  {label}:")
            for v in items:
                click.echo(f"    {v}")
        click.echo(f"\nUsage: corvus voice --voice bf_emma")
    except Exception as exc:
        click.echo(f"Error listing voices: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option("--keep-alive", default="30m", show_default=True, help="Ollama keep_alive duration.")
@click.option("--new", "start_new", is_flag=True, help="Start a fresh conversation.")
@click.option("--resume", "resume_id", default=None, help="Resume a specific conversation (accepts ID prefix).")
@click.option("--voice", "voice_name", default=None, help="Kokoro voice name (e.g. af_heart, bf_emma).")
@click.option("--no-wakeword", is_flag=True, help="Skip wake word detection, press Enter to trigger listening.")
@click.option("--list-voices", is_flag=True, help="List available TTS voices and exit.")
def voice(
    model: str | None,
    keep_alive: str,
    start_new: bool,
    resume_id: str | None,
    voice_name: str | None,
    no_wakeword: bool,
    list_voices: bool,
) -> None:
    """Voice assistant — wake word, speech-to-text, text-to-speech."""
    if list_voices:
        _list_voices()
        return
    _check_voice_deps()
    _validate_config()
    asyncio.run(_voice_async(model, keep_alive, start_new, resume_id, voice_name, no_wakeword))


async def _voice_async(
    model: str | None,
    keep_alive: str,
    start_new: bool,
    resume_id: str | None,
    voice_name: str | None,
    no_wakeword: bool,
) -> None:
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.conversation_store import ConversationStore
    from corvus.voice.pipeline import VoicePipeline

    chat_model_resolved = CHAT_MODEL if CHAT_MODEL else None
    tts_voice = voice_name or VOICE_TTS_VOICE

    store = ConversationStore(CONVERSATION_DB_PATH)
    try:
        async with (
            PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
            OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama,
        ):
            resolved_model = await _resolve_model(ollama, model)
            if chat_model_resolved:
                click.echo(f"Chat model: {CHAT_MODEL}")

            # Determine conversation mode
            conversation_id: str | None = None
            if resume_id:
                conv = store.get_conversation(resume_id)
                if conv is None:
                    click.echo(f"Error: No conversation found matching '{resume_id}'.", err=True)
                    return
                conversation_id = conv["id"]
                click.echo(f"Resuming: {conv['title']}")
            elif start_new:
                click.echo("Starting new conversation.")
            else:
                recent_id = store.get_most_recent()
                if recent_id:
                    conv = store.get_conversation(recent_id)
                    conversation_id = recent_id
                    click.echo(f"Resuming: {conv['title']}")
                else:
                    click.echo("Starting new conversation.")

            click.echo("Loading voice models...")

            pipeline = VoicePipeline(
                ollama=ollama,
                paperless=paperless,
                model=resolved_model,
                keep_alive=keep_alive,
                chat_model=chat_model_resolved,
                queue_db_path=QUEUE_DB_PATH,
                audit_log_path=AUDIT_LOG_PATH,
                stt_model=VOICE_STT_MODEL,
                stt_device=VOICE_STT_DEVICE,
                stt_compute_type=VOICE_STT_COMPUTE_TYPE,
                stt_beam_size=VOICE_STT_BEAM_SIZE,
                tts_lang_code=VOICE_TTS_LANG_CODE,
                tts_voice=tts_voice,
                tts_speed=VOICE_TTS_SPEED,
                wakeword_model_path=VOICE_WAKEWORD_MODEL_PATH,
                wakeword_threshold=VOICE_WAKEWORD_THRESHOLD,
                silence_duration=VOICE_SILENCE_DURATION,
                max_listen_duration=VOICE_MAX_LISTEN_DURATION,
                store=store,
                conversation_id=conversation_id,
            )

            if no_wakeword:
                click.echo("Voice ready. Press Enter to speak, Ctrl+C to exit.")
            else:
                click.echo("Voice ready. Say the wake word to begin, Ctrl+C to exit.")

            try:
                await pipeline.run(no_wakeword=no_wakeword)
            except KeyboardInterrupt:
                click.echo("\nGoodbye.")
    finally:
        store.close()


# ------------------------------------------------------------------
# corvus email (command group)
# ------------------------------------------------------------------


def _validate_email_config() -> list[dict]:
    """Load and validate email account configs. Exits on failure."""
    accounts = load_email_accounts()
    if not accounts:
        click.echo(
            "Error: No email accounts configured.\n"
            "Create secrets/email_accounts.json with your account settings.",
            err=True,
        )
        sys.exit(1)
    return accounts


def _find_email_account(accounts: list[dict], email_filter: str | None) -> list[dict]:
    """Filter accounts by email address, or return all."""
    if email_filter:
        matched = [a for a in accounts if a.get("email") == email_filter]
        if not matched:
            click.echo(f"Error: No account found matching '{email_filter}'.", err=True)
            click.echo("Configured accounts:")
            for a in accounts:
                click.echo(f"  - {a.get('email')} ({a.get('name', '?')})")
            sys.exit(1)
        return matched
    return accounts


@cli.group()
def email() -> None:
    """Email inbox management."""


@email.command()
@click.option("--account", "-a", default=None, help="Account email (default: all).")
@click.option("--limit", "-n", default=None, type=int, help="Max emails to process.")
@click.option("--model", "-m", default=None, help="Ollama model name (auto-detected if omitted).")
@click.option("--keep-alive", default="5m", show_default=True, help="Ollama keep_alive duration.")
@click.option("--apply", "apply_actions", is_flag=True, help="Apply actions (skip review queue).")
def triage(
    account: str | None,
    limit: int | None,
    model: str | None,
    keep_alive: str,
    apply_actions: bool,
) -> None:
    """Classify and triage unread emails."""
    accounts = _validate_email_config()
    targets = _find_email_account(accounts, account)
    asyncio.run(_email_triage_async(targets, limit, model, keep_alive, apply_actions))


async def _email_triage_async(
    accounts: list[dict],
    limit: int | None,
    model: str | None,
    keep_alive: str,
    apply_actions: bool,
) -> None:
    from corvus.integrations.ollama import OllamaClient
    from corvus.orchestrator.email_pipelines import run_email_triage
    from corvus.schemas.email import EmailAccountConfig

    async with OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama:
        resolved_model = await _resolve_model(ollama, model)

        for account_dict in accounts:
            config = EmailAccountConfig(**account_dict)
            click.echo(f"\n{'='*60}")
            click.echo(f"Account: {config.name} ({config.email})")
            click.echo(f"{'='*60}")

            result = await run_email_triage(
                account_config=config,
                ollama=ollama,
                model=resolved_model,
                keep_alive=keep_alive,
                limit=limit or EMAIL_BATCH_SIZE,
                force_queue=not apply_actions,
                review_db_path=EMAIL_REVIEW_DB_PATH,
                audit_log_path=EMAIL_AUDIT_LOG_PATH,
                on_progress=click.echo,
            )

            click.echo(f"\nResults for {config.email}:")
            click.echo(f"  Processed: {result.processed}")
            click.echo(f"  Auto-acted: {result.auto_acted}")
            click.echo(f"  Queued for review: {result.queued}")
            click.echo(f"  Errors: {result.errors}")
            if result.categories:
                click.echo("  Categories:")
                for cat, count in sorted(result.categories.items()):
                    click.echo(f"    {cat}: {count}")


@email.command("review")
def email_review() -> None:
    """Review queued email actions."""
    asyncio.run(_email_review_async())


async def _email_review_async() -> None:
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.integrations.imap import ImapClient
    from corvus.queue.email_review import EmailReviewQueue
    from corvus.router.email import execute_email_action
    from corvus.schemas.email import EmailAccountConfig

    audit_log = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)

    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as review_queue:
        pending = review_queue.list_pending()
        if not pending:
            click.echo("No pending email actions to review.")
            return

        click.echo(f"Found {len(pending)} pending email action(s).\n")

        approved = 0
        rejected = 0
        skipped = 0

        # Group by account for IMAP connections
        accounts_cache: dict[str, EmailAccountConfig] = {}
        all_accounts = load_email_accounts()
        for a in all_accounts:
            accounts_cache[a["email"]] = EmailAccountConfig(**a)

        for i, item in enumerate(pending, 1):
            task = item.task
            click.echo(f"--- [{i}/{len(pending)}] Email from {task.from_address} ---")
            click.echo(f"  Subject: {task.subject}")
            click.echo(f"  Account: {task.account_email}")
            click.echo(f"  Category: {task.classification.category.value}")
            click.echo(f"  Confidence: {task.overall_confidence:.0%}")
            click.echo(f"  Proposed: {task.proposed_action.action_type.value}")
            if task.proposed_action.target_folder:
                click.echo(f"  Target folder: {task.proposed_action.target_folder}")
            if task.classification.summary:
                click.echo(f"  Summary: {task.classification.summary}")
            click.echo(f"  Reasoning: {task.classification.reasoning}")

            choice = click.prompt(
                "  Action",
                type=click.Choice(["a", "r", "s", "q"], case_sensitive=False),
                prompt_suffix=" [a]pprove / [r]eject / [s]kip / [q]uit: ",
            )

            if choice == "a":
                account_config = accounts_cache.get(task.account_email)
                if not account_config:
                    click.echo(
                        f"  -> ERROR: Account {task.account_email} not found in config.",
                        err=True,
                    )
                    continue

                try:
                    async with ImapClient(account_config) as imap:
                        await execute_email_action(task, imap=imap)
                    review_queue.approve(item.id, notes="Approved via CLI")
                    audit_log.log_review_approved(task)
                    click.echo("  -> Approved and applied.")
                    approved += 1
                except Exception:
                    logger.exception("Error applying email action for %s", task.uid)
                    click.echo(
                        "  -> ERROR: Failed to apply. Item remains pending.",
                        err=True,
                    )
            elif choice == "r":
                notes = click.prompt(
                    "  Rejection notes (optional)",
                    default="",
                    show_default=False,
                )
                review_queue.reject(item.id, notes=notes or None)
                audit_log.log_review_rejected(task)
                click.echo("  -> Rejected.")
                rejected += 1
            elif choice == "s":
                click.echo("  -> Skipped.")
                skipped += 1
            elif choice == "q":
                click.echo("  -> Quitting review.")
                break

        click.echo(
            f"\nReview complete. Approved: {approved}, "
            f"Rejected: {rejected}, Skipped: {skipped}"
        )


@email.command("summary")
@click.option("--account", "-a", default=None, help="Account email (default: all).")
@click.option("--limit", "-n", default=50, show_default=True, help="Max emails to summarize.")
@click.option("--model", "-m", default=None, help="Ollama model name.")
@click.option("--keep-alive", default="5m", show_default=True, help="Ollama keep_alive duration.")
def email_summary(
    account: str | None, limit: int, model: str | None, keep_alive: str
) -> None:
    """Summarize unread emails (most recent first)."""
    accounts = _validate_email_config()
    targets = _find_email_account(accounts, account)
    asyncio.run(_email_summary_async(targets, limit, model, keep_alive))


async def _email_summary_async(
    accounts: list[dict],
    limit: int,
    model: str | None,
    keep_alive: str,
) -> None:
    from corvus.integrations.ollama import OllamaClient
    from corvus.orchestrator.email_pipelines import run_email_summary
    from corvus.schemas.email import EmailAccountConfig

    async with OllamaClient(OLLAMA_BASE_URL, default_keep_alive=keep_alive) as ollama:
        resolved_model = await _resolve_model(ollama, model)

        for account_dict in accounts:
            config = EmailAccountConfig(**account_dict)
            click.echo(f"\n{'='*60}")
            click.echo(f"Account: {config.name} ({config.email})")
            click.echo(f"{'='*60}")

            result = await run_email_summary(
                account_config=config,
                ollama=ollama,
                model=resolved_model,
                keep_alive=keep_alive,
                limit=limit,
                on_progress=click.echo,
            )

            click.echo(f"\nInbox Summary ({result.total_unread} unread):")
            click.echo(result.summary)

            if result.by_category:
                click.echo("\nBy category:")
                for cat, count in sorted(result.by_category.items()):
                    click.echo(f"  {cat}: {count}")

            if result.important_subjects:
                click.echo("\nImportant:")
                for subj in result.important_subjects:
                    click.echo(f"  - {subj}")

            if result.action_items:
                click.echo("\nAction items:")
                for ai in result.action_items:
                    deadline = f" (by {ai.deadline})" if ai.deadline else ""
                    click.echo(f"  - {ai.description}{deadline}")


@email.command("status")
def email_status() -> None:
    """Show email pipeline statistics."""
    from datetime import UTC, datetime, timedelta

    from corvus.audit.email_logger import EmailAuditLog
    from corvus.queue.email_review import EmailReviewQueue

    audit_log = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)
    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as review_queue:
        pending_count = review_queue.count_pending()

    since = datetime.now(UTC) - timedelta(hours=24)
    entries = audit_log.read_entries(since=since)

    auto_applied = sum(1 for e in entries if e.action == "auto_applied")
    queued = sum(1 for e in entries if e.action == "queued_for_review")
    approved = sum(1 for e in entries if e.action == "review_approved")
    rejected = sum(1 for e in entries if e.action == "review_rejected")

    click.echo("Email Pipeline Status (last 24h)")
    click.echo(f"  Pending review:  {pending_count}")
    click.echo(f"  Auto-applied:    {auto_applied}")
    click.echo(f"  Queued:          {queued}")
    click.echo(f"  Approved:        {approved}")
    click.echo(f"  Rejected:        {rejected}")


@email.command("accounts")
def email_accounts() -> None:
    """List configured email accounts."""
    accounts = load_email_accounts()
    if not accounts:
        click.echo("No email accounts configured.")
        click.echo("Create secrets/email_accounts.json with your account settings.")
        return

    click.echo(f"Configured email accounts ({len(accounts)}):")
    for a in accounts:
        gmail_tag = " [Gmail]" if a.get("is_gmail") else ""
        click.echo(f"  - {a.get('name', '?')} ({a.get('email', '?')}){gmail_tag}")
        folders = a.get("folders", {})
        if folders:
            click.echo(f"    Folders: {', '.join(f'{k}={v}' for k, v in folders.items())}")
