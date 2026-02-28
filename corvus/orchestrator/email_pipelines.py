"""Pipeline handlers for the email triage and summarization features.

Each handler encapsulates a complete pipeline and returns a typed result.
CLI commands and the orchestrator router both call these handlers.
"""

import logging
from collections.abc import Callable

from corvus.audit.email_logger import EmailAuditLog
from corvus.integrations.imap import ImapClient
from corvus.integrations.ollama import OllamaClient
from corvus.queue.email_review import EmailReviewQueue
from corvus.schemas.email import (
    ActionItem,
    EmailAccountConfig,
    EmailCategory,
    EmailSummaryResult,
    EmailTriageResult,
)

logger = logging.getLogger(__name__)


async def run_email_triage(
    *,
    account_config: EmailAccountConfig,
    ollama: OllamaClient,
    model: str,
    keep_alive: str = "5m",
    limit: int = 50,
    force_queue: bool = True,
    review_db_path: str,
    audit_log_path: str,
    on_progress: Callable[[str], None] | None = None,
) -> EmailTriageResult:
    """Triage unread emails in an account's inbox.

    Flow:
    1. Connect to IMAP, ensure target folders exist.
    2. Fetch unread emails (limited by batch size).
    3. For each email: classify, optionally extract data, route action.
    4. Return result with counts.

    Args:
        account_config: Email account configuration.
        ollama: An open OllamaClient instance.
        model: Ollama model name.
        keep_alive: Ollama keep_alive duration.
        limit: Maximum emails to process.
        force_queue: Queue all actions for review (initial safe posture).
        review_db_path: Path to the email review queue SQLite database.
        audit_log_path: Path to the email audit log file.
        on_progress: Optional callback for progress messages.

    Returns:
        EmailTriageResult with counts and category breakdown.
    """
    from corvus.executors.email_classifier import classify_email
    from corvus.executors.email_extractor import EXTRACTABLE_CATEGORIES, extract_email_data
    from corvus.router.email import route_email_action

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    audit_log = EmailAuditLog(audit_log_path)
    folders = account_config.folders

    processed = 0
    auto_acted = 0
    queued = 0
    errors = 0
    categories: dict[str, int] = {}

    async with ImapClient(account_config) as imap:
        # Ensure target folders exist
        target_folders = [
            v for k, v in folders.items() if k != "inbox"
        ]
        if target_folders:
            await imap.ensure_folders(target_folders)

        # Fetch unread emails (full messages for classification)
        inbox_folder = folders.get("inbox", "INBOX")
        _emit(f"Fetching emails from {inbox_folder}...")
        messages = await imap.fetch_messages(inbox_folder, limit=limit)

        if not messages:
            _emit("No unread emails to process.")
            return EmailTriageResult(account_email=account_config.email)

        _emit(f"Found {len(messages)} unread email(s). Processing...")

        with EmailReviewQueue(review_db_path) as review_queue:
            for i, email in enumerate(messages, 1):
                try:
                    _emit(
                        f"\n[{i}/{len(messages)}] {email.subject}"
                        f"\n  From: {email.from_address}"
                    )

                    # Classify
                    task, _raw = await classify_email(
                        email,
                        ollama=ollama,
                        model=model,
                        folders=folders,
                        keep_alive=keep_alive,
                    )

                    category = task.classification.category.value
                    categories[category] = categories.get(category, 0) + 1

                    _emit(
                        f"  Category: {category} "
                        f"(confidence={task.overall_confidence:.0%})"
                    )
                    if task.classification.summary:
                        _emit(f"  Summary: {task.classification.summary}")

                    # Extract data from receipts/invoices/action items
                    if task.classification.category in EXTRACTABLE_CATEGORIES:
                        try:
                            extraction, _raw2 = await extract_email_data(
                                email,
                                task.classification,
                                ollama=ollama,
                                model=model,
                                keep_alive=keep_alive,
                            )
                            if extraction.invoice:
                                inv = extraction.invoice
                                _emit(
                                    f"  Invoice: {inv.vendor} "
                                    f"${inv.amount or '?'} {inv.currency}"
                                )
                            if extraction.action_items:
                                for ai in extraction.action_items:
                                    _emit(f"  Action: {ai.description}")
                        except Exception:
                            logger.exception(
                                "Error extracting data from email %s", email.uid
                            )

                    # Route through confidence gate
                    applied = await route_email_action(
                        task,
                        imap=imap,
                        force_queue=force_queue,
                        audit_log=audit_log,
                        review_queue=review_queue,
                    )

                    if applied:
                        auto_acted += 1
                        _emit(
                            f"  Action: {task.proposed_action.action_type.value} "
                            f"(auto-applied)"
                        )
                    else:
                        queued += 1
                        _emit(
                            f"  Action: {task.proposed_action.action_type.value} "
                            f"(queued for review)"
                        )

                    processed += 1

                except Exception:
                    errors += 1
                    logger.exception(
                        "Error processing email %s: %s", email.uid, email.subject
                    )
                    _emit("  ERROR: Failed to process (see log for details)")

    _emit(f"\nDone. Processed: {processed}, Auto: {auto_acted}, Queued: {queued}, Errors: {errors}")

    return EmailTriageResult(
        account_email=account_config.email,
        processed=processed,
        auto_acted=auto_acted,
        queued=queued,
        errors=errors,
        categories=categories,
    )


async def run_email_summary(
    *,
    account_config: EmailAccountConfig,
    ollama: OllamaClient,
    model: str,
    keep_alive: str = "5m",
    on_progress: Callable[[str], None] | None = None,
) -> EmailSummaryResult:
    """Summarize unread emails in an account's inbox.

    Flow:
    1. Fetch unread emails (full messages for important ones).
    2. Classify each email.
    3. Extract action items from ACTION_REQUIRED emails.
    4. Generate summary via LLM chat.

    Args:
        account_config: Email account configuration.
        ollama: An open OllamaClient instance.
        model: Ollama model name.
        keep_alive: Ollama keep_alive duration.
        on_progress: Optional callback for progress messages.

    Returns:
        EmailSummaryResult with summary and category breakdown.
    """
    from corvus.executors.email_classifier import classify_email
    from corvus.executors.email_extractor import EXTRACTABLE_CATEGORIES, extract_email_data

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    async with ImapClient(account_config) as imap:
        inbox_folder = account_config.folders.get("inbox", "INBOX")
        _emit(f"Fetching emails from {inbox_folder}...")
        messages = await imap.fetch_messages(inbox_folder, limit=100)

        if not messages:
            _emit("No unread emails.")
            return EmailSummaryResult(
                account_email=account_config.email,
                summary="No unread emails in inbox.",
            )

        _emit(f"Found {len(messages)} unread email(s). Classifying...")

        by_category: dict[str, int] = {}
        important_subjects: list[str] = []
        action_items: list[ActionItem] = []
        classified_emails: list[str] = []

        for email in messages:
            try:
                task, _raw = await classify_email(
                    email,
                    ollama=ollama,
                    model=model,
                    folders=account_config.folders,
                    keep_alive=keep_alive,
                )

                cat = task.classification.category.value
                by_category[cat] = by_category.get(cat, 0) + 1

                # Track important subjects
                if task.classification.category in (
                    EmailCategory.IMPORTANT,
                    EmailCategory.ACTION_REQUIRED,
                ):
                    important_subjects.append(email.subject)

                # Extract action items
                if task.classification.category in EXTRACTABLE_CATEGORIES:
                    try:
                        extraction, _raw2 = await extract_email_data(
                            email,
                            task.classification,
                            ollama=ollama,
                            model=model,
                            keep_alive=keep_alive,
                        )
                        action_items.extend(extraction.action_items)
                    except Exception:
                        logger.debug(
                            "Extraction failed for %s", email.uid, exc_info=True
                        )

                # Build context for summary
                classified_emails.append(
                    f"- [{cat}] From: {email.from_address} | "
                    f"Subject: {email.subject}"
                    + (
                        f" | Summary: {task.classification.summary}"
                        if task.classification.summary
                        else ""
                    )
                )

            except Exception:
                logger.exception("Error classifying email %s", email.uid)

        _emit("Generating summary...")

        # Generate natural language summary
        context = "\n".join(classified_emails)
        summary_prompt = (
            f"Summarize this inbox ({len(messages)} unread emails):\n\n{context}"
        )
        if action_items:
            ai_text = "\n".join(
                f"- {ai.description}"
                + (f" (deadline: {ai.deadline})" if ai.deadline else "")
                for ai in action_items
            )
            summary_prompt += f"\n\nAction items found:\n{ai_text}"

        summary_text, _raw = await ollama.chat(
            model=model,
            system=(
                "You are Corvus, a helpful email assistant. Summarize the inbox "
                "concisely: highlight important/action-required emails first, "
                "then give category counts. Keep it to 3-5 sentences."
            ),
            prompt=summary_prompt,
            keep_alive=keep_alive,
            temperature=0.3,
        )

    return EmailSummaryResult(
        account_email=account_config.email,
        total_unread=len(messages),
        summary=summary_text,
        by_category=by_category,
        important_subjects=important_subjects,
        action_items=action_items,
    )
