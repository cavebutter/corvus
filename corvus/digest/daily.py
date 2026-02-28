"""Daily digest generation for the document tagging and email pipelines.

Reads audit log entries and review queue state, produces a structured
summary that can be rendered to text (and later to email, HTML, etc.).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from corvus.audit.logger import AuditLog
from corvus.queue.review import ReviewQueue
from corvus.schemas.document_tagging import AuditEntry, GateAction

if TYPE_CHECKING:
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.queue.email_review import EmailReviewQueue

logger = logging.getLogger(__name__)


class DigestItem(BaseModel):
    """A single item in the digest summary."""

    document_id: int
    document_title: str
    confidence: float
    gate_action: GateAction
    tags: list[str]
    correspondent: str | None = None
    document_type: str | None = None
    reasoning: str = ""


class DailyDigest(BaseModel):
    """Structured daily digest of pipeline activity."""

    generated_at: datetime
    period_start: datetime
    period_end: datetime
    auto_applied: list[DigestItem] = Field(default_factory=list)
    flagged: list[DigestItem] = Field(default_factory=list)
    queued_for_review: list[DigestItem] = Field(default_factory=list)
    review_approved: list[DigestItem] = Field(default_factory=list)
    review_rejected: list[DigestItem] = Field(default_factory=list)
    pending_review_count: int = 0

    # Email stats
    email_processed: int = 0
    email_auto_applied: int = 0
    email_queued: int = 0
    email_approved: int = 0
    email_rejected: int = 0
    email_pending_review: int = 0

    @property
    def total_processed(self) -> int:
        return (
            len(self.auto_applied)
            + len(self.flagged)
            + len(self.queued_for_review)
        )

    @property
    def total_reviewed(self) -> int:
        return len(self.review_approved) + len(self.review_rejected)


def _entry_to_item(entry: AuditEntry) -> DigestItem:
    """Convert an audit entry to a digest item."""
    return DigestItem(
        document_id=entry.document_id,
        document_title=entry.document_title,
        confidence=entry.task.overall_confidence,
        gate_action=entry.gate_action,
        tags=[t.tag_name for t in entry.task.result.suggested_tags],
        correspondent=entry.task.result.suggested_correspondent,
        document_type=entry.task.result.suggested_document_type,
        reasoning=entry.task.result.reasoning,
    )


def generate_digest(
    audit_log: AuditLog,
    review_queue: ReviewQueue,
    *,
    email_audit_log: EmailAuditLog | None = None,
    email_review_queue: EmailReviewQueue | None = None,
    since: datetime | None = None,
    hours: int = 24,
) -> DailyDigest:
    """Generate a digest of pipeline activity.

    Args:
        audit_log: The audit log to read entries from.
        review_queue: The review queue to check pending count.
        email_audit_log: Optional email audit log for email stats.
        email_review_queue: Optional email review queue for pending count.
        since: Start of the digest period. Defaults to `hours` ago.
        hours: Fallback period in hours if `since` is not provided.

    Returns:
        A structured DailyDigest.
    """
    now = datetime.now(UTC)
    period_start = since or (now - timedelta(hours=hours))

    entries = audit_log.read_entries(since=period_start)

    auto_applied: list[DigestItem] = []
    flagged: list[DigestItem] = []
    queued: list[DigestItem] = []
    approved: list[DigestItem] = []
    rejected: list[DigestItem] = []

    for entry in entries:
        item = _entry_to_item(entry)
        if entry.action == "auto_applied":
            if entry.gate_action == GateAction.FLAG_IN_DIGEST:
                flagged.append(item)
            else:
                auto_applied.append(item)
        elif entry.action == "queued_for_review":
            queued.append(item)
        elif entry.action == "review_approved":
            approved.append(item)
        elif entry.action == "review_rejected":
            rejected.append(item)

    pending_count = review_queue.count_pending()

    # Email stats
    email_processed = 0
    email_auto = 0
    email_queued = 0
    email_approved_count = 0
    email_rejected_count = 0
    email_pending = 0

    if email_audit_log is not None:
        email_entries = email_audit_log.read_entries(since=period_start)
        email_processed = len(email_entries)
        for e_entry in email_entries:
            if e_entry.action == "auto_applied":
                email_auto += 1
            elif e_entry.action == "queued_for_review":
                email_queued += 1
            elif e_entry.action == "review_approved":
                email_approved_count += 1
            elif e_entry.action == "review_rejected":
                email_rejected_count += 1

    if email_review_queue is not None:
        email_pending = email_review_queue.count_pending()

    digest = DailyDigest(
        generated_at=now,
        period_start=period_start,
        period_end=now,
        auto_applied=auto_applied,
        flagged=flagged,
        queued_for_review=queued,
        review_approved=approved,
        review_rejected=rejected,
        pending_review_count=pending_count,
        email_processed=email_processed,
        email_auto_applied=email_auto,
        email_queued=email_queued,
        email_approved=email_approved_count,
        email_rejected=email_rejected_count,
        email_pending_review=email_pending,
    )

    logger.info(
        "Digest: %d processed, %d reviewed, %d pending",
        digest.total_processed,
        digest.total_reviewed,
        pending_count,
    )

    return digest


def render_text(digest: DailyDigest) -> str:
    """Render a digest as plain text / markdown."""
    lines: list[str] = []
    lines.append("# Corvus Daily Digest")
    lines.append("")
    lines.append(
        f"Period: {digest.period_start:%Y-%m-%d %H:%M} "
        f"to {digest.period_end:%Y-%m-%d %H:%M} UTC"
    )
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- Auto-applied: {len(digest.auto_applied)}")
    lines.append(f"- Flagged (applied, needs attention): {len(digest.flagged)}")
    lines.append(f"- Queued for review: {len(digest.queued_for_review)}")
    lines.append(f"- Reviews completed: {digest.total_reviewed}")
    lines.append(f"- Pending review: {digest.pending_review_count}")
    lines.append("")

    if digest.flagged:
        lines.append("## Flagged Items (Applied, Needs Attention)")
        for item in digest.flagged:
            _append_item(lines, item)
        lines.append("")

    if digest.auto_applied:
        lines.append("## Auto-Applied")
        for item in digest.auto_applied:
            _append_item(lines, item)
        lines.append("")

    if digest.queued_for_review:
        lines.append("## Queued for Review")
        for item in digest.queued_for_review:
            _append_item(lines, item)
        lines.append("")

    if digest.review_approved:
        lines.append("## Approved")
        for item in digest.review_approved:
            _append_item(lines, item)
        lines.append("")

    if digest.review_rejected:
        lines.append("## Rejected")
        for item in digest.review_rejected:
            _append_item(lines, item)
        lines.append("")

    # Email section
    if digest.email_processed > 0 or digest.email_pending_review > 0:
        lines.append("## Email Pipeline")
        lines.append(f"- Processed: {digest.email_processed}")
        lines.append(f"- Auto-applied: {digest.email_auto_applied}")
        lines.append(f"- Queued for review: {digest.email_queued}")
        lines.append(f"- Approved: {digest.email_approved}")
        lines.append(f"- Rejected: {digest.email_rejected}")
        lines.append(f"- Pending review: {digest.email_pending_review}")
        lines.append("")

    has_doc_activity = digest.total_processed > 0 or digest.total_reviewed > 0
    has_email_activity = digest.email_processed > 0 or digest.email_pending_review > 0
    if not has_doc_activity and not has_email_activity:
        lines.append("No activity during this period.")
        lines.append("")

    return "\n".join(lines)


def _append_item(lines: list[str], item: DigestItem) -> None:
    """Append a formatted digest item to the lines list."""
    lines.append(f"- **[{item.document_id}] {item.document_title}**")
    lines.append(f"  Confidence: {item.confidence:.0%} | Gate: {item.gate_action.value}")
    if item.tags:
        lines.append(f"  Tags: {', '.join(item.tags)}")
    if item.correspondent:
        lines.append(f"  Correspondent: {item.correspondent}")
    if item.document_type:
        lines.append(f"  Type: {item.document_type}")
