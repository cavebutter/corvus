"""Append-only audit log for the email triage pipeline.

Writes EmailAuditEntry records as JSON Lines (one JSON object per line).
Separate from the document tagging audit log to keep concerns isolated.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from corvus.schemas.email import EmailAuditEntry, EmailTriageTask

logger = logging.getLogger(__name__)


class EmailAuditLog:
    """Append-only JSONL audit log for email actions.

    Usage::

        audit = EmailAuditLog("/path/to/email_audit.jsonl")
        audit.log_auto_applied(task)

        entries = audit.read_entries(since=some_datetime)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: EmailAuditEntry) -> None:
        """Append a single audit entry to the log file."""
        with self._path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")
        logger.debug(
            "Email audit: %s uid=%s subject=%s applied=%s",
            entry.action,
            entry.uid,
            entry.subject,
            entry.applied,
        )

    def log_auto_applied(self, task: EmailTriageTask) -> EmailAuditEntry:
        """Log an auto-applied action (confidence >= threshold)."""
        entry = self._build_entry("auto_applied", task, applied=True)
        self.log(entry)
        return entry

    def log_queued_for_review(self, task: EmailTriageTask) -> EmailAuditEntry:
        """Log an action that was queued for human review."""
        entry = self._build_entry("queued_for_review", task, applied=False)
        self.log(entry)
        return entry

    def log_review_approved(self, task: EmailTriageTask) -> EmailAuditEntry:
        """Log a review approval (human confirmed the action)."""
        entry = self._build_entry("review_approved", task, applied=True)
        self.log(entry)
        return entry

    def log_review_rejected(self, task: EmailTriageTask) -> EmailAuditEntry:
        """Log a review rejection (human rejected the action)."""
        entry = self._build_entry("review_rejected", task, applied=False)
        self.log(entry)
        return entry

    def log_sender_list_applied(self, task: EmailTriageTask) -> EmailAuditEntry:
        """Log an action applied via sender list (deterministic, no LLM)."""
        entry = self._build_entry("sender_list_applied", task, applied=True)
        self.log(entry)
        return entry

    def read_entries(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[EmailAuditEntry]:
        """Read audit entries, optionally filtered by timestamp.

        Args:
            since: Only return entries after this timestamp.
            limit: Maximum number of entries to return (newest first after filtering).

        Returns:
            List of EmailAuditEntry objects, oldest first.
        """
        if not self._path.exists():
            return []

        entries: list[EmailAuditEntry] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = EmailAuditEntry.model_validate_json(line)
                if since and entry.timestamp <= since:
                    continue
                entries.append(entry)

        if limit is not None:
            entries = entries[-limit:]

        return entries

    def _build_entry(
        self,
        action: str,
        task: EmailTriageTask,
        *,
        applied: bool,
    ) -> EmailAuditEntry:
        return EmailAuditEntry(
            timestamp=datetime.now(UTC),
            action=action,
            account_email=task.account_email,
            uid=task.uid,
            subject=task.subject,
            from_address=task.from_address,
            category=task.classification.category,
            email_action=task.proposed_action,
            gate_action=task.gate_action,
            applied=applied,
        )
