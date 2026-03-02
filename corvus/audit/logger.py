"""Append-only audit log for the document tagging pipeline.

Writes AuditEntry records as JSON Lines (one JSON object per line).
Every action — auto-applied, queued, approved, rejected — is logged here.
"""

import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from corvus.schemas.document_tagging import (
    AuditEntry,
    DocumentTaggingTask,
    ProposedDocumentUpdate,
)

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only JSONL audit log.

    Usage::

        audit = AuditLog("/path/to/audit.jsonl")
        audit.log_auto_applied(task, proposed_update)

        entries = audit.read_entries(since=some_datetime)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        """Append a single audit entry to the log file."""
        with self._path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")
        logger.debug(
            "Audit: %s doc=%d applied=%s",
            entry.action,
            entry.document_id,
            entry.applied,
        )

    def log_auto_applied(
        self,
        task: DocumentTaggingTask,
        proposed_update: ProposedDocumentUpdate,
    ) -> AuditEntry:
        """Log an auto-applied action (confidence >= threshold)."""
        entry = self._build_entry("auto_applied", task, proposed_update, applied=True)
        self.log(entry)
        return entry

    def log_queued_for_review(
        self,
        task: DocumentTaggingTask,
        proposed_update: ProposedDocumentUpdate,
    ) -> AuditEntry:
        """Log an action that was queued for human review."""
        entry = self._build_entry("queued_for_review", task, proposed_update, applied=False)
        self.log(entry)
        return entry

    def log_review_approved(
        self,
        task: DocumentTaggingTask,
        proposed_update: ProposedDocumentUpdate,
    ) -> AuditEntry:
        """Log a review approval (human confirmed the action)."""
        entry = self._build_entry("review_approved", task, proposed_update, applied=True)
        self.log(entry)
        return entry

    def log_review_rejected(
        self,
        task: DocumentTaggingTask,
        proposed_update: ProposedDocumentUpdate,
    ) -> AuditEntry:
        """Log a review rejection (human rejected the action)."""
        entry = self._build_entry("review_rejected", task, proposed_update, applied=False)
        self.log(entry)
        return entry

    def read_entries(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        """Read audit entries, optionally filtered by timestamp.

        Args:
            since: Only return entries after this timestamp.
            limit: Maximum number of entries to return (newest first after filtering).

        Returns:
            List of AuditEntry objects, oldest first.
        """
        if not self._path.exists():
            return []

        entries: list[AuditEntry] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = AuditEntry.model_validate_json(line)
                if since and entry.timestamp <= since:
                    continue
                entries.append(entry)

        if limit is not None:
            entries = entries[-limit:]

        return entries

    def purge_before(self, cutoff: datetime) -> int:
        """Remove entries older than *cutoff*. Returns count of purged entries.

        Uses atomic rewrite (temp file + os.replace) to avoid corruption.
        Skips the rewrite entirely if nothing would be purged.
        """
        if not self._path.exists():
            return 0

        keep_lines: list[str] = []
        purged = 0
        with self._path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = AuditEntry.model_validate_json(stripped)
                if entry.timestamp < cutoff:
                    purged += 1
                else:
                    keep_lines.append(stripped)

        if purged == 0:
            return 0

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            os.write(fd, "".join(ln + "\n" for ln in keep_lines).encode())
        finally:
            os.close(fd)
        try:
            os.replace(tmp_path, self._path)
        except BaseException:
            os.unlink(tmp_path)
            raise

        logger.info("Purged %d audit entries older than %s", purged, cutoff)
        return purged

    def _build_entry(
        self,
        action: str,
        task: DocumentTaggingTask,
        proposed_update: ProposedDocumentUpdate,
        *,
        applied: bool,
    ) -> AuditEntry:
        return AuditEntry(
            timestamp=datetime.now(UTC),
            action=action,
            document_id=task.document_id,
            document_title=task.document_title,
            task=task,
            proposed_update=proposed_update,
            gate_action=task.gate_action,
            applied=applied,
        )
