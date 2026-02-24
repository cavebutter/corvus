"""Append-only JSONL audit log for the scan folder watchdog.

Separate from the tagging audit log — watchdog events have a different shape
(file-oriented rather than document/tag-oriented).
"""

import logging
from datetime import datetime
from pathlib import Path

from corvus.schemas.watchdog import TransferStatus, WatchdogEvent

logger = logging.getLogger(__name__)


class WatchdogAuditLog:
    """Append-only JSONL audit log for watchdog file events.

    Usage::

        audit = WatchdogAuditLog("/path/to/watchdog_audit.jsonl")
        audit.log(event)
        entries = audit.read_entries(since=some_datetime)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: WatchdogEvent) -> None:
        """Append a single event to the log file."""
        with self._path.open("a") as f:
            f.write(event.model_dump_json() + "\n")
        logger.debug(
            "Watchdog audit: %s %s status=%s",
            event.file_name,
            event.transfer_method,
            event.transfer_status,
        )

    def read_entries(
        self,
        *,
        since: datetime | None = None,
        status: TransferStatus | None = None,
        limit: int | None = None,
    ) -> list[WatchdogEvent]:
        """Read audit entries with optional filtering.

        Args:
            since: Only return entries after this timestamp.
            status: Only return entries with this transfer status.
            limit: Maximum number of entries to return (newest after filtering).

        Returns:
            List of WatchdogEvent objects, oldest first.
        """
        if not self._path.exists():
            return []

        entries: list[WatchdogEvent] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = WatchdogEvent.model_validate_json(line)
                if since and event.timestamp <= since:
                    continue
                if status and event.transfer_status != status:
                    continue
                entries.append(event)

        if limit is not None:
            entries = entries[-limit:]

        return entries
