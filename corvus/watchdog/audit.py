"""Append-only JSONL audit log for the scan folder watchdog.

Separate from the tagging audit log — watchdog events have a different shape
(file-oriented rather than document/tag-oriented).
"""

import logging
import os
import tempfile
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
                event = WatchdogEvent.model_validate_json(stripped)
                if event.timestamp < cutoff:
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

        logger.info("Purged %d watchdog audit entries older than %s", purged, cutoff)
        return purged
