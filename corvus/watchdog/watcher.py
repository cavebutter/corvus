"""Filesystem watcher for the scan folder watchdog.

Uses the ``watchdog`` library (inotify on Linux) to detect new files.
The Observer runs in a background thread and schedules async ``process_file``
calls onto the asyncio event loop.
"""

import asyncio
import logging
import time
from pathlib import Path

from watchdog.events import FileClosedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.watchdog import TransferMethod
from corvus.watchdog.audit import WatchdogAuditLog
from corvus.watchdog.hash_store import HashStore
from corvus.watchdog.transfer import process_file

logger = logging.getLogger(__name__)

# Debounce window — ignore events for the same file within this period
DEBOUNCE_SECONDS = 2.0


class ScanFolderHandler(FileSystemEventHandler):
    """Handles filesystem events in the scan directory.

    Uses ``on_closed`` (inotify IN_CLOSE_WRITE) as the primary trigger,
    meaning the file is complete when the writer closes the handle.
    Falls back to ``on_created`` with a debounce for platforms without
    IN_CLOSE_WRITE support.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        method: TransferMethod,
        hash_store: HashStore,
        audit_log: WatchdogAuditLog,
        file_patterns: list[str],
        consume_dir: Path | None = None,
        paperless_client: PaperlessClient | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._method = method
        self._hash_store = hash_store
        self._audit_log = audit_log
        self._file_patterns = [p.lower() for p in file_patterns]
        self._consume_dir = consume_dir
        self._paperless_client = paperless_client
        self._last_seen: dict[str, float] = {}

    def _matches_pattern(self, path: Path) -> bool:
        """Check if a file matches configured patterns (by suffix)."""
        name_lower = path.name.lower()
        for pattern in self._file_patterns:
            # Pattern like "*.pdf" — match by suffix
            if pattern.startswith("*."):
                if name_lower.endswith(pattern[1:]):
                    return True
            elif name_lower == pattern:
                return True
        return False

    def _should_debounce(self, path_str: str) -> bool:
        """Return True if this file was seen too recently."""
        now = time.monotonic()
        last = self._last_seen.get(path_str, 0.0)
        if now - last < DEBOUNCE_SECONDS:
            return True
        self._last_seen[path_str] = now
        return False

    def _schedule_processing(self, src_path: str) -> None:
        """Schedule async process_file on the event loop."""
        path = Path(src_path)

        if not path.is_file():
            return
        if not self._matches_pattern(path):
            return
        if self._should_debounce(src_path):
            return

        logger.info("Detected file: %s", path.name)
        asyncio.run_coroutine_threadsafe(
            process_file(
                path,
                method=self._method,
                hash_store=self._hash_store,
                audit_log=self._audit_log,
                consume_dir=self._consume_dir,
                paperless_client=self._paperless_client,
            ),
            self._loop,
        )

    def on_closed(self, event: FileClosedEvent) -> None:
        """Triggered when a file is closed after writing (inotify)."""
        if event.is_directory:
            return
        self._schedule_processing(event.src_path)

    def on_created(self, event) -> None:
        """Fallback for platforms without on_closed support.

        Also handles files deposited by atomic move (rename) into the dir.
        """
        if event.is_directory:
            return
        self._schedule_processing(event.src_path)


async def scan_existing(
    scan_dir: Path,
    *,
    method: TransferMethod,
    hash_store: HashStore,
    audit_log: WatchdogAuditLog,
    file_patterns: list[str],
    consume_dir: Path | None = None,
    paperless_client: PaperlessClient | None = None,
) -> list:
    """Scan existing files in the directory (--once mode).

    Returns a list of WatchdogEvent results.
    """
    results = []
    patterns_lower = [p.lower() for p in file_patterns]

    for item in sorted(scan_dir.iterdir()):
        if not item.is_file():
            continue

        name_lower = item.name.lower()
        matched = False
        for pattern in patterns_lower:
            if (
                (pattern.startswith("*.") and name_lower.endswith(pattern[1:]))
                or name_lower == pattern
            ):
                matched = True
                break

        if not matched:
            continue

        event = await process_file(
            item,
            method=method,
            hash_store=hash_store,
            audit_log=audit_log,
            consume_dir=consume_dir,
            paperless_client=paperless_client,
        )
        results.append(event)

    return results


async def watch_folder(
    scan_dir: Path,
    *,
    method: TransferMethod,
    hash_store: HashStore,
    audit_log: WatchdogAuditLog,
    file_patterns: list[str],
    consume_dir: Path | None = None,
    paperless_client: PaperlessClient | None = None,
) -> None:
    """Watch a folder for new files and process them continuously.

    Runs until interrupted (KeyboardInterrupt / cancellation).
    """
    loop = asyncio.get_running_loop()

    handler = ScanFolderHandler(
        loop=loop,
        method=method,
        hash_store=hash_store,
        audit_log=audit_log,
        file_patterns=file_patterns,
        consume_dir=consume_dir,
        paperless_client=paperless_client,
    )

    observer = Observer()
    observer.schedule(handler, str(scan_dir), recursive=False)
    observer.start()
    logger.info("Watching %s for new files…", scan_dir)

    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        observer.stop()
        observer.join()
        logger.info("Watcher stopped.")
