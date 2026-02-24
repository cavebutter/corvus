"""Core file transfer logic for the scan folder watchdog.

Handles hashing, duplicate detection, and two transfer methods:
  - move: direct file move to a mounted Paperless consume directory
  - upload: POST via Paperless-ngx REST API
"""

import hashlib
import logging
import shutil
from pathlib import Path

from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.watchdog import TransferMethod, TransferStatus, WatchdogEvent
from corvus.watchdog.audit import WatchdogAuditLog
from corvus.watchdog.hash_store import HashStore

logger = logging.getLogger(__name__)

HASH_CHUNK_SIZE = 65536  # 64 KB chunks for hashing


def compute_file_hash(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


def transfer_by_move(source: Path, consume_dir: Path) -> Path:
    """Move a file to the Paperless consume directory.

    If a file with the same name already exists at the destination,
    appends a numeric suffix (e.g. ``doc_1.pdf``, ``doc_2.pdf``).

    Returns:
        The final destination path.
    """
    dest = consume_dir / source.name

    # Handle name collisions
    if dest.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while dest.exists():
            dest = consume_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.move(str(source), str(dest))
    return dest


async def transfer_by_upload(source: Path, client: PaperlessClient) -> str:
    """Upload a file to Paperless-ngx via the REST API.

    Returns:
        The Paperless task UUID.
    """
    return await client.upload_document(source)


async def process_file(
    file_path: Path,
    *,
    method: TransferMethod,
    hash_store: HashStore,
    audit_log: WatchdogAuditLog,
    consume_dir: Path | None = None,
    paperless_client: PaperlessClient | None = None,
) -> WatchdogEvent:
    """Process a single file: hash, dedup check, transfer, audit.

    Args:
        file_path: Path to the file to process.
        method: Transfer method (move or upload).
        hash_store: For duplicate detection.
        audit_log: For recording the event.
        consume_dir: Required if method is MOVE.
        paperless_client: Required if method is UPLOAD.

    Returns:
        The WatchdogEvent recording what happened.
    """
    from datetime import UTC, datetime

    file_hash = compute_file_hash(file_path)
    file_size = file_path.stat().st_size

    # Duplicate check
    if hash_store.contains(file_hash):
        logger.info("Duplicate detected: %s (hash=%s…)", file_path.name, file_hash[:12])
        event = WatchdogEvent(
            timestamp=datetime.now(UTC),
            source_path=str(file_path),
            file_name=file_path.name,
            file_hash=file_hash,
            transfer_method=method,
            transfer_status=TransferStatus.DUPLICATE,
            file_size_bytes=file_size,
        )
        audit_log.log(event)
        return event

    # Transfer
    destination = ""
    try:
        if method == TransferMethod.MOVE:
            if consume_dir is None:
                raise ValueError("consume_dir is required for move transfer method")
            dest_path = transfer_by_move(file_path, consume_dir)
            destination = str(dest_path)
            logger.info("Moved %s → %s", file_path.name, dest_path)
        elif method == TransferMethod.UPLOAD:
            if paperless_client is None:
                raise ValueError("paperless_client is required for upload transfer method")
            task_uuid = await transfer_by_upload(file_path, paperless_client)
            destination = task_uuid
            logger.info("Uploaded %s → task %s", file_path.name, task_uuid)
        else:
            raise ValueError(f"Unknown transfer method: {method}")

        hash_store.add(file_hash, str(file_path))

        event = WatchdogEvent(
            timestamp=datetime.now(UTC),
            source_path=str(file_path),
            file_name=file_path.name,
            file_hash=file_hash,
            transfer_method=method,
            transfer_status=TransferStatus.SUCCESS,
            destination=destination,
            file_size_bytes=file_size,
        )
        audit_log.log(event)
        return event

    except Exception as exc:
        logger.exception("Failed to transfer %s: %s", file_path.name, exc)
        event = WatchdogEvent(
            timestamp=datetime.now(UTC),
            source_path=str(file_path),
            file_name=file_path.name,
            file_hash=file_hash,
            transfer_method=method,
            transfer_status=TransferStatus.ERROR,
            error_message=str(exc),
            file_size_bytes=file_size,
        )
        audit_log.log(event)
        return event
