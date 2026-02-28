"""SQLite-backed review queue for email triage tasks.

Stores email actions that failed the confidence gate and require human approval.
Uses stdlib sqlite3 — same pattern as the document review queue but with a
separate table for email actions.
"""

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from corvus.schemas.document_tagging import ReviewStatus
from corvus.schemas.email import EmailReviewQueueItem, EmailTriageTask

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS email_review_queue (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    uid             TEXT NOT NULL,
    account_email   TEXT NOT NULL,
    subject         TEXT NOT NULL,
    from_address    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    task_json       TEXT NOT NULL,
    reviewed_at     TEXT,
    reviewer_notes  TEXT
)
"""

_INSERT = """
INSERT INTO email_review_queue
    (id, created_at, uid, account_email, subject, from_address, status, task_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_BY_ID = "SELECT * FROM email_review_queue WHERE id = ?"
_SELECT_PENDING = (
    "SELECT * FROM email_review_queue WHERE status = 'pending' ORDER BY created_at ASC"
)
_SELECT_ALL = "SELECT * FROM email_review_queue ORDER BY created_at DESC LIMIT ?"

_UPDATE_STATUS = """
UPDATE email_review_queue SET status = ?, reviewed_at = ?, reviewer_notes = ? WHERE id = ?
"""


def _row_to_item(row: sqlite3.Row) -> EmailReviewQueueItem:
    """Convert a database row to an EmailReviewQueueItem."""
    return EmailReviewQueueItem(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        task=EmailTriageTask.model_validate_json(row["task_json"]),
        status=ReviewStatus(row["status"]),
        reviewed_at=(
            datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None
        ),
        reviewer_notes=row["reviewer_notes"],
    )


class EmailReviewQueue:
    """SQLite-backed queue for email triage actions awaiting review.

    Usage::

        queue = EmailReviewQueue("/path/to/email_queue.db")
        queue.add(task)

        for item in queue.list_pending():
            print(item.task.subject)

        queue.approve(item_id, notes="Looks correct")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmailReviewQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, task: EmailTriageTask) -> EmailReviewQueueItem:
        """Add a new email triage task to the review queue.

        Returns:
            The created EmailReviewQueueItem with a generated ID and timestamp.
        """
        item_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        self._conn.execute(
            _INSERT,
            (
                item_id,
                now.isoformat(),
                task.uid,
                task.account_email,
                task.subject,
                task.from_address,
                ReviewStatus.PENDING.value,
                task.model_dump_json(),
            ),
        )
        self._conn.commit()

        logger.info(
            "Queued email %s for review (queue_id=%s, subject=%s)",
            task.uid,
            item_id,
            task.subject,
        )

        return EmailReviewQueueItem(
            id=item_id,
            created_at=now,
            task=task,
        )

    def get(self, item_id: str) -> EmailReviewQueueItem | None:
        """Fetch a single queue item by ID."""
        row = self._conn.execute(_SELECT_BY_ID, (item_id,)).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def list_pending(self) -> list[EmailReviewQueueItem]:
        """List all items awaiting review, oldest first."""
        rows = self._conn.execute(_SELECT_PENDING).fetchall()
        return [_row_to_item(r) for r in rows]

    def list_all(self, limit: int = 50) -> list[EmailReviewQueueItem]:
        """List all items, newest first."""
        rows = self._conn.execute(_SELECT_ALL, (limit,)).fetchall()
        return [_row_to_item(r) for r in rows]

    def approve(self, item_id: str, *, notes: str | None = None) -> EmailReviewQueueItem:
        """Mark an item as approved.

        Raises:
            ValueError: If the item doesn't exist or isn't pending.
        """
        return self._set_status(item_id, ReviewStatus.APPROVED, notes)

    def reject(self, item_id: str, *, notes: str | None = None) -> EmailReviewQueueItem:
        """Mark an item as rejected.

        Raises:
            ValueError: If the item doesn't exist or isn't pending.
        """
        return self._set_status(item_id, ReviewStatus.REJECTED, notes)

    def count_pending(self) -> int:
        """Return the number of items awaiting review."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM email_review_queue WHERE status = 'pending'"
        ).fetchone()
        return row[0]

    def _set_status(
        self,
        item_id: str,
        status: ReviewStatus,
        notes: str | None,
    ) -> EmailReviewQueueItem:
        """Update the status of a queue item."""
        item = self.get(item_id)
        if item is None:
            raise ValueError(f"Queue item not found: {item_id}")
        if item.status != ReviewStatus.PENDING:
            raise ValueError(
                f"Cannot {status.value} item {item_id}: "
                f"current status is {item.status.value}"
            )

        now = datetime.now(UTC)
        self._conn.execute(_UPDATE_STATUS, (status.value, now.isoformat(), notes, item_id))
        self._conn.commit()

        logger.info("Email review item %s: %s", item_id, status.value)

        item.status = status
        item.reviewed_at = now
        item.reviewer_notes = notes
        return item
