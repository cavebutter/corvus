"""SQLite-backed SHA-256 hash store for duplicate file detection.

Tracks which files have already been processed by the watchdog so that
restarts, re-scans, or scanner re-deposits don't cause duplicate uploads.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class HashStore:
    """Persistent store mapping file SHA-256 hashes to processing records.

    Usage::

        store = HashStore("/path/to/hashes.db")
        if store.contains("abcdef1234..."):
            print("Already processed")
        else:
            store.add("abcdef1234...", "/scans/doc.pdf")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                sha256      TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                file_name   TEXT NOT NULL,
                processed_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "HashStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def contains(self, sha256: str) -> bool:
        """Check if a hash has already been recorded."""
        row = self._conn.execute(
            "SELECT 1 FROM file_hashes WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row is not None

    def add(self, sha256: str, source_path: str) -> None:
        """Record a file hash as processed.

        Raises sqlite3.IntegrityError if the hash already exists.
        """
        self._conn.execute(
            "INSERT INTO file_hashes (sha256, source_path, file_name, processed_at)"
            " VALUES (?, ?, ?, ?)",
            (sha256, source_path, Path(source_path).name, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def get(self, sha256: str) -> dict | None:
        """Retrieve the record for a hash, or None if not found."""
        row = self._conn.execute(
            "SELECT sha256, source_path, file_name, processed_at FROM file_hashes WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        return {
            "sha256": row[0],
            "source_path": row[1],
            "file_name": row[2],
            "processed_at": row[3],
        }

    def count(self) -> int:
        """Return the total number of recorded hashes."""
        row = self._conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()
        return row[0]

    def remove(self, sha256: str) -> bool:
        """Remove a hash record. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM file_hashes WHERE sha256 = ?", (sha256,))
        self._conn.commit()
        return cursor.rowcount > 0
