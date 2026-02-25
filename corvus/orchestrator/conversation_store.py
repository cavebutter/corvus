"""SQLite-backed conversation persistence for the chat REPL.

Stores conversations and messages so users can resume sessions across
restarts. Uses stdlib sqlite3 — follows the same pattern as ReviewQueue.
"""

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
ON messages(conversation_id)
"""

_INSERT_CONVERSATION = """
INSERT INTO conversations (id, title, created_at, updated_at)
VALUES (?, ?, ?, ?)
"""

_INSERT_MESSAGE = """
INSERT INTO messages (conversation_id, role, content, created_at)
VALUES (?, ?, ?, ?)
"""

_UPDATE_UPDATED_AT = """
UPDATE conversations SET updated_at = ? WHERE id = ?
"""

_SELECT_MESSAGES = """
SELECT role, content FROM messages
WHERE conversation_id = ?
ORDER BY id ASC
"""

_SELECT_MOST_RECENT = """
SELECT id FROM conversations
ORDER BY updated_at DESC
LIMIT 1
"""

_SELECT_CONVERSATIONS = """
SELECT c.id, c.title, c.created_at, c.updated_at,
       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
FROM conversations c
ORDER BY c.updated_at DESC
LIMIT ?
"""

_SELECT_CONVERSATION_BY_ID = """
SELECT c.id, c.title, c.created_at, c.updated_at,
       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
FROM conversations c
WHERE c.id = ?
"""

_SELECT_CONVERSATION_BY_PREFIX = """
SELECT c.id, c.title, c.created_at, c.updated_at,
       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
FROM conversations c
WHERE c.id LIKE ?
ORDER BY c.updated_at DESC
LIMIT 1
"""


def _generate_title(first_message: str, max_length: int = 60) -> str:
    """Generate a conversation title from the first user message.

    Truncates on a word boundary at *max_length* characters.
    """
    text = first_message.strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


class ConversationStore:
    """SQLite-backed store for chat conversations.

    Usage::

        with ConversationStore("data/conversations.db") as store:
            conv_id = store.create("Hello, Corvus!")
            store.add_message(conv_id, "user", "Hello, Corvus!")
            store.add_message(conv_id, "assistant", "Hi there!")
            messages = store.load_messages(conv_id)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_CONVERSATIONS)
        self._conn.execute(_CREATE_MESSAGES)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ConversationStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create(self, first_message: str) -> str:
        """Create a new conversation and return its UUID.

        The title is auto-generated from the first user message.
        """
        conv_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        title = _generate_title(first_message)

        self._conn.execute(_INSERT_CONVERSATION, (conv_id, title, now, now))
        self._conn.commit()

        logger.info("Created conversation %s: %s", conv_id[:8], title)
        return conv_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Add a message to a conversation and update its timestamp."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(_INSERT_MESSAGE, (conversation_id, role, content, now))
        self._conn.execute(_UPDATE_UPDATED_AT, (now, conversation_id))
        self._conn.commit()

    def load_messages(self, conversation_id: str) -> list[dict[str, str]]:
        """Load all messages for a conversation, ordered by insertion.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        rows = self._conn.execute(_SELECT_MESSAGES, (conversation_id,)).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_most_recent(self) -> str | None:
        """Return the ID of the most recently updated conversation, or None."""
        row = self._conn.execute(_SELECT_MOST_RECENT).fetchone()
        if row is None:
            return None
        return row["id"]

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """List recent conversations with metadata.

        Returns:
            List of dicts with keys: id, title, created_at, updated_at, message_count.
        """
        rows = self._conn.execute(_SELECT_CONVERSATIONS, (limit,)).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Fetch a single conversation's metadata.

        Accepts a full UUID or a prefix (first N characters).

        Returns:
            Dict with id, title, created_at, updated_at, message_count — or None.
        """
        # Try exact match first
        row = self._conn.execute(
            _SELECT_CONVERSATION_BY_ID, (conversation_id,)
        ).fetchone()
        if row is not None:
            return {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }

        # Try prefix match
        row = self._conn.execute(
            _SELECT_CONVERSATION_BY_PREFIX, (conversation_id + "%",)
        ).fetchone()
        if row is not None:
            return {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }

        return None
