"""Conversation history for the chat REPL, with optional SQLite persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from corvus.schemas.email import EmailSummaryResult, EmailTriageResult
from corvus.schemas.orchestrator import (
    FetchPipelineResult,
    OrchestratorAction,
    OrchestratorResponse,
    StatusResult,
    TagPipelineResult,
    WebSearchResult,
)

if TYPE_CHECKING:
    from corvus.orchestrator.conversation_store import ConversationStore


class ConversationHistory:
    """Conversation history with optional SQLite persistence.

    Stores user/assistant message pairs and provides helpers for
    building Ollama-compatible message lists and formatted context strings
    for the intent classifier.

    When a ``store`` and ``conversation_id`` are set (via constructor or
    :meth:`set_persistence`), messages are also written to SQLite.
    """

    def __init__(
        self,
        max_turns: int = 20,
        *,
        store: ConversationStore | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns
        self._store = store
        self._conversation_id = conversation_id

    @classmethod
    def from_store(
        cls,
        store: ConversationStore,
        conversation_id: str,
        max_turns: int = 20,
    ) -> ConversationHistory:
        """Load an existing conversation from the store.

        All messages are loaded from SQLite; only the most recent
        ``max_turns * 2`` are kept in memory for the LLM context window.
        """
        instance = cls(max_turns=max_turns, store=store, conversation_id=conversation_id)
        all_messages = store.load_messages(conversation_id)
        max_messages = max_turns * 2
        instance._messages = all_messages[-max_messages:]
        return instance

    def set_persistence(self, store: ConversationStore, conversation_id: str) -> None:
        """Enable persistence after construction (deferred creation pattern)."""
        self._store = store
        self._conversation_id = conversation_id

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()
        if self._store and self._conversation_id:
            self._store.add_message(self._conversation_id, "user", content)

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._trim()
        if self._store and self._conversation_id:
            self._store.add_message(self._conversation_id, "assistant", content)

    def get_messages(self) -> list[dict[str, str]]:
        """Return Ollama-compatible message list."""
        return list(self._messages)

    def get_recent_context(self, max_turns: int = 5) -> str:
        """Return formatted recent conversation for the intent classifier.

        A "turn" is one user+assistant pair. Returns the most recent
        *max_turns* pairs as ``User: ...\\nCorvus: ...`` lines.
        """
        if not self._messages:
            return ""

        # Collect recent messages (up to max_turns * 2 items)
        recent = self._messages[-(max_turns * 2):]
        lines: list[str] = []
        for msg in recent:
            if msg["role"] == "user":
                lines.append(f"User: {msg['content']}")
            else:
                lines.append(f"Corvus: {msg['content']}")
        return "\n".join(lines)

    def _trim(self) -> None:
        """Drop oldest messages when exceeding max_turns (pairs)."""
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]


def summarize_response(response: OrchestratorResponse) -> str:
    """Create a brief conversation-appropriate summary of a response.

    Used to store assistant turns in conversation history without
    dumping entire pipeline results.
    """
    if response.action == OrchestratorAction.CHAT_RESPONSE:
        return response.message

    if response.action in (
        OrchestratorAction.NEEDS_CLARIFICATION,
        OrchestratorAction.INTERACTIVE_REQUIRED,
    ):
        return response.message

    if response.result is None:
        return response.message or "Done."

    result = response.result

    if isinstance(result, FetchPipelineResult):
        if not result.documents:
            return "No documents found."
        titles = ", ".join(d.get("title", "?") for d in result.documents[:5])
        return f"Found {result.documents_found} document(s): {titles}"

    if isinstance(result, TagPipelineResult):
        return (
            f"Tagged {result.processed} document(s), "
            f"{result.queued} queued for review."
        )

    if isinstance(result, StatusResult):
        return (
            f"Status: {result.pending_count} pending, "
            f"{result.processed_24h} processed, "
            f"{result.reviewed_24h} reviewed in last 24h."
        )

    if isinstance(result, WebSearchResult):
        return result.summary

    if isinstance(result, EmailTriageResult):
        r = result
        return (
            f"Processed {r.processed} emails. "
            f"{r.auto_acted} auto-applied, {r.queued} queued for review."
        )

    if isinstance(result, EmailSummaryResult):
        return result.summary

    return response.message or "Done."
