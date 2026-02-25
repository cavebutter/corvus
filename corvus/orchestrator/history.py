"""In-memory conversation history for the chat REPL (V1 — lost on exit)."""

from __future__ import annotations

from corvus.schemas.orchestrator import (
    FetchPipelineResult,
    OrchestratorAction,
    OrchestratorResponse,
    StatusResult,
    TagPipelineResult,
    WebSearchResult,
)


class ConversationHistory:
    """Simple in-memory conversation history.

    Stores user/assistant message pairs and provides helpers for
    building Ollama-compatible message lists and formatted context strings
    for the intent classifier.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

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

    return response.message or "Done."
