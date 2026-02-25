"""Tests for ConversationHistory and summarize_response."""

import pytest

from corvus.orchestrator.conversation_store import ConversationStore
from corvus.orchestrator.history import ConversationHistory, summarize_response
from corvus.schemas.orchestrator import (
    FetchPipelineResult,
    Intent,
    OrchestratorAction,
    OrchestratorResponse,
    StatusResult,
    TagPipelineResult,
    WebSearchResult,
)


# ------------------------------------------------------------------
# ConversationHistory basics
# ------------------------------------------------------------------


class TestConversationHistory:
    def test_empty_history(self):
        h = ConversationHistory()
        assert h.get_messages() == []

    def test_add_messages(self):
        h = ConversationHistory()
        h.add_user_message("hello")
        h.add_assistant_message("hi there")
        msgs = h.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def test_turn_count_and_trim(self):
        h = ConversationHistory(max_turns=2)
        # Add 3 turns (6 messages), should trim to 2 turns (4 messages)
        for i in range(3):
            h.add_user_message(f"user {i}")
            h.add_assistant_message(f"assistant {i}")
        msgs = h.get_messages()
        assert len(msgs) == 4
        # Oldest turn (user 0, assistant 0) should be trimmed
        assert msgs[0]["content"] == "user 1"
        assert msgs[1]["content"] == "assistant 1"

    def test_get_messages_returns_copy(self):
        h = ConversationHistory()
        h.add_user_message("test")
        msgs = h.get_messages()
        msgs.clear()
        assert len(h.get_messages()) == 1


# ------------------------------------------------------------------
# get_recent_context
# ------------------------------------------------------------------


class TestGetRecentContext:
    def test_empty_context(self):
        h = ConversationHistory()
        assert h.get_recent_context() == ""

    def test_formatting(self):
        h = ConversationHistory()
        h.add_user_message("find my invoices")
        h.add_assistant_message("Found 3 document(s): Invoice A, Invoice B, Invoice C")
        context = h.get_recent_context()
        assert "User: find my invoices" in context
        assert "Corvus: Found 3 document(s)" in context

    def test_max_turns_limit(self):
        h = ConversationHistory()
        for i in range(10):
            h.add_user_message(f"msg {i}")
            h.add_assistant_message(f"reply {i}")
        context = h.get_recent_context(max_turns=2)
        # Should only contain the last 2 turns (4 lines)
        lines = context.strip().split("\n")
        assert len(lines) == 4
        assert "msg 8" in lines[0]
        assert "reply 9" in lines[-1]


# ------------------------------------------------------------------
# summarize_response
# ------------------------------------------------------------------


class TestSummarizeResponse:
    def test_chat_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.CHAT_RESPONSE,
            intent=Intent.GENERAL_CHAT,
            message="Hello! I'm Corvus.",
        )
        assert summarize_response(resp) == "Hello! I'm Corvus."

    def test_fetch_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.FETCH_DOCUMENT,
            result=FetchPipelineResult(
                documents_found=3,
                documents=[
                    {"title": "Invoice A"},
                    {"title": "Invoice B"},
                    {"title": "Invoice C"},
                ],
            ),
        )
        summary = summarize_response(resp)
        assert "3 document(s)" in summary
        assert "Invoice A" in summary
        assert "Invoice B" in summary

    def test_fetch_no_results(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.FETCH_DOCUMENT,
            result=FetchPipelineResult(documents_found=0, documents=[]),
        )
        assert "No documents found" in summarize_response(resp)

    def test_tag_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.TAG_DOCUMENTS,
            result=TagPipelineResult(processed=5, queued=5, auto_applied=0, errors=0),
        )
        summary = summarize_response(resp)
        assert "Tagged 5" in summary
        assert "5 queued" in summary

    def test_status_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.SHOW_STATUS,
            result=StatusResult(pending_count=2, processed_24h=10, reviewed_24h=8),
        )
        summary = summarize_response(resp)
        assert "2 pending" in summary
        assert "10 processed" in summary
        assert "8 reviewed" in summary

    def test_web_search_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.WEB_SEARCH,
            result=WebSearchResult(
                summary="It's sunny in NYC.",
                sources=[],
                query="weather",
            ),
        )
        assert summarize_response(resp) == "It's sunny in NYC."

    def test_clarification_response(self):
        resp = OrchestratorResponse(
            action=OrchestratorAction.NEEDS_CLARIFICATION,
            message="Could you rephrase?",
        )
        assert summarize_response(resp) == "Could you rephrase?"


# ------------------------------------------------------------------
# Persistence integration
# ------------------------------------------------------------------


class TestPersistence:
    def test_add_message_persists_to_store(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Hello")
            h = ConversationHistory(store=store, conversation_id=conv_id)
            h.add_user_message("Hello")
            h.add_assistant_message("Hi there!")
            msgs = store.load_messages(conv_id)
            assert len(msgs) == 2
            assert msgs[0] == {"role": "user", "content": "Hello"}
            assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_add_message_without_store(self):
        """Backward compat: no store means no persistence, no errors."""
        h = ConversationHistory()
        h.add_user_message("Hello")
        h.add_assistant_message("Hi")
        assert len(h.get_messages()) == 2

    def test_from_store_loads_messages(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test")
            store.add_message(conv_id, "user", "hello")
            store.add_message(conv_id, "assistant", "hi")
            store.add_message(conv_id, "user", "how are you?")
            store.add_message(conv_id, "assistant", "good!")

            h = ConversationHistory.from_store(store, conv_id)
            msgs = h.get_messages()
            assert len(msgs) == 4
            assert msgs[0]["content"] == "hello"
            assert msgs[3]["content"] == "good!"

    def test_from_store_trims_to_max_turns(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test")
            # Add 10 turns (20 messages)
            for i in range(10):
                store.add_message(conv_id, "user", f"msg {i}")
                store.add_message(conv_id, "assistant", f"reply {i}")

            h = ConversationHistory.from_store(store, conv_id, max_turns=3)
            msgs = h.get_messages()
            # max_turns=3 → 6 messages in memory
            assert len(msgs) == 6
            # Should be the last 3 turns
            assert msgs[0]["content"] == "msg 7"
            assert msgs[5]["content"] == "reply 9"
            # DB still has all 20
            assert len(store.load_messages(conv_id)) == 20

    def test_set_persistence_enables_persistence(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            h = ConversationHistory()
            # No persistence yet
            h.add_user_message("before")
            assert store.list_conversations() == []

            conv_id = store.create("Test")
            h.set_persistence(store, conv_id)
            h.add_user_message("after")
            msgs = store.load_messages(conv_id)
            assert len(msgs) == 1
            assert msgs[0]["content"] == "after"

    def test_conversation_id_property(self, tmp_path):
        h = ConversationHistory()
        assert h.conversation_id is None

        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test")
            h.set_persistence(store, conv_id)
            assert h.conversation_id == conv_id
