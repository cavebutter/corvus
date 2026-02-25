"""Tests for the ConversationStore SQLite persistence layer."""

import time

import pytest

from corvus.orchestrator.conversation_store import ConversationStore, _generate_title


# ------------------------------------------------------------------
# _generate_title helper
# ------------------------------------------------------------------


class TestGenerateTitle:
    def test_short_message(self):
        assert _generate_title("Hello") == "Hello"

    def test_exact_max_length(self):
        msg = "x" * 60
        assert _generate_title(msg) == msg

    def test_long_message_truncated_on_word_boundary(self):
        msg = "This is a fairly long message that should be truncated at a word boundary to fit within sixty chars"
        title = _generate_title(msg)
        assert len(title) <= 63  # 60 + "..."
        assert title.endswith("...")
        # The part before "..." should end at a word boundary (space was the split point)
        stem = title[:-3]
        assert stem == msg[: len(stem)]

    def test_long_single_word(self):
        msg = "x" * 100
        title = _generate_title(msg)
        assert title.endswith("...")
        assert len(title) <= 63

    def test_strips_whitespace(self):
        assert _generate_title("  hello  ") == "hello"

    def test_replaces_newlines(self):
        assert _generate_title("line1\nline2") == "line1 line2"


# ------------------------------------------------------------------
# ConversationStore.create
# ------------------------------------------------------------------


class TestCreate:
    def test_create_returns_uuid(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Hello Corvus")
            assert len(conv_id) == 36  # UUID format
            assert "-" in conv_id

    def test_create_sets_title(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Find my AT&T invoice from last month")
            conv = store.get_conversation(conv_id)
            assert conv["title"] == "Find my AT&T invoice from last month"

    def test_create_truncates_long_title(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            long_msg = "word " * 20  # 100 chars
            conv_id = store.create(long_msg)
            conv = store.get_conversation(conv_id)
            assert len(conv["title"]) <= 63  # 60 + "..."


# ------------------------------------------------------------------
# ConversationStore.add_message
# ------------------------------------------------------------------


class TestAddMessage:
    def test_add_and_load(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Hello")
            store.add_message(conv_id, "user", "Hello")
            store.add_message(conv_id, "assistant", "Hi there!")
            msgs = store.load_messages(conv_id)
            assert len(msgs) == 2
            assert msgs[0] == {"role": "user", "content": "Hello"}
            assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_updates_updated_at(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Hello")
            conv_before = store.get_conversation(conv_id)
            time.sleep(0.01)
            store.add_message(conv_id, "user", "Hello")
            conv_after = store.get_conversation(conv_id)
            assert conv_after["updated_at"] >= conv_before["updated_at"]

    def test_message_ordering_preserved(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("First")
            for i in range(5):
                store.add_message(conv_id, "user", f"msg {i}")
                store.add_message(conv_id, "assistant", f"reply {i}")
            msgs = store.load_messages(conv_id)
            assert len(msgs) == 10
            assert msgs[0]["content"] == "msg 0"
            assert msgs[9]["content"] == "reply 4"


# ------------------------------------------------------------------
# ConversationStore.load_messages
# ------------------------------------------------------------------


class TestLoadMessages:
    def test_empty_conversation(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Empty")
            msgs = store.load_messages(conv_id)
            assert msgs == []

    def test_returns_role_and_content_only(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test")
            store.add_message(conv_id, "user", "hello")
            msgs = store.load_messages(conv_id)
            assert set(msgs[0].keys()) == {"role", "content"}


# ------------------------------------------------------------------
# ConversationStore.get_most_recent
# ------------------------------------------------------------------


class TestGetMostRecent:
    def test_empty_store(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            assert store.get_most_recent() is None

    def test_returns_last_updated(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            id1 = store.create("First conversation")
            store.add_message(id1, "user", "First conversation")
            id2 = store.create("Second conversation")
            store.add_message(id2, "user", "Second conversation")
            # id2 was created/updated last
            assert store.get_most_recent() == id2
            # Now update id1 — it should become most recent
            time.sleep(0.01)
            store.add_message(id1, "user", "Update first")
            assert store.get_most_recent() == id1


# ------------------------------------------------------------------
# ConversationStore.list_conversations
# ------------------------------------------------------------------


class TestListConversations:
    def test_empty_store(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            assert store.list_conversations() == []

    def test_ordering_newest_first(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            id1 = store.create("First")
            time.sleep(0.01)
            id2 = store.create("Second")
            convs = store.list_conversations()
            assert len(convs) == 2
            assert convs[0]["id"] == id2
            assert convs[1]["id"] == id1

    def test_includes_message_count(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test")
            store.add_message(conv_id, "user", "Hello")
            store.add_message(conv_id, "assistant", "Hi")
            convs = store.list_conversations()
            assert convs[0]["message_count"] == 2

    def test_limit(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            for i in range(5):
                store.create(f"Conv {i}")
            convs = store.list_conversations(limit=3)
            assert len(convs) == 3

    def test_dict_keys(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            store.create("Test")
            conv = store.list_conversations()[0]
            assert set(conv.keys()) == {"id", "title", "created_at", "updated_at", "message_count"}


# ------------------------------------------------------------------
# ConversationStore.get_conversation
# ------------------------------------------------------------------


class TestGetConversation:
    def test_nonexistent(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            assert store.get_conversation("nonexistent-id") is None

    def test_exact_match(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Test conversation")
            conv = store.get_conversation(conv_id)
            assert conv is not None
            assert conv["id"] == conv_id
            assert conv["title"] == "Test conversation"

    def test_prefix_match(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Prefix test")
            prefix = conv_id[:8]
            conv = store.get_conversation(prefix)
            assert conv is not None
            assert conv["id"] == conv_id

    def test_prefix_no_match(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            store.create("Something")
            assert store.get_conversation("zzz") is None


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------


class TestContextManager:
    def test_enter_exit(self, tmp_path):
        with ConversationStore(tmp_path / "test.db") as store:
            conv_id = store.create("Hello")
            assert conv_id is not None
        # After exit, connection should be closed
        # We can verify by trying to create a new store at the same path
        with ConversationStore(tmp_path / "test.db") as store2:
            conv = store2.get_conversation(conv_id)
            assert conv is not None
