"""Tests for sender list management (corvus/sender_lists.py).

Covers: load/save, lookup, priority resolution, add/remove,
case-insensitive matching, rationalize, missing file handling,
and build_task_from_sender_match.
"""

import json
from datetime import datetime, timezone

import pytest

from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailActionType,
    EmailCategory,
    EmailMessage,
)
from corvus.schemas.sender_lists import SenderListConfig, SenderListsFile, SenderMatch
from corvus.sender_lists import SenderListManager


# --- Fixtures ---


def _sample_data() -> dict:
    """Return a sample sender_lists.json structure."""
    return {
        "lists": {
            "white": {
                "description": "Known humans.",
                "action": "keep",
                "addresses": ["alice@example.com", "Bob@Work.com"],
            },
            "vendor": {
                "description": "Vendors.",
                "action": "move",
                "folder_key": "approved_ads",
                "cleanup_days": 14,
                "addresses": ["deals@amazon.com"],
            },
            "headhunter": {
                "description": "Headhunters.",
                "action": "move",
                "folder_key": "headhunt",
                "cleanup_days": 14,
                "addresses": ["recruiter@staffing.com"],
            },
            "black": {
                "description": "Blacklisted.",
                "action": "delete",
                "addresses": ["spam@scammer.com"],
            },
        },
        "priority": ["white", "black", "vendor", "headhunter"],
    }


@pytest.fixture
def sender_lists_path(tmp_path):
    """Write sample sender lists to a temp file and return the path."""
    path = tmp_path / "sender_lists.json"
    path.write_text(json.dumps(_sample_data(), indent=2))
    return path


@pytest.fixture
def mgr(sender_lists_path):
    """Load a SenderListManager from the sample data."""
    return SenderListManager.load(sender_lists_path)


def _make_email(
    uid: str = "100",
    from_addr: str = "sender@example.com",
    subject: str = "Test",
) -> EmailMessage:
    return EmailMessage(
        uid=uid,
        account_email="test@test.com",
        from_address=from_addr,
        subject=subject,
        date=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        body_text="Hello.",
    )


# --- Load / Save ---


class TestLoadSave:
    def test_load_from_file(self, mgr):
        assert len(mgr.data.lists) == 4
        assert "white" in mgr.data.lists
        assert "black" in mgr.data.lists

    def test_load_missing_file(self, tmp_path):
        mgr = SenderListManager.load(tmp_path / "nonexistent.json")
        assert len(mgr.data.lists) == 0
        assert mgr.lookup("anyone@example.com") is None

    def test_save_and_reload(self, sender_lists_path):
        mgr = SenderListManager.load(sender_lists_path)
        mgr.add("black", "new-spam@evil.com")

        # Reload from the same file
        mgr2 = SenderListManager.load(sender_lists_path)
        match = mgr2.lookup("new-spam@evil.com")
        assert match is not None
        assert match.list_name == "black"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deep" / "sender_lists.json"
        mgr = SenderListManager.load(path)
        mgr.add("white", "test@example.com")
        # File should now exist
        assert path.exists()
        data = json.loads(path.read_text())
        assert "test@example.com" in data["lists"]["white"]["addresses"]


# --- Lookup ---


class TestLookup:
    def test_exact_match(self, mgr):
        match = mgr.lookup("alice@example.com")
        assert match is not None
        assert match.list_name == "white"
        assert match.action == "keep"

    def test_case_insensitive(self, mgr):
        match = mgr.lookup("ALICE@EXAMPLE.COM")
        assert match is not None
        assert match.list_name == "white"

    def test_mixed_case_stored(self, mgr):
        """Bob@Work.com was stored with mixed case, should still match."""
        match = mgr.lookup("bob@work.com")
        assert match is not None
        assert match.list_name == "white"

    def test_no_match(self, mgr):
        assert mgr.lookup("unknown@example.com") is None

    def test_blacklist_match(self, mgr):
        match = mgr.lookup("spam@scammer.com")
        assert match is not None
        assert match.list_name == "black"
        assert match.action == "delete"

    def test_vendor_match(self, mgr):
        match = mgr.lookup("deals@amazon.com")
        assert match is not None
        assert match.list_name == "vendor"
        assert match.action == "move"
        assert match.folder_key == "approved_ads"
        assert match.cleanup_days == 14


# --- Priority ---


class TestPriority:
    def test_priority_resolution(self, tmp_path):
        """If same address in white and black, white wins (first in priority)."""
        data = {
            "lists": {
                "white": {"action": "keep", "addresses": ["conflict@example.com"]},
                "black": {"action": "delete", "addresses": ["conflict@example.com"]},
            },
            "priority": ["white", "black"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))

        mgr = SenderListManager.load(path)
        match = mgr.lookup("conflict@example.com")
        assert match is not None
        assert match.list_name == "white"

    def test_reversed_priority(self, tmp_path):
        """If black is higher priority, it should win."""
        data = {
            "lists": {
                "white": {"action": "keep", "addresses": ["conflict@example.com"]},
                "black": {"action": "delete", "addresses": ["conflict@example.com"]},
            },
            "priority": ["black", "white"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))

        mgr = SenderListManager.load(path)
        match = mgr.lookup("conflict@example.com")
        assert match is not None
        assert match.list_name == "black"

    def test_lists_not_in_priority_added_at_end(self, tmp_path):
        """Lists not in the priority array still get indexed."""
        data = {
            "lists": {
                "custom": {"action": "move", "folder_key": "custom_folder", "addresses": ["x@y.com"]},
            },
            "priority": [],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))

        mgr = SenderListManager.load(path)
        match = mgr.lookup("x@y.com")
        assert match is not None
        assert match.list_name == "custom"


# --- Add / Remove ---


class TestAddRemove:
    def test_add_to_existing_list(self, mgr):
        assert mgr.add("black", "new-spam@evil.com") is True
        match = mgr.lookup("new-spam@evil.com")
        assert match is not None
        assert match.list_name == "black"

    def test_add_duplicate_returns_false(self, mgr):
        assert mgr.add("white", "alice@example.com") is False

    def test_add_duplicate_case_insensitive(self, mgr):
        assert mgr.add("white", "ALICE@EXAMPLE.COM") is False

    def test_add_to_new_list(self, mgr):
        assert mgr.add("custom", "user@custom.com") is True
        match = mgr.lookup("user@custom.com")
        assert match is not None
        assert match.list_name == "custom"
        # New list should be in priority
        assert "custom" in mgr.data.priority

    def test_remove_existing(self, mgr):
        assert mgr.remove("white", "alice@example.com") is True
        assert mgr.lookup("alice@example.com") is None

    def test_remove_case_insensitive(self, mgr):
        assert mgr.remove("white", "ALICE@EXAMPLE.COM") is True
        assert mgr.lookup("alice@example.com") is None

    def test_remove_nonexistent_address(self, mgr):
        assert mgr.remove("white", "nobody@example.com") is False

    def test_remove_nonexistent_list(self, mgr):
        assert mgr.remove("nonexistent", "alice@example.com") is False


# --- Rationalize ---


class TestRationalize:
    def test_dedup_within_list(self, tmp_path):
        data = {
            "lists": {
                "black": {
                    "action": "delete",
                    "addresses": ["dupe@spam.com", "DUPE@SPAM.COM", "other@spam.com"],
                },
            },
            "priority": ["black"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))

        mgr = SenderListManager.load(path)
        actions = mgr.rationalize()

        assert len(mgr.data.lists["black"].addresses) == 2
        assert any("duplicate" in a.lower() for a in actions)

    def test_cross_list_conflict_resolution(self, tmp_path):
        data = {
            "lists": {
                "white": {"action": "keep", "addresses": ["shared@example.com"]},
                "black": {"action": "delete", "addresses": ["shared@example.com"]},
            },
            "priority": ["white", "black"],
        }
        path = tmp_path / "sender_lists.json"
        path.write_text(json.dumps(data))

        mgr = SenderListManager.load(path)
        actions = mgr.rationalize()

        assert "shared@example.com" in mgr.data.lists["white"].addresses
        assert "shared@example.com" not in mgr.data.lists["black"].addresses
        assert any("higher-priority" in a for a in actions)

    def test_no_changes_returns_empty(self, mgr):
        actions = mgr.rationalize()
        assert actions == []


# --- build_task_from_sender_match ---


class TestBuildTask:
    def test_build_delete_task(self, mgr):
        email = _make_email(from_addr="spam@scammer.com")
        match = mgr.lookup("spam@scammer.com")
        folders = {"inbox": "INBOX", "approved_ads": "Corvus/Ads"}

        task = mgr.build_task_from_sender_match(email, match, folders)

        assert task.uid == "100"
        assert task.sender_list == "black"
        assert task.proposed_action.action_type == EmailActionType.DELETE
        assert task.overall_confidence == 1.0
        assert task.gate_action == GateAction.AUTO_EXECUTE
        assert task.classification.category == EmailCategory.SPAM

    def test_build_move_task(self, mgr):
        email = _make_email(from_addr="deals@amazon.com")
        match = mgr.lookup("deals@amazon.com")
        folders = {
            "inbox": "INBOX",
            "approved_ads": "Corvus/Ads",
            "headhunt": "Corvus/Headhunt",
        }

        task = mgr.build_task_from_sender_match(email, match, folders)

        assert task.sender_list == "vendor"
        assert task.proposed_action.action_type == EmailActionType.MOVE
        assert task.proposed_action.target_folder == "Corvus/Ads"
        assert task.classification.category == EmailCategory.NEWSLETTER

    def test_build_keep_task(self, mgr):
        email = _make_email(from_addr="alice@example.com")
        match = mgr.lookup("alice@example.com")
        folders = {"inbox": "INBOX"}

        task = mgr.build_task_from_sender_match(email, match, folders)

        assert task.sender_list == "white"
        assert task.proposed_action.action_type == EmailActionType.KEEP
        assert task.classification.category == EmailCategory.PERSONAL

    def test_build_headhunter_task(self, mgr):
        email = _make_email(from_addr="recruiter@staffing.com")
        match = mgr.lookup("recruiter@staffing.com")
        folders = {"inbox": "INBOX", "headhunt": "Corvus/Headhunt"}

        task = mgr.build_task_from_sender_match(email, match, folders)

        assert task.sender_list == "headhunter"
        assert task.proposed_action.action_type == EmailActionType.MOVE
        assert task.proposed_action.target_folder == "Corvus/Headhunt"
        assert task.classification.category == EmailCategory.JOB_ALERT


# --- Schema validation ---


class TestSchemas:
    def test_sender_list_config(self):
        cfg = SenderListConfig(
            description="Test",
            action="move",
            folder_key="test_folder",
            cleanup_days=7,
            addresses=["a@b.com"],
        )
        assert cfg.action == "move"
        assert cfg.cleanup_days == 7

    def test_sender_lists_file(self):
        data = SenderListsFile.model_validate(_sample_data())
        assert len(data.lists) == 4
        assert data.priority == ["white", "black", "vendor", "headhunter"]

    def test_sender_match(self):
        match = SenderMatch(
            list_name="black",
            address="spam@scammer.com",
            action="delete",
        )
        assert match.folder_key is None
        assert match.cleanup_days is None


# --- Create / Delete ---


class TestCreateDelete:
    def test_create_new_list(self, mgr):
        mgr.create("finance", action="move", folder_key="receipts", description="Finance emails")
        assert "finance" in mgr.data.lists
        lst = mgr.data.lists["finance"]
        assert lst.action == "move"
        assert lst.folder_key == "receipts"
        assert lst.addresses == []
        assert lst.description == "Finance emails"

    def test_create_persists_to_disk(self, sender_lists_path):
        mgr = SenderListManager.load(sender_lists_path)
        mgr.create("finance", action="keep")

        mgr2 = SenderListManager.load(sender_lists_path)
        assert "finance" in mgr2.data.lists

    def test_create_duplicate_raises(self, mgr):
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("white", action="keep")

    def test_create_with_cleanup_days(self, mgr):
        mgr.create("promo", action="move", folder_key="promos", cleanup_days=30)
        assert mgr.data.lists["promo"].cleanup_days == 30

    def test_create_adds_to_priority(self, mgr):
        mgr.create("finance", action="keep")
        assert "finance" in mgr.data.priority
        # Should be appended at end
        assert mgr.data.priority[-1] == "finance"

    def test_delete_existing_list(self, mgr):
        mgr.delete("vendor")
        assert "vendor" not in mgr.data.lists
        assert "vendor" not in mgr.data.priority

    def test_delete_removes_addresses_from_index(self, mgr):
        # Verify address exists first
        assert mgr.lookup("deals@amazon.com") is not None
        mgr.delete("vendor")
        assert mgr.lookup("deals@amazon.com") is None

    def test_delete_nonexistent_raises(self, mgr):
        with pytest.raises(KeyError, match="does not exist"):
            mgr.delete("nonexistent")

    def test_delete_persists_to_disk(self, sender_lists_path):
        mgr = SenderListManager.load(sender_lists_path)
        mgr.delete("vendor")

        mgr2 = SenderListManager.load(sender_lists_path)
        assert "vendor" not in mgr2.data.lists

    def test_custom_list_falls_back_to_other(self, mgr):
        """A custom list name not in category_map should get EmailCategory.OTHER."""
        mgr.create("finance", action="move", folder_key="receipts")
        mgr.add("finance", "billing@example.com")

        email = _make_email(from_addr="billing@example.com")
        match = mgr.lookup("billing@example.com")
        folders = {"inbox": "INBOX", "receipts": "Corvus/Receipts"}

        task = mgr.build_task_from_sender_match(email, match, folders)
        assert task.classification.category == EmailCategory.OTHER
        assert task.sender_list == "finance"
