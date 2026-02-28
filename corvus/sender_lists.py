"""Sender list manager for deterministic email handling.

Loads sender lists from a JSON file, provides O(1) address lookup,
and supports add/remove/rationalize operations with atomic saves.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailMessage,
    EmailTriageTask,
)
from corvus.schemas.sender_lists import SenderListConfig, SenderListsFile, SenderMatch

logger = logging.getLogger(__name__)


class SenderListManager:
    """Manages sender lists with O(1) address lookup.

    Usage::

        mgr = SenderListManager.load("data/sender_lists.json")
        match = mgr.lookup("spam@scammer.com")
        if match:
            task = mgr.build_task(email_msg, match, folders)
    """

    def __init__(self, data: SenderListsFile, path: Path) -> None:
        self._data = data
        self._path = path
        self._index: dict[str, SenderMatch] = {}
        self._rebuild_index()

    @classmethod
    def load(cls, path: str | Path) -> "SenderListManager":
        """Load sender lists from a JSON file.

        If the file does not exist, returns a manager with empty lists.
        """
        path = Path(path)
        if not path.exists():
            logger.info("Sender lists file not found at %s, using empty lists", path)
            return cls(SenderListsFile(), path)

        raw = json.loads(path.read_text())
        data = SenderListsFile.model_validate(raw)
        logger.info(
            "Loaded sender lists from %s: %d list(s), %d total address(es)",
            path,
            len(data.lists),
            sum(len(lst.addresses) for lst in data.lists.values()),
        )
        return cls(data, path)

    def _rebuild_index(self) -> None:
        """Build the reverse lookup dict, respecting priority order."""
        self._index.clear()
        # Process in reverse priority order so higher-priority lists overwrite
        ordered = list(self._data.priority)
        # Include any lists not in the priority array (append at end)
        for name in self._data.lists:
            if name not in ordered:
                ordered.append(name)
        # Reverse so that first-in-priority wins
        for list_name in reversed(ordered):
            lst = self._data.lists.get(list_name)
            if lst is None:
                continue
            for addr in lst.addresses:
                normalized = addr.lower()
                self._index[normalized] = SenderMatch(
                    list_name=list_name,
                    address=normalized,
                    action=lst.action,
                    folder_key=lst.folder_key,
                    cleanup_days=lst.cleanup_days,
                )

    def lookup(self, address: str) -> SenderMatch | None:
        """Look up an address in the sender lists. O(1)."""
        return self._index.get(address.lower())

    def add(self, list_name: str, address: str) -> bool:
        """Add an address to a list. Returns True if added, False if already present.

        Creates the list if it doesn't exist (with action="keep" default).
        Saves to disk after modification.
        """
        normalized = address.lower()

        if list_name not in self._data.lists:
            self._data.lists[list_name] = SenderListConfig(
                action="keep", addresses=[]
            )
            if list_name not in self._data.priority:
                self._data.priority.append(list_name)

        lst = self._data.lists[list_name]
        existing = {a.lower() for a in lst.addresses}
        if normalized in existing:
            return False

        lst.addresses.append(normalized)
        self._rebuild_index()
        self.save()
        return True

    def remove(self, list_name: str, address: str) -> bool:
        """Remove an address from a list. Returns True if removed, False if not found."""
        if list_name not in self._data.lists:
            return False

        normalized = address.lower()
        lst = self._data.lists[list_name]
        original_len = len(lst.addresses)
        lst.addresses = [a for a in lst.addresses if a.lower() != normalized]

        if len(lst.addresses) == original_len:
            return False

        self._rebuild_index()
        self.save()
        return True

    def save(self) -> None:
        """Atomic write: temp file + rename to prevent corruption."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = self._data.model_dump(mode="json")
        content = json.dumps(data, indent=2) + "\n"

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            os.write(fd, content.encode())
            os.close(fd)
            os.replace(tmp_path, str(self._path))
        except BaseException:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def rationalize(self) -> list[str]:
        """Dedup within lists and resolve cross-list conflicts by priority.

        Returns a list of human-readable actions taken.
        """
        actions: list[str] = []

        # 1. Dedup within each list
        for name, lst in self._data.lists.items():
            seen: set[str] = set()
            deduped: list[str] = []
            for addr in lst.addresses:
                normalized = addr.lower()
                if normalized in seen:
                    actions.append(f"Removed duplicate '{normalized}' from '{name}'")
                else:
                    seen.add(normalized)
                    deduped.append(normalized)
            lst.addresses = deduped

        # 2. Resolve cross-list conflicts by priority
        # Build priority order
        ordered = list(self._data.priority)
        for name in self._data.lists:
            if name not in ordered:
                ordered.append(name)

        seen_global: dict[str, str] = {}  # address -> winning list name
        for list_name in ordered:
            lst = self._data.lists.get(list_name)
            if lst is None:
                continue
            to_remove: list[str] = []
            for addr in lst.addresses:
                if addr in seen_global:
                    winner = seen_global[addr]
                    actions.append(
                        f"Removed '{addr}' from '{list_name}' "
                        f"(already in higher-priority list '{winner}')"
                    )
                    to_remove.append(addr)
                else:
                    seen_global[addr] = list_name
            if to_remove:
                lst.addresses = [a for a in lst.addresses if a not in to_remove]

        self._rebuild_index()
        if actions:
            self.save()
        return actions

    @property
    def data(self) -> SenderListsFile:
        """Access the underlying data model."""
        return self._data

    def build_task_from_sender_match(
        self,
        email: EmailMessage,
        match: SenderMatch,
        folders: dict[str, str],
    ) -> EmailTriageTask:
        """Build an EmailTriageTask from a sender list match, without LLM.

        Args:
            email: The email message.
            match: The sender list match result.
            folders: Account folder mapping (logical name -> IMAP path).

        Returns:
            An EmailTriageTask with deterministic classification and action.
        """
        # Determine action type and target folder
        if match.action == "delete":
            action_type = EmailActionType.DELETE
            target_folder = None
        elif match.action == "move":
            action_type = EmailActionType.MOVE
            target_folder = folders.get(match.folder_key or "", match.folder_key)
        else:  # "keep"
            action_type = EmailActionType.KEEP
            target_folder = None

        # Map list name to a reasonable category
        category_map = {
            "black": EmailCategory.SPAM,
            "vendor": EmailCategory.NEWSLETTER,
            "headhunter": EmailCategory.JOB_ALERT,
            "white": EmailCategory.PERSONAL,
        }
        category = category_map.get(match.list_name, EmailCategory.OTHER)

        return EmailTriageTask(
            uid=email.uid,
            account_email=email.account_email,
            subject=email.subject,
            from_address=email.from_address,
            sender_list=match.list_name,
            classification=EmailClassification(
                category=category,
                confidence=1.0,
                reasoning=f"Sender '{match.address}' is on the '{match.list_name}' list",
                suggested_action=match.action,
                is_automated=True,
                summary="",
            ),
            proposed_action=EmailAction(
                action_type=action_type,
                target_folder=target_folder,
            ),
            overall_confidence=1.0,
            gate_action=GateAction.AUTO_EXECUTE,
        )
