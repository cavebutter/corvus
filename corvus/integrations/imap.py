"""Async IMAP client wrapping imap-tools.

imap-tools is synchronous; all public methods use asyncio.to_thread()
for non-blocking operation — the same pattern used for sounddevice in the
voice pipeline.

Usage::

    async with ImapClient(account_config) as imap:
        emails = await imap.fetch_envelopes("INBOX", limit=50)
        await imap.move(["uid1", "uid2"], "Corvus/Processed")
"""

import asyncio
import logging
from email.utils import parseaddr

from imap_tools import AND, MailBox, MailboxLoginError, MailMessage

from corvus.schemas.email import EmailAccountConfig, EmailEnvelope, EmailMessage

logger = logging.getLogger(__name__)


def _parse_envelope(msg: MailMessage, account_email: str) -> EmailEnvelope:
    """Convert an imap-tools MailMessage to an EmailEnvelope (headers only)."""
    from_name, from_addr = parseaddr(msg.from_)
    return EmailEnvelope(
        uid=msg.uid,
        account_email=account_email,
        from_address=from_addr or msg.from_,
        from_name=from_name,
        to=[addr for addr in msg.to],
        subject=msg.subject or "(no subject)",
        date=msg.date,
        flags=list(msg.flags),
        size_bytes=msg.size,
    )


def _parse_message(msg: MailMessage, account_email: str) -> EmailMessage:
    """Convert an imap-tools MailMessage to a full EmailMessage."""
    from_name, from_addr = parseaddr(msg.from_)
    attachments = list(msg.attachments)
    return EmailMessage(
        uid=msg.uid,
        account_email=account_email,
        from_address=from_addr or msg.from_,
        from_name=from_name,
        to=[addr for addr in msg.to],
        subject=msg.subject or "(no subject)",
        date=msg.date,
        flags=list(msg.flags),
        size_bytes=msg.size,
        body_text=msg.text or "",
        body_html=msg.html or "",
        has_attachments=len(attachments) > 0,
        attachment_names=[a.filename for a in attachments if a.filename],
    )


class ImapClient:
    """Async IMAP client wrapping imap-tools.

    Usage::

        async with ImapClient(account_config) as imap:
            emails = await imap.fetch_envelopes("INBOX", limit=50)
            await imap.move(["uid1", "uid2"], "Corvus/Processed")
    """

    def __init__(self, config: EmailAccountConfig) -> None:
        self._config = config
        self._mailbox: MailBox | None = None

    async def __aenter__(self) -> "ImapClient":
        self._mailbox = await asyncio.to_thread(self._connect)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._mailbox:
            await asyncio.to_thread(self._disconnect)
            self._mailbox = None

    def _connect(self) -> MailBox:
        """Connect and login (sync, called via to_thread)."""
        if self._config.ssl:
            mb = MailBox(self._config.server, port=self._config.port)
        else:
            from imap_tools import MailBoxUnencrypted

            mb = MailBoxUnencrypted(self._config.server, port=self._config.port)

        try:
            mb.login(self._config.email, self._config.password)
        except MailboxLoginError:
            logger.error("IMAP login failed for %s", self._config.email)
            raise

        logger.info("Connected to %s as %s", self._config.server, self._config.email)
        return mb

    def _disconnect(self) -> None:
        """Logout and close (sync, called via to_thread)."""
        if self._mailbox:
            try:
                self._mailbox.logout()
            except Exception:
                logger.debug("Error during IMAP logout", exc_info=True)

    @property
    def mailbox(self) -> MailBox:
        if self._mailbox is None:
            raise RuntimeError("ImapClient is not connected. Use 'async with' context.")
        return self._mailbox

    # --- Fetch ---

    async def fetch_envelopes(
        self, folder: str, *, limit: int = 0
    ) -> list[EmailEnvelope]:
        """Fetch email envelopes (headers only) from a folder.

        Args:
            folder: IMAP folder name (e.g. "INBOX").
            limit: Maximum number of emails to fetch (0 = all unseen).

        Returns:
            List of EmailEnvelope objects, newest first.
        """

        def _fetch() -> list[EmailEnvelope]:
            self.mailbox.folder.set(folder)
            criteria = AND(seen=False)
            msgs = self.mailbox.fetch(
                criteria,
                headers_only=True,
                mark_seen=False,
                reverse=True,
                limit=limit if limit > 0 else None,
            )
            return [_parse_envelope(m, self._config.email) for m in msgs]

        return await asyncio.to_thread(_fetch)

    async def fetch_message(self, folder: str, uid: str) -> EmailMessage:
        """Fetch a single full email message by UID.

        Args:
            folder: IMAP folder name.
            uid: Message UID.

        Returns:
            EmailMessage with full body content.

        Raises:
            ValueError: If message with given UID is not found.
        """

        def _fetch() -> EmailMessage:
            self.mailbox.folder.set(folder)
            msgs = list(
                self.mailbox.fetch(AND(uid=uid), mark_seen=False, limit=1)
            )
            if not msgs:
                raise ValueError(f"Email UID {uid} not found in {folder}")
            return _parse_message(msgs[0], self._config.email)

        return await asyncio.to_thread(_fetch)

    async def fetch_messages(
        self, folder: str, *, limit: int = 0
    ) -> list[EmailMessage]:
        """Fetch full email messages from a folder.

        Args:
            folder: IMAP folder name.
            limit: Maximum number of emails to fetch (0 = all unseen).

        Returns:
            List of EmailMessage objects, newest first.
        """

        def _fetch() -> list[EmailMessage]:
            self.mailbox.folder.set(folder)
            criteria = AND(seen=False)
            msgs = self.mailbox.fetch(
                criteria,
                mark_seen=False,
                reverse=True,
                limit=limit if limit > 0 else None,
            )
            return [_parse_message(m, self._config.email) for m in msgs]

        return await asyncio.to_thread(_fetch)

    # --- Actions ---

    async def move(self, uids: list[str], target_folder: str) -> None:
        """Move emails to a target folder.

        Uses Gmail-specific COPY+DELETE for Gmail accounts (Gmail doesn't
        support standard MOVE).
        """
        if not uids:
            return

        if self._config.is_gmail:
            await self._gmail_move(uids, target_folder)
        else:
            await asyncio.to_thread(self._move_sync, uids, target_folder)

    def _move_sync(self, uids: list[str], target_folder: str) -> None:
        """Standard IMAP move (sync)."""
        self.mailbox.move(uids, target_folder)
        logger.info("Moved %d email(s) to %s", len(uids), target_folder)

    async def _gmail_move(self, uids: list[str], target_folder: str) -> None:
        """Gmail-specific move: COPY to target, then DELETE from source.

        Gmail IMAP doesn't support the MOVE command reliably. The standard
        workaround is COPY + set \\Deleted + EXPUNGE.
        """

        def _do() -> None:
            self.mailbox.copy(uids, target_folder)
            self.mailbox.delete(uids)
            logger.info("Gmail-moved %d email(s) to %s", len(uids), target_folder)

        await asyncio.to_thread(_do)

    async def delete(self, uids: list[str]) -> None:
        """Delete emails (move to trash / expunge)."""
        if not uids:
            return

        def _do() -> None:
            self.mailbox.delete(uids)
            logger.info("Deleted %d email(s)", len(uids))

        await asyncio.to_thread(_do)

    async def flag(self, uids: list[str], flag: str, *, value: bool = True) -> None:
        """Set or unset a flag on emails."""
        if not uids:
            return

        def _do() -> None:
            self.mailbox.flag(uids, {flag}, value)
            logger.info(
                "%s flag %s on %d email(s)",
                "Set" if value else "Cleared",
                flag,
                len(uids),
            )

        await asyncio.to_thread(_do)

    async def mark_read(self, uids: list[str]) -> None:
        """Mark emails as read (set \\Seen flag)."""
        if not uids:
            return
        await self.flag(uids, "\\Seen", value=True)

    # --- Folder management ---

    async def ensure_folders(self, folders: list[str]) -> None:
        """Create folders if they don't exist."""
        existing = await self.list_folders()
        existing_set = set(existing)

        for folder in folders:
            if folder not in existing_set:
                await asyncio.to_thread(self.mailbox.folder.create, folder)
                logger.info("Created IMAP folder: %s", folder)

    async def list_folders(self) -> list[str]:
        """List all IMAP folders."""

        def _list() -> list[str]:
            return [f.name for f in self.mailbox.folder.list()]

        return await asyncio.to_thread(_list)
