"""Ntfy push notification integration.

Sends notifications via HTTP POST to a Ntfy server (self-hosted or ntfy.sh).
All sends are fire-and-forget — failures are logged but never block the caller.

Reference: https://docs.ntfy.sh/publish/
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Ntfy priority levels (1-5)
PRIORITY_MIN = "1"
PRIORITY_LOW = "2"
PRIORITY_DEFAULT = "3"
PRIORITY_HIGH = "4"
PRIORITY_URGENT = "5"


async def send(
    *,
    server: str,
    topic: str,
    message: str,
    title: str | None = None,
    priority: str = PRIORITY_DEFAULT,
    tags: list[str] | None = None,
    click_url: str | None = None,
) -> bool:
    """Send a notification to a Ntfy topic.

    Args:
        server: Ntfy server URL (e.g. "https://ntfy.sh").
        topic: Topic name to publish to.
        message: Notification body text.
        title: Optional notification title.
        priority: Priority level ("1"-"5").
        tags: Optional emoji tag names (e.g. ["warning", "email"]).
        click_url: Optional URL to open when notification is tapped.

    Returns:
        True if the notification was sent successfully, False otherwise.
    """
    if not server or not topic:
        logger.debug("Ntfy not configured (no server/topic), skipping notification")
        return False

    url = f"{server.rstrip('/')}/{topic}"
    headers: dict[str, str] = {}

    if title:
        headers["Title"] = title
    if priority != PRIORITY_DEFAULT:
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url

    try:
        # Suppress httpx request logging to avoid leaking the topic (secret) in logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Ntfy notification sent: %s", title or message[:50])
        return True
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Failed to send ntfy notification: HTTP %d", exc.response.status_code
        )
        return False
    except Exception:
        logger.error("Failed to send ntfy notification: connection error")
        return False


async def notify(
    message: str,
    *,
    title: str | None = None,
    priority: str = PRIORITY_DEFAULT,
    tags: list[str] | None = None,
    click_url: str | None = None,
) -> bool:
    """Send a notification using the configured server and topic.

    Convenience wrapper around send() that reads config automatically.
    """
    from corvus.config import NTFY_SERVER, NTFY_TOPIC

    return await send(
        server=NTFY_SERVER,
        topic=NTFY_TOPIC,
        message=message,
        title=title,
        priority=priority,
        tags=tags,
        click_url=click_url,
    )


# ── Pipeline notification helpers ────────────────────────────────────


async def notify_triage_complete(
    *,
    account_email: str,
    processed: int,
    queued: int,
    auto_acted: int,
    errors: int,
) -> bool:
    """Notify after email triage completes (only if items were queued)."""
    if queued == 0 and errors == 0:
        return False

    parts = []
    if queued:
        parts.append(f"{queued} queued for review")
    if auto_acted:
        parts.append(f"{auto_acted} auto-applied")
    if errors:
        parts.append(f"{errors} error(s)")

    message = f"{account_email}: {processed} processed — {', '.join(parts)}"
    tags = ["email"]
    if errors:
        tags.append("warning")

    priority = PRIORITY_HIGH if queued > 0 else PRIORITY_DEFAULT
    return await notify(
        message,
        title="Email triage complete",
        priority=priority,
        tags=tags,
    )


async def notify_tag_pipeline_complete(
    *,
    processed: int,
    queued: int,
    auto_applied: int,
    errors: int,
) -> bool:
    """Notify after document tagging completes (only if items were queued)."""
    if queued == 0 and errors == 0:
        return False

    parts = []
    if queued:
        parts.append(f"{queued} queued for review")
    if auto_applied:
        parts.append(f"{auto_applied} auto-applied")
    if errors:
        parts.append(f"{errors} error(s)")

    message = f"{processed} documents processed — {', '.join(parts)}"
    tags = ["page_facing_up"]
    if errors:
        tags.append("warning")

    priority = PRIORITY_HIGH if queued > 0 else PRIORITY_DEFAULT
    return await notify(
        message,
        title="Document tagging complete",
        priority=priority,
        tags=tags,
    )


async def notify_digest_ready(summary: str) -> bool:
    """Notify when a daily digest has been generated."""
    return await notify(
        summary,
        title="Daily digest",
        tags=["clipboard"],
    )


async def notify_pipeline_error(
    pipeline: str,
    error: str,
) -> bool:
    """Notify on a critical pipeline error."""
    return await notify(
        f"{pipeline}: {error}",
        title="Pipeline error",
        priority=PRIORITY_HIGH,
        tags=["rotating_light"],
    )
