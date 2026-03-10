"""API route handlers for the Corvus web interface.

All endpoints require API key auth unless noted otherwise.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from corvus.web.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


# ── Status ───────────────────────────────────────────────────────────


@router.get("/status")
async def status():
    """Combined pipeline status — pending counts and 24h activity.

    Mirrors `corvus status` + `corvus email status`.
    """
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.audit.logger import AuditLog
    from corvus.config import (
        AUDIT_LOG_PATH,
        EMAIL_AUDIT_LOG_PATH,
        EMAIL_REVIEW_DB_PATH,
        QUEUE_DB_PATH,
    )
    from corvus.queue.email_review import EmailReviewQueue
    from corvus.queue.review import ReviewQueue

    since = datetime.now(UTC) - timedelta(hours=24)

    # Document pipeline
    with ReviewQueue(QUEUE_DB_PATH) as doc_queue:
        doc_pending = doc_queue.count_pending()

    doc_audit = AuditLog(AUDIT_LOG_PATH)
    doc_entries = doc_audit.read_entries(since=since)
    doc_processed = sum(1 for e in doc_entries if e.action in ("auto_applied", "queued_for_review"))
    doc_reviewed = sum(1 for e in doc_entries if e.action in ("review_approved", "review_rejected"))

    # Email pipeline
    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as email_queue:
        email_pending = email_queue.count_pending()

    email_audit = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)
    email_entries = email_audit.read_entries(since=since)
    email_auto = sum(1 for e in email_entries if e.action == "auto_applied")
    email_queued = sum(1 for e in email_entries if e.action == "queued_for_review")
    email_approved = sum(1 for e in email_entries if e.action == "review_approved")
    email_rejected = sum(1 for e in email_entries if e.action == "review_rejected")
    email_sender_list = sum(1 for e in email_entries if e.action == "sender_list_applied")

    return {
        "documents": {
            "pending_review": doc_pending,
            "processed_24h": doc_processed,
            "reviewed_24h": doc_reviewed,
        },
        "email": {
            "pending_review": email_pending,
            "auto_applied_24h": email_auto,
            "queued_24h": email_queued,
            "approved_24h": email_approved,
            "rejected_24h": email_rejected,
            "sender_list_24h": email_sender_list,
        },
    }


# ── Audit: Documents ────────────────────────────────────────────────


@router.get("/audit/documents")
async def audit_documents(
    since: str | None = Query(None, description="ISO datetime — only entries after this time"),
    limit: int = Query(50, ge=1, le=500),
):
    """Document audit log entries."""
    from corvus.audit.logger import AuditLog
    from corvus.config import AUDIT_LOG_PATH

    audit = AuditLog(AUDIT_LOG_PATH)
    since_dt = datetime.fromisoformat(since) if since else None
    entries = audit.read_entries(since=since_dt, limit=limit)
    return [e.model_dump(mode="json") for e in entries]


# ── Audit: Email ─────────────────────────────────────────────────────


@router.get("/audit/email")
async def audit_email(
    since: str | None = Query(None, description="ISO datetime — only entries after this time"),
    limit: int = Query(50, ge=1, le=500),
):
    """Email audit log entries."""
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.config import EMAIL_AUDIT_LOG_PATH

    audit = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)
    since_dt = datetime.fromisoformat(since) if since else None
    entries = audit.read_entries(since=since_dt, limit=limit)
    return [e.model_dump(mode="json") for e in entries]


# ── Audit: Watchdog ──────────────────────────────────────────────────


@router.get("/audit/watchdog")
async def audit_watchdog(
    since: str | None = Query(None, description="ISO datetime — only entries after this time"),
    limit: int = Query(50, ge=1, le=500),
):
    """Watchdog audit log entries."""
    from corvus.config import WATCHDOG_AUDIT_LOG_PATH
    from corvus.watchdog.audit import WatchdogAuditLog

    audit = WatchdogAuditLog(WATCHDOG_AUDIT_LOG_PATH)
    since_dt = datetime.fromisoformat(since) if since else None
    entries = audit.read_entries(since=since_dt, limit=limit)
    return [e.model_dump(mode="json") for e in entries]


# ── Review: Request body ─────────────────────────────────────────────


class ReviewAction(BaseModel):
    notes: str | None = None


# ── Review: Documents ────────────────────────────────────────────────


@router.get("/review/documents")
async def review_documents_list():
    """List pending document review items."""
    from corvus.config import QUEUE_DB_PATH
    from corvus.queue.review import ReviewQueue

    with ReviewQueue(QUEUE_DB_PATH) as q:
        items = q.list_pending()
    return [item.model_dump(mode="json") for item in items]


@router.post("/review/documents/{item_id}/approve")
async def review_documents_approve(item_id: str, body: ReviewAction | None = None):
    """Approve a document review item and apply changes to Paperless."""
    import httpx

    from corvus.audit.logger import AuditLog
    from corvus.config import (
        AUDIT_LOG_PATH,
        OLLAMA_BASE_URL,
        PAPERLESS_API_TOKEN,
        PAPERLESS_BASE_URL,
        QUEUE_DB_PATH,
    )
    from corvus.integrations.paperless import PaperlessClient
    from corvus.queue.review import ReviewQueue
    from corvus.router.tagging import apply_approved_update

    notes = body.notes if body else None

    with ReviewQueue(QUEUE_DB_PATH) as q:
        item = q.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status.value != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Item already {item.status.value}",
            )

        async with httpx.AsyncClient() as http:
            paperless = PaperlessClient(
                base_url=PAPERLESS_BASE_URL,
                token=PAPERLESS_API_TOKEN,
                client=http,
            )
            try:
                result = await apply_approved_update(item, paperless=paperless)
            except Exception:
                logger.exception("Failed to apply document update %s", item_id)
                raise HTTPException(
                    status_code=502,
                    detail="Failed to apply update to Paperless",
                )

        q.approve(item_id, notes=notes or "Approved via web")

        audit = AuditLog(AUDIT_LOG_PATH)
        audit.log_review_approved(item.task, result.proposed_update)

    return {"status": "approved", "item_id": item_id}


@router.post("/review/documents/{item_id}/reject")
async def review_documents_reject(item_id: str, body: ReviewAction | None = None):
    """Reject a document review item."""
    from corvus.audit.logger import AuditLog
    from corvus.config import AUDIT_LOG_PATH, QUEUE_DB_PATH
    from corvus.queue.review import ReviewQueue

    notes = body.notes if body else None

    with ReviewQueue(QUEUE_DB_PATH) as q:
        item = q.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status.value != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Item already {item.status.value}",
            )
        q.reject(item_id, notes=notes or "Rejected via web")

    audit = AuditLog(AUDIT_LOG_PATH)
    audit.log_review_rejected(item.task, item.proposed_update)

    return {"status": "rejected", "item_id": item_id}


# ── Review: Email ────────────────────────────────────────────────────


@router.get("/review/email")
async def review_email_list():
    """List pending email review items."""
    from corvus.config import EMAIL_REVIEW_DB_PATH
    from corvus.queue.email_review import EmailReviewQueue

    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as q:
        items = q.list_pending()
    return [item.model_dump(mode="json") for item in items]


@router.post("/review/email/{item_id}/approve")
async def review_email_approve(item_id: str, body: ReviewAction | None = None):
    """Approve an email review item and execute the IMAP action."""
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.config import (
        EMAIL_AUDIT_LOG_PATH,
        EMAIL_REVIEW_DB_PATH,
        load_email_accounts,
    )
    from corvus.integrations.imap import ImapClient
    from corvus.queue.email_review import EmailReviewQueue
    from corvus.router.email import execute_email_action
    from corvus.schemas.email import EmailAccountConfig

    notes = body.notes if body else None

    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as q:
        item = q.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status.value != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Item already {item.status.value}",
            )

        # Find the account config for this email
        accounts = load_email_accounts()
        account_config = None
        for a in accounts:
            if a["email"] == item.task.account_email:
                account_config = EmailAccountConfig(**a)
                break

        if account_config is None:
            raise HTTPException(
                status_code=400,
                detail=f"No account config for {item.task.account_email}",
            )

        try:
            async with ImapClient(account_config) as imap:
                applied = await execute_email_action(item.task, imap=imap)
        except Exception:
            logger.exception("Failed to execute email action for %s", item_id)
            raise HTTPException(
                status_code=502,
                detail="Failed to execute IMAP action",
            )

        if applied:
            q.approve(item_id, notes=notes or "Approved via web")
        else:
            q.approve(
                item_id,
                notes=(notes or "Approved via web")
                + " (stale — message no longer on server)",
            )

    audit = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)
    audit.log_review_approved(item.task)

    return {"status": "approved", "item_id": item_id, "applied": applied}


@router.post("/review/email/{item_id}/reject")
async def review_email_reject(item_id: str, body: ReviewAction | None = None):
    """Reject an email review item."""
    from corvus.audit.email_logger import EmailAuditLog
    from corvus.config import EMAIL_AUDIT_LOG_PATH, EMAIL_REVIEW_DB_PATH
    from corvus.queue.email_review import EmailReviewQueue

    notes = body.notes if body else None

    with EmailReviewQueue(EMAIL_REVIEW_DB_PATH) as q:
        item = q.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status.value != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Item already {item.status.value}",
            )
        q.reject(item_id, notes=notes or "Rejected via web")

    audit = EmailAuditLog(EMAIL_AUDIT_LOG_PATH)
    audit.log_review_rejected(item.task)

    return {"status": "rejected", "item_id": item_id}
