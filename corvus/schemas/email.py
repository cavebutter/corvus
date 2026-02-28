"""Schemas for the email triage pipeline (Phase 3).

Covers the full lifecycle:
  IMAP fetch -> LLM classification -> confidence gate -> review queue / auto-execute -> audit log
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from corvus.schemas.document_tagging import GateAction, ReviewStatus

# --- Config ---


class EmailAccountConfig(BaseModel):
    """Configuration for a single email account."""

    name: str
    server: str
    email: str
    password: str
    folders: dict[str, str]  # logical name -> IMAP folder path
    is_gmail: bool = False
    port: int = 993
    ssl: bool = True


# --- Email data ---


class EmailEnvelope(BaseModel):
    """Lightweight email representation (headers only)."""

    uid: str
    account_email: str
    from_address: str
    from_name: str = ""
    to: list[str] = Field(default_factory=list)
    subject: str
    date: datetime
    flags: list[str] = Field(default_factory=list)
    size_bytes: int = 0


class EmailMessage(EmailEnvelope):
    """Full email with body content."""

    body_text: str = ""
    body_html: str = ""
    has_attachments: bool = False
    attachment_names: list[str] = Field(default_factory=list)


# --- Classification ---


class EmailCategory(StrEnum):
    """Email category as determined by LLM classification."""

    SPAM = "spam"
    NEWSLETTER = "newsletter"
    JOB_ALERT = "job_alert"
    RECEIPT = "receipt"
    INVOICE = "invoice"
    PACKAGE_NOTICE = "package_notice"
    ACTION_REQUIRED = "action_required"
    PERSONAL = "personal"
    IMPORTANT = "important"
    OTHER = "other"


class EmailClassification(BaseModel):
    """LLM executor output — email classification."""

    category: EmailCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggested_action: str  # "delete", "move_to_receipts", "flag", "keep"
    is_automated: bool = False  # True if from a bot/system/noreply
    summary: str = ""  # 1-2 sentence summary of the email


# --- Actions ---


class EmailActionType(StrEnum):
    """Concrete IMAP action type."""

    MOVE = "move"
    DELETE = "delete"
    FLAG = "flag"
    MARK_READ = "mark_read"
    KEEP = "keep"  # no action


class EmailAction(BaseModel):
    """Concrete IMAP action to perform on an email."""

    action_type: EmailActionType
    target_folder: str | None = None  # for MOVE
    flag_name: str | None = None  # for FLAG


# --- Tasks ---


class EmailTriageTask(BaseModel):
    """Workflow state for a single email triage decision."""

    task_type: Literal["email_triage"] = "email_triage"
    uid: str
    account_email: str
    subject: str
    from_address: str
    classification: EmailClassification
    proposed_action: EmailAction
    overall_confidence: float = Field(ge=0.0, le=1.0)
    gate_action: GateAction


# --- Extraction ---


class InvoiceData(BaseModel):
    """Extracted invoice fields."""

    vendor: str
    amount: float | None = None
    currency: str = "USD"
    due_date: str | None = None
    invoice_number: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ActionItem(BaseModel):
    """An action item extracted from an email."""

    description: str
    deadline: str | None = None
    priority: str = "normal"  # low, normal, high, urgent


class EmailExtractionResult(BaseModel):
    """LLM executor output — structured data extraction from emails."""

    invoice: InvoiceData | None = None
    action_items: list[ActionItem] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)


# --- Pipeline results ---


class EmailTriageResult(BaseModel):
    """Pipeline result for email triage."""

    account_email: str
    processed: int = 0
    auto_acted: int = 0
    queued: int = 0
    errors: int = 0
    categories: dict[str, int] = Field(default_factory=dict)  # category -> count


class EmailSummaryResult(BaseModel):
    """Pipeline result for inbox summarization."""

    account_email: str
    total_unread: int = 0
    summary: str = ""
    by_category: dict[str, int] = Field(default_factory=dict)
    important_subjects: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


# --- Review queue ---


class EmailReviewQueueItem(BaseModel):
    """An item in the email review queue."""

    id: str = Field(description="Unique queue item ID")
    created_at: datetime
    task: EmailTriageTask
    status: ReviewStatus = ReviewStatus.PENDING
    reviewed_at: datetime | None = None
    reviewer_notes: str | None = None


# --- Audit ---


class EmailAuditEntry(BaseModel):
    """A record of an email action taken (or queued) by the system."""

    timestamp: datetime
    action: Literal[
        "auto_applied", "queued_for_review", "review_approved", "review_rejected"
    ]
    account_email: str
    uid: str
    subject: str
    from_address: str
    category: EmailCategory
    email_action: EmailAction
    gate_action: GateAction
    applied: bool
