"""Schemas for the document tagging pipeline (Phase 1).

Covers the full lifecycle:
  LLM suggestion → confidence gate → review queue / auto-execute → audit log
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class GateAction(StrEnum):
    """Action determined by the confidence gate."""

    AUTO_EXECUTE = "auto_execute"
    FLAG_IN_DIGEST = "flag_in_digest"
    QUEUE_FOR_REVIEW = "queue_for_review"


class ReviewStatus(StrEnum):
    """Status of an item in the review queue."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


# --- LLM Output Schema (what the executor returns) ---


class TagSuggestion(BaseModel):
    """A single tag suggestion from the LLM."""

    tag_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class DocumentTaggingResult(BaseModel):
    """Structured output from the document tagger executor.

    This is the schema the LLM must conform to. The executor validates
    the raw LLM response against this model before returning.
    """

    suggested_tags: list[TagSuggestion]
    suggested_correspondent: str | None = None
    suggested_document_type: str | None = None
    correspondent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    document_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str


# --- Task Schema (what flows through the router) ---


class DocumentTaggingTask(BaseModel):
    """A tagging task emitted by the planner and routed to the executor."""

    task_type: Literal["tag_document"] = "tag_document"
    document_id: int
    document_title: str
    content_snippet: str = Field(
        description="Truncated content for logging/review (not the full OCR text)"
    )
    result: DocumentTaggingResult
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate confidence across all suggestions",
    )
    gate_action: GateAction


# --- Proposed Changes (what gets applied to Paperless) ---


class ProposedDocumentUpdate(BaseModel):
    """The concrete changes to apply to a Paperless document.

    Tag/correspondent/document_type IDs are resolved from names
    by the router before this is created.
    """

    document_id: int
    add_tag_ids: list[int] = Field(default_factory=list)
    set_correspondent_id: int | None = None
    set_document_type_id: int | None = None


# --- Review Queue Item ---


class ReviewQueueItem(BaseModel):
    """An item in the human review queue."""

    id: str = Field(description="Unique queue item ID")
    created_at: datetime
    task: DocumentTaggingTask
    proposed_update: ProposedDocumentUpdate
    status: ReviewStatus = ReviewStatus.PENDING
    reviewed_at: datetime | None = None
    reviewer_notes: str | None = None


# --- Audit Log Entry ---


class AuditEntry(BaseModel):
    """A record of an action taken (or queued) by the system."""

    timestamp: datetime
    action: Literal["auto_applied", "queued_for_review", "review_approved", "review_rejected"]
    document_id: int
    document_title: str
    task: DocumentTaggingTask
    proposed_update: ProposedDocumentUpdate
    gate_action: GateAction
    applied: bool = Field(description="Whether the update was actually written to Paperless")
