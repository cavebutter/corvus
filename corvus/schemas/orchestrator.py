"""Schemas for the orchestrator / intent classification pipeline (Phase 2, Epic 6).

Covers: user intent classification -> orchestrator dispatch -> pipeline results.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """User intent categories recognized by the orchestrator."""

    TAG_DOCUMENTS = "tag_documents"
    FETCH_DOCUMENT = "fetch_document"
    REVIEW_QUEUE = "review_queue"
    SHOW_DIGEST = "show_digest"
    SHOW_STATUS = "show_status"
    WATCH_FOLDER = "watch_folder"
    GENERAL_CHAT = "general_chat"


# --- LLM Output Schema (what the intent classifier returns) ---


class IntentClassification(BaseModel):
    """Structured output from the LLM intent classifier.

    Flat model: all possible per-intent params are optional fields.
    The ``intent`` field determines which params are relevant.
    """

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    # TAG_DOCUMENTS params
    tag_limit: int | None = Field(
        default=None,
        description="Max documents to tag (0 or None = all)",
    )
    tag_include_tagged: bool = Field(
        default=False,
        description="Include already-tagged documents",
    )

    # FETCH_DOCUMENT params
    fetch_query: str | None = Field(
        default=None,
        description="Natural language search query to pass to the retrieval pipeline",
    )
    fetch_delivery_method: Literal["browser", "download"] | None = Field(
        default=None,
        description="How to deliver the result",
    )

    # SHOW_DIGEST params
    digest_hours: int | None = Field(
        default=None,
        description="Lookback period in hours for digest",
    )


# --- Pipeline Result Schemas ---


class TagPipelineResult(BaseModel):
    """Result from the tagging pipeline handler."""

    processed: int
    queued: int
    auto_applied: int
    errors: int
    details: list[str] = Field(default_factory=list)


class FetchPipelineResult(BaseModel):
    """Result from the fetch/retrieval pipeline handler."""

    documents_found: int
    documents: list[dict] = Field(
        default_factory=list,
        description="Simplified document dicts (id, title, created)",
    )
    warnings: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    interpretation_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence from the query interpretation step",
    )


class StatusResult(BaseModel):
    """Result from the status pipeline handler."""

    pending_count: int
    processed_24h: int
    reviewed_24h: int


class DigestResult(BaseModel):
    """Result from the digest pipeline handler."""

    rendered_text: str


# --- Orchestrator Response ---


class OrchestratorAction(StrEnum):
    """What the orchestrator decided to do."""

    DISPATCHED = "dispatched"
    NEEDS_CLARIFICATION = "needs_clarification"
    INTERACTIVE_REQUIRED = "interactive_required"
    CHAT_RESPONSE = "chat_response"


class OrchestratorResponse(BaseModel):
    """Unified response from the orchestrator router."""

    action: OrchestratorAction
    intent: Intent | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    clarification_prompt: str | None = None
    result: TagPipelineResult | FetchPipelineResult | StatusResult | DigestResult | None = None
