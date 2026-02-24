"""Schemas for the document retrieval pipeline (Epic 3).

Covers: LLM query interpretation -> name resolution -> Paperless search -> delivery.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class DeliveryMethod(StrEnum):
    """How to deliver a retrieved document to the user."""

    BROWSER = "browser"
    DOWNLOAD = "download"


# --- LLM Output Schema (what the query interpreter returns) ---


class QueryInterpretation(BaseModel):
    """Structured output from the LLM query interpreter.

    The LLM parses a natural language query into searchable fields.
    """

    correspondent_name: str | None = None
    document_type_name: str | None = None
    tag_names: list[str] = Field(default_factory=list)
    text_search: str | None = None
    date_range_start: str | None = Field(
        default=None, description="ISO YYYY-MM-DD"
    )
    date_range_end: str | None = Field(
        default=None, description="ISO YYYY-MM-DD"
    )
    sort_order: Literal["newest", "oldest", "relevance"] = "newest"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# --- Router Output (resolved IDs + filter params for Paperless API) ---


class ResolvedSearchParams(BaseModel):
    """Search parameters with names resolved to Paperless IDs."""

    text_search: str | None = None
    correspondent_id: int | None = None
    document_type_id: int | None = None
    tag_ids: list[int] = Field(default_factory=list)
    date_range_start: str | None = None
    date_range_end: str | None = None
    ordering: str | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about unresolved names or other issues",
    )
    used_fallback: bool = False
