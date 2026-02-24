"""Deterministic router for the document tagging pipeline.

Receives a DocumentTaggingTask from the executor, resolves LLM-suggested
names to Paperless IDs, builds a ProposedDocumentUpdate, and either
applies it or holds it for review based on the confidence gate.

No LLM calls — pure Python logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from corvus.schemas.document_tagging import ReviewQueueItem

from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.document_tagging import (
    DocumentTaggingTask,
    GateAction,
    ProposedDocumentUpdate,
)
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocumentType,
    PaperlessTag,
)

logger = logging.getLogger(__name__)


class RoutingResult(BaseModel):
    """Outcome of routing a tagging task."""

    task: DocumentTaggingTask
    proposed_update: ProposedDocumentUpdate
    applied: bool = Field(description="Whether the update was written to Paperless")
    effective_action: GateAction = Field(
        description="The action that was actually taken (may differ from task.gate_action "
        "when force_queue is enabled)"
    )


# ------------------------------------------------------------------
# Name → ID resolution (case-insensitive)
# ------------------------------------------------------------------


def resolve_tag(name: str, existing: list[PaperlessTag]) -> int | None:
    """Match a tag name to an existing Paperless tag (case-insensitive)."""
    lower = name.lower().strip()
    for tag in existing:
        if tag.name.lower() == lower:
            return tag.id
    return None


def resolve_correspondent(
    name: str, existing: list[PaperlessCorrespondent]
) -> int | None:
    """Match a correspondent name to an existing one (case-insensitive)."""
    lower = name.lower().strip()
    for c in existing:
        if c.name.lower() == lower:
            return c.id
    return None


def resolve_document_type(
    name: str, existing: list[PaperlessDocumentType]
) -> int | None:
    """Match a document type name to an existing one (case-insensitive)."""
    lower = name.lower().strip()
    for dt in existing:
        if dt.name.lower() == lower:
            return dt.id
    return None


# ------------------------------------------------------------------
# Main routing function
# ------------------------------------------------------------------


async def resolve_and_route(
    task: DocumentTaggingTask,
    *,
    paperless: PaperlessClient,
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
    existing_doc_tag_ids: list[int],
    force_queue: bool = True,
    force_apply: bool = False,
) -> RoutingResult:
    """Resolve names to IDs, apply confidence gate, and optionally write to Paperless.

    Args:
        task: The tagging task from the executor.
        paperless: An open PaperlessClient for creating entities and applying updates.
        tags: All existing Paperless tags (for name resolution).
        correspondents: All existing Paperless correspondents.
        document_types: All existing Paperless document types.
        existing_doc_tag_ids: The document's current tag IDs (for merging).
        force_queue: If True, override the gate and queue everything for review.
            This is the initial posture per CLAUDE.md.
        force_apply: If True, override the gate and always apply the update.
            Used after human approval in the review flow.

    Returns:
        RoutingResult with the proposed update and whether it was applied.
    """
    # --- 1. Resolve names to IDs ---
    resolved_tag_ids: list[int] = []
    unresolved_tag_names: list[str] = []

    for suggestion in task.result.suggested_tags:
        tag_id = resolve_tag(suggestion.tag_name, tags)
        if tag_id is not None:
            resolved_tag_ids.append(tag_id)
        else:
            unresolved_tag_names.append(suggestion.tag_name)

    resolved_correspondent_id: int | None = None
    unresolved_correspondent: str | None = None
    if task.result.suggested_correspondent:
        resolved_correspondent_id = resolve_correspondent(
            task.result.suggested_correspondent, correspondents
        )
        if resolved_correspondent_id is None:
            unresolved_correspondent = task.result.suggested_correspondent

    resolved_doc_type_id: int | None = None
    unresolved_doc_type: str | None = None
    if task.result.suggested_document_type:
        resolved_doc_type_id = resolve_document_type(
            task.result.suggested_document_type, document_types
        )
        if resolved_doc_type_id is None:
            unresolved_doc_type = task.result.suggested_document_type

    logger.info(
        "Document %d: resolved %d/%d tags, correspondent=%s, doc_type=%s",
        task.document_id,
        len(resolved_tag_ids),
        len(task.result.suggested_tags),
        "resolved" if resolved_correspondent_id else (unresolved_correspondent or "none"),
        "resolved" if resolved_doc_type_id else (unresolved_doc_type or "none"),
    )

    # --- 2. Determine effective action ---
    if force_apply:
        effective_action = GateAction.AUTO_EXECUTE
    elif force_queue:
        effective_action = GateAction.QUEUE_FOR_REVIEW
    else:
        effective_action = task.gate_action
    should_apply = effective_action in (GateAction.AUTO_EXECUTE, GateAction.FLAG_IN_DIGEST)

    # --- 3. Create missing entities if we're applying ---
    if should_apply:
        for name in unresolved_tag_names:
            new_tag = await paperless.create_tag(name)
            resolved_tag_ids.append(new_tag.id)
            logger.info("Created new tag: %s (id=%d)", name, new_tag.id)

        if unresolved_correspondent:
            new_corr = await paperless.create_correspondent(unresolved_correspondent)
            resolved_correspondent_id = new_corr.id
            logger.info(
                "Created new correspondent: %s (id=%d)",
                unresolved_correspondent,
                new_corr.id,
            )

        if unresolved_doc_type:
            new_dt = await paperless.create_document_type(unresolved_doc_type)
            resolved_doc_type_id = new_dt.id
            logger.info(
                "Created new document type: %s (id=%d)",
                unresolved_doc_type,
                new_dt.id,
            )

    # --- 4. Build proposed update ---
    proposed = ProposedDocumentUpdate(
        document_id=task.document_id,
        add_tag_ids=resolved_tag_ids,
        set_correspondent_id=resolved_correspondent_id,
        set_document_type_id=resolved_doc_type_id,
    )

    # --- 5. Apply to Paperless if gate allows ---
    applied = False
    if should_apply:
        patch: dict = {}

        if resolved_tag_ids:
            # Paperless PATCH replaces the full tag list, so merge existing + new
            merged_tags = list(set(existing_doc_tag_ids + resolved_tag_ids))
            patch["tags"] = merged_tags

        if resolved_correspondent_id is not None:
            patch["correspondent"] = resolved_correspondent_id

        if resolved_doc_type_id is not None:
            patch["document_type"] = resolved_doc_type_id

        if patch:
            await paperless.update_document(task.document_id, patch)
            applied = True
            logger.info(
                "Applied update to document %d (gate=%s)",
                task.document_id,
                effective_action.value,
            )
        else:
            logger.info("Document %d: nothing to apply (no resolved IDs)", task.document_id)
    else:
        logger.info(
            "Document %d: queued for review (confidence=%.2f, force_queue=%s)",
            task.document_id,
            task.overall_confidence,
            force_queue,
        )

    return RoutingResult(
        task=task,
        proposed_update=proposed,
        applied=applied,
        effective_action=effective_action,
    )


async def apply_approved_update(
    item: "ReviewQueueItem",
    *,
    paperless: PaperlessClient,
    extra_tag_names: list[str] | None = None,
) -> RoutingResult:
    """Apply a previously-reviewed queue item to Paperless.

    Re-fetches current Paperless state (tags, correspondents, document types,
    and the document itself) to get up-to-date IDs, then routes with
    force_apply=True to create any missing entities and write the update.

    Args:
        item: The approved ReviewQueueItem.
        paperless: An open PaperlessClient.
        extra_tag_names: Additional tag names to apply (user-added during review).

    Returns:
        RoutingResult with applied=True on success.
    """
    task = item.task
    if extra_tag_names:
        from corvus.schemas.document_tagging import TagSuggestion

        extra_suggestions = [
            TagSuggestion(tag_name=name, confidence=1.0) for name in extra_tag_names
        ]
        task = task.model_copy(
            update={
                "result": task.result.model_copy(
                    update={
                        "suggested_tags": list(task.result.suggested_tags)
                        + extra_suggestions,
                    }
                )
            }
        )

    doc = await paperless.get_document(task.document_id)
    tags = await paperless.list_tags()
    correspondents = await paperless.list_correspondents()
    document_types = await paperless.list_document_types()

    return await resolve_and_route(
        task,
        paperless=paperless,
        tags=tags,
        correspondents=correspondents,
        document_types=document_types,
        existing_doc_tag_ids=doc.tags,
        force_apply=True,
    )
