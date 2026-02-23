"""Document tagger executor — classifies a Paperless document via LLM.

Stateless: receives all inputs (document, available metadata, clients),
returns a fully populated DocumentTaggingTask. No side effects.
"""

import logging

from corvus.integrations.ollama import OllamaClient, OllamaResponse
from corvus.schemas.document_tagging import (
    DocumentTaggingResult,
    DocumentTaggingTask,
    GateAction,
)
from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocument,
    PaperlessDocumentType,
    PaperlessTag,
)

logger = logging.getLogger(__name__)

# Truncate document content sent to the LLM to stay within context limits.
MAX_CONTENT_CHARS = 8000

# Truncated snippet stored on the task for logging/review.
SNIPPET_CHARS = 500

# Confidence gate thresholds (from CLAUDE.md).
HIGH_CONFIDENCE = 0.9
MEDIUM_CONFIDENCE = 0.7

SYSTEM_PROMPT = """\
You are a document classification assistant for a personal document management system.

Your job is to analyze a document and suggest:
- **Tags** that describe the document's content and purpose.
- A **correspondent** (the person or organization the document is from/about).
- A **document type** (the category of document).

## Rules
1. Reuse existing tags, correspondents, and document types when they fit.
2. Suggest NEW names only when no existing option is a good match.
3. Tag names should be lowercase, concise, and use hyphens for multi-word tags \
(e.g. "utility-bill", "tax-return").
4. Assign a confidence score (0.0-1.0) to each suggestion. \
Be honest - use lower scores when uncertain.
5. Provide brief reasoning explaining your classification choices.

## Available Tags
{tags}

## Available Correspondents
{correspondents}

## Available Document Types
{document_types}
"""

USER_PROMPT = """\
Classify this document.

**Title:** {title}
**Filename:** {filename}

**Content:**
{content}
"""


def _compute_overall_confidence(result: DocumentTaggingResult) -> float:
    """Compute a weighted average confidence across all suggestions.

    Weighting: tag confidences contribute equally, correspondent and document
    type each count as one "vote" alongside the tags.
    """
    scores: list[float] = [t.confidence for t in result.suggested_tags]

    if result.suggested_correspondent:
        scores.append(result.correspondent_confidence)
    if result.suggested_document_type:
        scores.append(result.document_type_confidence)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _determine_gate_action(confidence: float) -> GateAction:
    """Map overall confidence to a gate action per CLAUDE.md thresholds."""
    if confidence >= HIGH_CONFIDENCE:
        return GateAction.AUTO_EXECUTE
    if confidence >= MEDIUM_CONFIDENCE:
        return GateAction.FLAG_IN_DIGEST
    return GateAction.QUEUE_FOR_REVIEW


def _build_system_prompt(
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
) -> str:
    tag_list = ", ".join(t.name for t in tags) if tags else "(none yet)"
    corr_list = ", ".join(c.name for c in correspondents) if correspondents else "(none yet)"
    dtype_list = ", ".join(d.name for d in document_types) if document_types else "(none yet)"

    return SYSTEM_PROMPT.format(
        tags=tag_list,
        correspondents=corr_list,
        document_types=dtype_list,
    )


def _build_user_prompt(doc: PaperlessDocument) -> str:
    content = doc.content[:MAX_CONTENT_CHARS]
    if len(doc.content) > MAX_CONTENT_CHARS:
        content += "\n\n[... content truncated ...]"

    return USER_PROMPT.format(
        title=doc.title,
        filename=doc.original_filename or "(unknown)",
        content=content,
    )


async def tag_document(
    document: PaperlessDocument,
    *,
    ollama: OllamaClient,
    model: str,
    tags: list[PaperlessTag],
    correspondents: list[PaperlessCorrespondent],
    document_types: list[PaperlessDocumentType],
    temperature: float = 0.2,
    keep_alive: str | None = None,
) -> tuple[DocumentTaggingTask, OllamaResponse]:
    """Classify a single document using the LLM.

    Args:
        document: The Paperless document to classify.
        ollama: An open OllamaClient instance.
        model: Ollama model name to use for inference.
        tags: All existing Paperless tags (for context).
        correspondents: All existing Paperless correspondents (for context).
        document_types: All existing Paperless document types (for context).
        temperature: LLM sampling temperature.
        keep_alive: Ollama keep_alive parameter.

    Returns:
        Tuple of (DocumentTaggingTask, OllamaResponse).
    """
    system = _build_system_prompt(tags, correspondents, document_types)
    prompt = _build_user_prompt(document)

    logger.info("Tagging document %d: %s", document.id, document.title)

    result, raw = await ollama.generate_structured(
        model=model,
        schema_class=DocumentTaggingResult,
        system=system,
        prompt=prompt,
        temperature=temperature,
        keep_alive=keep_alive,
    )

    overall_confidence = _compute_overall_confidence(result)
    gate_action = _determine_gate_action(overall_confidence)

    snippet = document.content[:SNIPPET_CHARS]
    if len(document.content) > SNIPPET_CHARS:
        snippet += "..."

    task = DocumentTaggingTask(
        document_id=document.id,
        document_title=document.title,
        content_snippet=snippet,
        result=result,
        overall_confidence=overall_confidence,
        gate_action=gate_action,
    )

    logger.info(
        "Document %d: confidence=%.2f gate=%s tags=%s",
        document.id,
        overall_confidence,
        gate_action.value,
        [t.tag_name for t in result.suggested_tags],
    )

    return task, raw
