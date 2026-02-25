"""Deterministic orchestrator router — dispatches classified intents to pipelines.

Receives an IntentClassification from the intent classifier, applies the
confidence gate, and dispatches to the appropriate pipeline handler.
No LLM calls (except for GENERAL_CHAT which delegates to ollama.chat()).
"""

import logging
from collections.abc import Callable

from corvus.integrations.ollama import OllamaClient
from corvus.integrations.paperless import PaperlessClient
from corvus.schemas.orchestrator import (
    Intent,
    IntentClassification,
    OrchestratorAction,
    OrchestratorResponse,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7

CORVUS_PERSONA = """\
You are Corvus, a helpful AI assistant for a personal homelab document \
management system. You are knowledgeable, concise, and friendly. \
Keep responses brief but informative.\
"""


async def dispatch(
    classification: IntentClassification,
    user_input: str,
    *,
    paperless: PaperlessClient,
    ollama: OllamaClient,
    model: str,
    keep_alive: str = "5m",
    queue_db_path: str,
    audit_log_path: str,
    on_progress: Callable[[str], None] | None = None,
    chat_model: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> OrchestratorResponse:
    """Route a classified intent to the appropriate pipeline handler.

    Args:
        classification: Intent classification from the LLM.
        user_input: Original user text (needed for GENERAL_CHAT).
        paperless: An open PaperlessClient.
        ollama: An open OllamaClient.
        model: Ollama model name.
        keep_alive: Ollama keep_alive duration.
        queue_db_path: Path to the review queue SQLite database.
        audit_log_path: Path to the audit log file.
        on_progress: Optional callback for progress messages.

    Returns:
        OrchestratorResponse with the action taken and any results.
    """
    intent = classification.intent
    confidence = classification.confidence
    effective_chat_model = chat_model or model

    # Confidence gate
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Low confidence (%.2f) for intent %s, asking for clarification",
            confidence, intent.value,
        )
        return OrchestratorResponse(
            action=OrchestratorAction.NEEDS_CLARIFICATION,
            intent=intent,
            confidence=confidence,
            message=f"I'm not sure I understood that (confidence: {confidence:.0%}).",
            clarification_prompt="Could you rephrase your request?",
        )

    # Interactive-only commands
    if intent == Intent.REVIEW_QUEUE:
        return OrchestratorResponse(
            action=OrchestratorAction.INTERACTIVE_REQUIRED,
            intent=intent,
            confidence=confidence,
            message="The review queue requires interactive input. Use `corvus review` directly.",
        )

    if intent == Intent.WATCH_FOLDER:
        return OrchestratorResponse(
            action=OrchestratorAction.INTERACTIVE_REQUIRED,
            intent=intent,
            confidence=confidence,
            message="The folder watcher is a long-running process. Use `corvus watch` directly.",
        )

    # Pipeline dispatches
    if intent == Intent.TAG_DOCUMENTS:
        return await _dispatch_tag(
            classification, paperless=paperless, ollama=ollama, model=model,
            keep_alive=keep_alive, queue_db_path=queue_db_path,
            audit_log_path=audit_log_path, on_progress=on_progress,
        )

    if intent == Intent.FETCH_DOCUMENT:
        return await _dispatch_fetch(
            classification, paperless=paperless, ollama=ollama, model=model,
            keep_alive=keep_alive, on_progress=on_progress,
        )

    if intent == Intent.SHOW_STATUS:
        return _dispatch_status(
            queue_db_path=queue_db_path, audit_log_path=audit_log_path,
            confidence=confidence,
        )

    if intent == Intent.SHOW_DIGEST:
        return _dispatch_digest(
            classification, queue_db_path=queue_db_path,
            audit_log_path=audit_log_path, confidence=confidence,
        )

    if intent == Intent.WEB_SEARCH:
        return await _dispatch_search(
            classification, user_input=user_input,
            ollama=ollama, model=effective_chat_model, keep_alive=keep_alive,
            on_progress=on_progress,
        )

    if intent == Intent.GENERAL_CHAT:
        return await _dispatch_chat(
            user_input, ollama=ollama, model=effective_chat_model,
            keep_alive=keep_alive, confidence=confidence,
            conversation_history=conversation_history,
        )

    # Should not reach here, but handle gracefully
    return OrchestratorResponse(
        action=OrchestratorAction.NEEDS_CLARIFICATION,
        intent=intent,
        confidence=confidence,
        message="I'm not sure how to handle that request.",
        clarification_prompt="Could you try rephrasing?",
    )


async def _dispatch_tag(
    classification: IntentClassification,
    *,
    paperless: PaperlessClient,
    ollama: OllamaClient,
    model: str,
    keep_alive: str,
    queue_db_path: str,
    audit_log_path: str,
    on_progress: Callable[[str], None] | None,
) -> OrchestratorResponse:
    from corvus.orchestrator.pipelines import run_tag_pipeline

    result = await run_tag_pipeline(
        paperless=paperless,
        ollama=ollama,
        model=model,
        limit=classification.tag_limit or 0,
        include_tagged=classification.tag_include_tagged,
        keep_alive=keep_alive,
        force_queue=True,
        queue_db_path=queue_db_path,
        audit_log_path=audit_log_path,
        on_progress=on_progress,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.DISPATCHED,
        intent=Intent.TAG_DOCUMENTS,
        confidence=classification.confidence,
        result=result,
    )


async def _dispatch_fetch(
    classification: IntentClassification,
    *,
    paperless: PaperlessClient,
    ollama: OllamaClient,
    model: str,
    keep_alive: str,
    on_progress: Callable[[str], None] | None,
) -> OrchestratorResponse:
    from corvus.orchestrator.pipelines import run_fetch_pipeline

    query = classification.fetch_query
    if not query:
        return OrchestratorResponse(
            action=OrchestratorAction.NEEDS_CLARIFICATION,
            intent=Intent.FETCH_DOCUMENT,
            confidence=classification.confidence,
            message="I understood you want to find a document, but I need more details.",
            clarification_prompt="What document are you looking for?",
        )

    result = await run_fetch_pipeline(
        paperless=paperless,
        ollama=ollama,
        model=model,
        query=query,
        keep_alive=keep_alive,
        on_progress=on_progress,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.DISPATCHED,
        intent=Intent.FETCH_DOCUMENT,
        confidence=classification.confidence,
        result=result,
    )


def _dispatch_status(
    *,
    queue_db_path: str,
    audit_log_path: str,
    confidence: float,
) -> OrchestratorResponse:
    from corvus.orchestrator.pipelines import run_status_pipeline

    result = run_status_pipeline(
        queue_db_path=queue_db_path,
        audit_log_path=audit_log_path,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.DISPATCHED,
        intent=Intent.SHOW_STATUS,
        confidence=confidence,
        result=result,
    )


def _dispatch_digest(
    classification: IntentClassification,
    *,
    queue_db_path: str,
    audit_log_path: str,
    confidence: float,
) -> OrchestratorResponse:
    from corvus.orchestrator.pipelines import run_digest_pipeline

    result = run_digest_pipeline(
        queue_db_path=queue_db_path,
        audit_log_path=audit_log_path,
        hours=classification.digest_hours or 24,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.DISPATCHED,
        intent=Intent.SHOW_DIGEST,
        confidence=confidence,
        result=result,
    )


async def _dispatch_search(
    classification: IntentClassification,
    *,
    user_input: str,
    ollama: OllamaClient,
    model: str,
    keep_alive: str,
    on_progress: Callable[[str], None] | None,
) -> OrchestratorResponse:
    from corvus.config import WEB_SEARCH_MAX_RESULTS
    from corvus.orchestrator.pipelines import run_search_pipeline

    query = classification.search_query or user_input

    result = await run_search_pipeline(
        ollama=ollama,
        model=model,
        query=query,
        keep_alive=keep_alive,
        max_results=WEB_SEARCH_MAX_RESULTS,
        on_progress=on_progress,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.DISPATCHED,
        intent=Intent.WEB_SEARCH,
        confidence=classification.confidence,
        result=result,
    )


async def _dispatch_chat(
    user_input: str,
    *,
    ollama: OllamaClient,
    model: str,
    keep_alive: str,
    confidence: float,
    conversation_history: list[dict[str, str]] | None = None,
) -> OrchestratorResponse:
    text, _raw = await ollama.chat(
        model=model,
        system=CORVUS_PERSONA,
        prompt=user_input,
        keep_alive=keep_alive,
        messages=conversation_history,
    )

    return OrchestratorResponse(
        action=OrchestratorAction.CHAT_RESPONSE,
        intent=Intent.GENERAL_CHAT,
        confidence=confidence,
        message=text,
    )
