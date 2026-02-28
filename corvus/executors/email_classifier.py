"""Email classifier executor — classifies an email via LLM.

Stateless: receives all inputs (email, clients), returns a fully populated
EmailTriageTask. No side effects. Follows the same pattern as document_tagger.py.
"""

import logging

from corvus.integrations.ollama import OllamaClient, OllamaResponse
from corvus.schemas.document_tagging import GateAction
from corvus.schemas.email import (
    EmailAction,
    EmailActionType,
    EmailCategory,
    EmailClassification,
    EmailMessage,
    EmailTriageTask,
)

logger = logging.getLogger(__name__)

# Truncate email body sent to the LLM to stay within context limits.
MAX_BODY_CHARS = 2000

# Confidence gate thresholds (from CLAUDE.md).
HIGH_CONFIDENCE = 0.9
MEDIUM_CONFIDENCE = 0.7

SYSTEM_PROMPT = """\
You are an email classification assistant for a personal email management system.

Your job is to analyze an email and determine:
- **Category**: What type of email this is.
- **Suggested action**: What should be done with it.
- **Confidence**: How certain you are (0.0-1.0).
- **Summary**: A 1-2 sentence summary of the email content.
- **Is automated**: Whether this is from a bot/system/noreply address.

## Categories

- **spam**: Unsolicited junk, phishing, scams
- **newsletter**: Mailing lists, marketing, promotional content
- **receipt**: Purchase confirmations, order receipts
- **invoice**: Bills, payment requests, statements
- **package_notice**: Shipping notifications, delivery updates
- **action_required**: Emails requiring a response or action
- **personal**: Direct personal correspondence
- **important**: High-priority business or legal correspondence
- **other**: Anything that doesn't fit above

## Suggested Actions

- "delete" — for spam
- "move_to_receipts" — for receipts and invoices
- "move_to_processed" — for newsletters, package notices, and other low-priority
- "flag" — for important or action-required emails
- "keep" — for personal emails and anything uncertain

## Rules

1. Be conservative with confidence scores. Use lower scores when uncertain.
2. Spam from clearly illegitimate senders should have high confidence.
3. Receipts/invoices from known vendors (Amazon, PayPal, banks) should have \
high confidence.
4. If the email could be personal or important, err on the side of "keep".
5. Automated emails (noreply@, marketing@, notifications@) should set \
is_automated to true.
6. Provide brief reasoning explaining your classification.
"""

USER_PROMPT = """\
Classify this email.

**From:** {from_address} ({from_name})
**To:** {to}
**Subject:** {subject}
**Date:** {date}

**Body:**
{body}
"""


# Maps EmailCategory -> (EmailActionType, folder_key_or_none)
_CATEGORY_ACTION_MAP: dict[EmailCategory, tuple[EmailActionType, str | None]] = {
    EmailCategory.SPAM: (EmailActionType.DELETE, None),
    EmailCategory.NEWSLETTER: (EmailActionType.MOVE, "processed"),
    EmailCategory.RECEIPT: (EmailActionType.MOVE, "receipts"),
    EmailCategory.INVOICE: (EmailActionType.MOVE, "receipts"),
    EmailCategory.PACKAGE_NOTICE: (EmailActionType.MOVE, "processed"),
    EmailCategory.ACTION_REQUIRED: (EmailActionType.FLAG, None),
    EmailCategory.PERSONAL: (EmailActionType.KEEP, None),
    EmailCategory.IMPORTANT: (EmailActionType.FLAG, None),
    EmailCategory.OTHER: (EmailActionType.KEEP, None),
}


def _determine_gate_action(confidence: float) -> GateAction:
    """Map confidence to a gate action per CLAUDE.md thresholds."""
    if confidence >= HIGH_CONFIDENCE:
        return GateAction.AUTO_EXECUTE
    if confidence >= MEDIUM_CONFIDENCE:
        return GateAction.FLAG_IN_DIGEST
    return GateAction.QUEUE_FOR_REVIEW


def _build_action(
    category: EmailCategory,
    folders: dict[str, str],
) -> EmailAction:
    """Map a classification category to a concrete IMAP action."""
    action_type, folder_key = _CATEGORY_ACTION_MAP.get(
        category, (EmailActionType.KEEP, None)
    )

    target_folder = None
    if action_type == EmailActionType.MOVE and folder_key:
        target_folder = folders.get(folder_key)

    flag_name = None
    if action_type == EmailActionType.FLAG:
        flag_name = "\\Flagged"

    return EmailAction(
        action_type=action_type,
        target_folder=target_folder,
        flag_name=flag_name,
    )


def _build_user_prompt(email: EmailMessage) -> str:
    """Build the user prompt from an email message."""
    body = email.body_text or email.body_html or "(no body)"
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n[... content truncated ...]"

    return USER_PROMPT.format(
        from_address=email.from_address,
        from_name=email.from_name or "(unknown)",
        to=", ".join(email.to) if email.to else "(unknown)",
        subject=email.subject,
        date=email.date.isoformat() if email.date else "(unknown)",
        body=body,
    )


async def classify_email(
    email: EmailMessage,
    *,
    ollama: OllamaClient,
    model: str,
    folders: dict[str, str] | None = None,
    keep_alive: str | None = None,
) -> tuple[EmailTriageTask, OllamaResponse]:
    """Classify an email and propose an action.

    Args:
        email: The full email message to classify.
        ollama: An open OllamaClient instance.
        model: Ollama model name to use for inference.
        folders: Account folder mapping (logical name -> IMAP path).
            Used to resolve target folders for move actions.
        keep_alive: Ollama keep_alive parameter.

    Returns:
        Tuple of (EmailTriageTask, OllamaResponse).
    """
    prompt = _build_user_prompt(email)
    folders = folders or {}

    logger.info("Classifying email: %s from %s", email.subject, email.from_address)

    classification, raw = await ollama.generate_structured(
        model=model,
        schema_class=EmailClassification,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        keep_alive=keep_alive,
    )

    proposed_action = _build_action(classification.category, folders)
    gate_action = _determine_gate_action(classification.confidence)

    task = EmailTriageTask(
        uid=email.uid,
        account_email=email.account_email,
        subject=email.subject,
        from_address=email.from_address,
        classification=classification,
        proposed_action=proposed_action,
        overall_confidence=classification.confidence,
        gate_action=gate_action,
    )

    logger.info(
        "Email %s: category=%s confidence=%.2f gate=%s action=%s",
        email.uid,
        classification.category.value,
        classification.confidence,
        gate_action.value,
        proposed_action.action_type.value,
    )

    return task, raw
