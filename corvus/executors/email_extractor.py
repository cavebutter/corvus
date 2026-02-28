"""Email data extractor executor — extracts structured data from emails via LLM.

Stateless: receives an email and its classification, returns extracted
structured data (invoice fields, action items, key dates). Only runs on
emails classified as RECEIPT, INVOICE, or ACTION_REQUIRED.
"""

import logging

from corvus.integrations.ollama import OllamaClient, OllamaResponse
from corvus.schemas.email import (
    EmailCategory,
    EmailClassification,
    EmailExtractionResult,
    EmailMessage,
)

logger = logging.getLogger(__name__)

# Truncate email body to stay within context limits.
MAX_BODY_CHARS = 4000

# Categories that warrant data extraction.
EXTRACTABLE_CATEGORIES = frozenset({
    EmailCategory.RECEIPT,
    EmailCategory.INVOICE,
    EmailCategory.ACTION_REQUIRED,
})

SYSTEM_PROMPT = """\
You are a data extraction assistant for a personal email management system.

Your job is to extract structured data from emails. Based on the email type, extract:

## For Receipts & Invoices
- **vendor**: The company/person who sent the bill or receipt
- **amount**: The total amount (number only, no currency symbols)
- **currency**: Three-letter currency code (default USD)
- **due_date**: Payment due date if present (YYYY-MM-DD format)
- **invoice_number**: Invoice or order number if present
- **confidence**: How confident you are in the extraction (0.0-1.0)

## For Action Required Emails
- **action_items**: List of specific actions the recipient needs to take
  - description: What needs to be done
  - deadline: When it's due (YYYY-MM-DD format) if mentioned
  - priority: low, normal, high, or urgent

## For All Types
- **key_dates**: Any significant dates mentioned (YYYY-MM-DD format)

## Rules

1. Only extract data that is explicitly stated in the email.
2. Do NOT guess or infer amounts, dates, or invoice numbers.
3. If a field is not present, leave it null/empty.
4. For amounts, extract the total/final amount, not subtotals.
5. Be conservative with confidence scores.
"""

USER_PROMPT = """\
Extract structured data from this email.

**Category:** {category}
**From:** {from_address}
**Subject:** {subject}

**Body:**
{body}
"""


async def extract_email_data(
    email: EmailMessage,
    classification: EmailClassification,
    *,
    ollama: OllamaClient,
    model: str,
    keep_alive: str | None = None,
) -> tuple[EmailExtractionResult, OllamaResponse]:
    """Extract structured data from a receipt/invoice/action email.

    Only runs on emails classified as RECEIPT, INVOICE, or ACTION_REQUIRED.
    For other categories, returns an empty result without calling the LLM.

    Args:
        email: The full email message.
        classification: The email's classification from the classifier.
        ollama: An open OllamaClient instance.
        model: Ollama model name to use for inference.
        keep_alive: Ollama keep_alive parameter.

    Returns:
        Tuple of (EmailExtractionResult, OllamaResponse).

    Raises:
        ValueError: If the email category is not extractable (should not
            happen if caller checks first).
    """
    if classification.category not in EXTRACTABLE_CATEGORIES:
        raise ValueError(
            f"Extraction not applicable for category: {classification.category.value}"
        )

    body = email.body_text or email.body_html or "(no body)"
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n[... content truncated ...]"

    prompt = USER_PROMPT.format(
        category=classification.category.value,
        from_address=email.from_address,
        subject=email.subject,
        body=body,
    )

    logger.info(
        "Extracting data from email %s (category=%s)",
        email.uid,
        classification.category.value,
    )

    result, raw = await ollama.generate_structured(
        model=model,
        schema_class=EmailExtractionResult,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        keep_alive=keep_alive,
    )

    logger.info(
        "Extraction for %s: invoice=%s action_items=%d key_dates=%d",
        email.uid,
        result.invoice is not None,
        len(result.action_items),
        len(result.key_dates),
    )

    return result, raw
