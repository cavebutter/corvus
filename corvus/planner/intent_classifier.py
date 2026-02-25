"""Intent classifier executor — classifies user natural language input via LLM.

Stateless: receives user input and clients, returns an IntentClassification.
No side effects. Follows the same pattern as document_tagger.py and
query_interpreter.py.
"""

import logging

from corvus.integrations.ollama import OllamaClient, OllamaResponse
from corvus.schemas.orchestrator import IntentClassification

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Corvus, an AI assistant for a personal document management system \
powered by Paperless-ngx. Your job is to classify user input into one of \
the following intents and extract any relevant parameters.

## Intents

1. **tag_documents** — User wants to tag/classify documents.
   Examples: "tag my documents", "classify the new scans", "run the tagger"
   Params: tag_limit (int, optional), tag_include_tagged (bool, default false)

2. **fetch_document** — User wants to find or retrieve a specific document.
   Examples: "find my AT&T invoice", "get the latest tax return", \
"show me documents from last month"
   Params: fetch_query (str, required — preserve the user's query language), \
fetch_delivery_method ("browser" or "download", optional)

3. **review_queue** — User wants to review pending items in the approval queue.
   Examples: "review pending items", "check the queue", "approve documents"
   No params.

4. **show_digest** — User wants a summary of recent activity.
   Examples: "what happened today", "show me the digest", "activity summary"
   Params: digest_hours (int, optional, default 24)

5. **show_status** — User wants a quick status overview.
   Examples: "status", "how many pending", "what's the queue look like"
   No params.

6. **watch_folder** — User wants to start watching a folder for new files.
   Examples: "watch my scan folder", "start the file watcher"
   No params.

7. **general_chat** — Anything else: greetings, questions, conversation.
   Examples: "hello", "what can you do?", "tell me a joke"
   No params.

## Rules

1. When ambiguous between tag_documents and fetch_document, prefer \
fetch_document (read-only, less destructive).
2. For fetch_document, copy the user's search terms into fetch_query as-is. \
Do NOT rewrite or interpret them — the downstream search pipeline handles that.
3. Assign a confidence score (0.0-1.0). Use lower scores when the intent is \
genuinely ambiguous.
4. Provide brief reasoning explaining your classification.
5. Only extract parameters that are explicitly mentioned or clearly implied.
"""

USER_PROMPT = """\
Classify this user input:

{user_input}
"""


async def classify_intent(
    user_input: str,
    *,
    ollama: OllamaClient,
    model: str,
    keep_alive: str | None = None,
) -> tuple[IntentClassification, OllamaResponse]:
    """Classify user natural language input into an intent.

    Args:
        user_input: Raw user input text.
        ollama: An open OllamaClient instance.
        model: Ollama model name to use for inference.
        keep_alive: Ollama keep_alive parameter.

    Returns:
        Tuple of (IntentClassification, OllamaResponse).
    """
    prompt = USER_PROMPT.format(user_input=user_input)

    logger.info("Classifying intent: %s", user_input)

    classification, raw = await ollama.generate_structured(
        model=model,
        schema_class=IntentClassification,
        system=SYSTEM_PROMPT,
        prompt=prompt,
        keep_alive=keep_alive,
    )

    logger.info(
        "Intent classified: intent=%s confidence=%.2f reasoning=%s",
        classification.intent.value,
        classification.confidence,
        classification.reasoning,
    )

    return classification, raw
