"""End-to-end integration test for the document tagging pipeline.

Exercises the full flow:
  Paperless document → LLM executor → router (force_queue) →
  review queue → audit log → daily digest

Requires live Paperless and Ollama instances.
"""

import pytest

from corvus.audit.logger import AuditLog
from corvus.config import OLLAMA_BASE_URL, PAPERLESS_API_TOKEN, PAPERLESS_BASE_URL
from corvus.digest.daily import generate_digest, render_text
from corvus.executors.document_tagger import tag_document
from corvus.integrations.ollama import OllamaClient, pick_instruct_model
from corvus.integrations.paperless import PaperlessClient
from corvus.queue.review import ReviewQueue
from corvus.router.tagging import resolve_and_route
from corvus.schemas.document_tagging import GateAction, ReviewStatus


@pytest.mark.slow
@pytest.mark.skipif(
    not PAPERLESS_API_TOKEN or PAPERLESS_API_TOKEN == "placeholder",
    reason="No real Paperless API token configured",
)
async def test_full_tagging_pipeline(tmp_path):
    """Full pipeline: fetch → classify → route → queue → audit → digest."""

    audit_log = AuditLog(tmp_path / "audit.jsonl")

    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL) as ollama,
    ):
        with ReviewQueue(tmp_path / "queue.db") as review_queue:
            # --- 1. Discover model and document ---
            models = await ollama.list_models()
            model_name = pick_instruct_model(models)
            if not model_name:
                pytest.skip("No models available on Ollama server")

            docs, _count = await paperless.list_documents(page_size=1)
            if not docs:
                pytest.skip("No documents in Paperless instance")
            doc = docs[0]

            tags = await paperless.list_tags()
            correspondents = await paperless.list_correspondents()
            doc_types = await paperless.list_document_types()

            # --- 2. Executor: classify via LLM ---
            task, raw = await tag_document(
                doc,
                ollama=ollama,
                model=model_name,
                tags=tags,
                correspondents=correspondents,
                document_types=doc_types,
                keep_alive="0",
            )

            assert task.document_id == doc.id
            assert raw.done is True

            # --- 3. Router: resolve names and apply gate ---
            routing_result = await resolve_and_route(
                task,
                paperless=paperless,
                tags=tags,
                correspondents=correspondents,
                document_types=doc_types,
                existing_doc_tag_ids=doc.tags,
                force_queue=True,
            )

            assert routing_result.applied is False
            assert routing_result.effective_action == GateAction.QUEUE_FOR_REVIEW

            # --- 4. Review queue: add item ---
            queue_item = review_queue.add(
                routing_result.task,
                routing_result.proposed_update,
            )

            assert queue_item.status == ReviewStatus.PENDING
            assert review_queue.count_pending() == 1

            # --- 5. Audit log: record the queued action ---
            audit_log.log_queued_for_review(
                routing_result.task,
                routing_result.proposed_update,
            )

            # --- 6. Simulate human approval ---
            review_queue.approve(queue_item.id, notes="E2E test approval")

            approved_item = review_queue.get(queue_item.id)
            assert approved_item.status == ReviewStatus.APPROVED
            assert review_queue.count_pending() == 0

            audit_log.log_review_approved(
                routing_result.task,
                routing_result.proposed_update,
            )

            # --- 7. Daily digest: verify everything shows up ---
            digest = generate_digest(audit_log, review_queue)

            assert len(digest.queued_for_review) == 1
            assert len(digest.review_approved) == 1
            assert digest.pending_review_count == 0
            assert digest.total_processed == 1
            assert digest.total_reviewed == 1

            text = render_text(digest)
            assert "Corvus Daily Digest" in text
            assert doc.title in text
            assert "Queued for Review" in text
            assert "Approved" in text
