# Corvus — Session Continuity Notes

> Updated at the end of each session or before conversation compacting. Read at the start of each new session.

---

## Session: 2026-02-23 (session 2)

### What was done
- Completed S2.1: Pydantic schemas for the document tagging pipeline
  - `corvus/schemas/paperless.py` — API response models (Document, Tag, Correspondent, DocumentType, PaginatedResponse)
  - `corvus/schemas/document_tagging.py` — full pipeline lifecycle (TagSuggestion, DocumentTaggingResult, DocumentTaggingTask, GateAction, ProposedDocumentUpdate, ReviewQueueItem, AuditEntry)
- Completed S2.2: Async Paperless-ngx API client
  - `corvus/integrations/paperless.py` — list/get/update documents, list/create tags/correspondents/doc types, auto-pagination
  - `tests/test_paperless_client.py` — 4 tests, all passing against live instance
  - Fixed: `original_filename` made optional (defaulted to `""`) — not always present in API response
- Completed S2.3: Ollama LLM client for structured output
  - `corvus/integrations/ollama.py` — `generate_structured()` sends Pydantic JSON schema as `format` param, returns validated model instance + raw response with timing/token metrics. Also: `list_models()`, `unload_model()`
  - `tests/test_ollama_client.py` — 2 tests passing (model list + structured output with DocumentTaggingResult)
- Added `requirements.txt` for PyCharm compatibility
- Added Epic 3 (Paperless Document Retrieval — post-MVP) and Epic 4 (Local Scan Folder Watchdog — post-MVP) to backlog
- Registered `slow` pytest mark in pyproject.toml

### Current state
- **Epic 1:** Complete (all 6 stories)
- **Epic 2:** S2.1, S2.2, S2.3 complete. Next is S2.4 (document tagger executor).
- **Epics 3 & 4:** Backlogged (post-MVP)
- **All tests passing:** 6 total (4 Paperless, 2 Ollama) against live services
- **Paperless instance:** `http://athena.eagle-mimosa.ts.net:8000` (token configured in secrets/internal.env)
- **Ollama:** Running locally on default port, at least one model available

### Decisions made
- LLM output is constrained via Pydantic `model_json_schema()` passed to Ollama's `format` param — guarantees valid structured JSON
- `keep_alive` defaults to `"5m"`, tests use `"0"` to free VRAM immediately
- Paperless API version header set to `version=5`

### Blockers
- None

### Next steps
- S2.4: Implement document tagger executor (`corvus/executors/document_tagger.py`) — ties together Paperless client + Ollama client, sends document content to LLM, returns validated `DocumentTaggingResult`
- S2.5: Confidence gate logic in router
- S2.6: Review queue (SQLite)
- Then: audit logging, daily digest, end-to-end test
