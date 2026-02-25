# Corvus — Session Continuity Notes

> Updated at the end of each session or before conversation compacting. Read at the start of each new session.

---

## Session: 2026-02-24 (session 9)

### What was done
- Completed Epic 6: Orchestrator Architecture — Phase 2 (5 stories: S6.1–S6.5)
- **S6.1** `corvus/schemas/orchestrator.py` (new): `Intent` enum (7 intents), `IntentClassification` flat model with per-intent optional params, pipeline result models (`TagPipelineResult`, `FetchPipelineResult`, `StatusResult`, `DigestResult`), `OrchestratorAction`, `OrchestratorResponse`
- **S6.2** `corvus/planner/intent_classifier.py` (new): Stateless async `classify_intent()` — calls Ollama with IntentClassification JSON schema; system prompt describes 7 intents + param extraction rules; "when ambiguous prefer fetch" rule
- **S6.3** `corvus/orchestrator/pipelines.py` (new): Extracted 4 pipeline handlers from CLI:
  - `run_tag_pipeline()` — full fetch→tag→route loop, returns `TagPipelineResult` with counts
  - `run_fetch_pipeline()` — interpret→search, returns `FetchPipelineResult` with doc dicts + confidence
  - `run_status_pipeline()` — pure Python, returns `StatusResult`
  - `run_digest_pipeline()` — pure Python, returns `DigestResult`
  - All handlers accept `on_progress: Callable | None` for output delegation
  - CLI commands refactored to call handlers: `tag`, `fetch`, `digest`, `status` all use pipeline handlers
  - `FetchPipelineResult` includes `interpretation_confidence` for CLI low-confidence check
- **S6.4** `corvus/orchestrator/router.py` (new): Deterministic `dispatch()` — confidence gate (< 0.7 → NEEDS_CLARIFICATION), INTERACTIVE_REQUIRED for review/watch, pipeline dispatch for tag/fetch/status/digest, GENERAL_CHAT calls `ollama.chat()`
  - `corvus/integrations/ollama.py`: Added `chat()` method — free-form response without JSON schema, temperature 0.7
- **S6.5** `corvus/cli.py`: Added `corvus ask` (single NL query) + `corvus chat` (interactive REPL):
  - `ask`: classify intent → dispatch → render + interactive fetch selection
  - `chat`: stateless REPL loop, 10m default keep_alive, quit/exit/q to leave
  - Shared `_render_orchestrator_response()` renders all action/result types
  - Shared `_resolve_model()` extracts model auto-detection from both ask/fetch flows

### Test summary
- **47 new tests** across 4 files:
  - `tests/test_intent_classifier.py` (new) — 11 tests: system prompt, mocked LLM for each intent, confidence, keep_alive, schema class, 1 live
  - `tests/test_pipeline_handlers.py` (new) — 13 tests: tag (no docs, queued, auto-applied, errors, no callback), fetch (found, empty, warnings, fallback), status (empty, pending), digest (empty, custom hours)
  - `tests/test_orchestrator_router.py` (new) — 11 tests: confidence gate, threshold boundary, interactive-required (review, watch), tag dispatch, fetch dispatch, fetch no query, status, digest, digest default hours, chat
  - `tests/test_cli.py` (extended) — 13 new tests: ask (fetch, status, clarification, interactive-required, chat response, empty query), chat (quit, process input, skip empty, handle error), help (ask, chat)
- **All 239 tests passing** (192 original + 47 new)

### Current state
- **Epics 1–6:** Complete (archived)
- **All tests passing:** 239 total (235 fast, 4 slow/live)
- **Test breakdown:**
  - `test_cli.py` — 39 (all unit, mocked)
  - `test_intent_classifier.py` — 11 (10 unit + 1 slow live)
  - `test_pipeline_handlers.py` — 13 (all unit)
  - `test_orchestrator_router.py` — 11 (all unit)
  - `test_query_interpreter.py` — 8 (7 unit + 1 slow live)
  - `test_retrieval_router.py` — 39 (all unit)
  - `test_paperless_client.py` — 4
  - `test_ollama_client.py` — 2 (1 slow)
  - `test_document_tagger.py` — 9 (1 slow)
  - `test_tagging_router.py` — 15 (1 slow)
  - `test_review_queue.py` — 19
  - `test_audit_log.py` — 14
  - `test_daily_digest.py` — 11
  - `test_e2e_tagging_pipeline.py` — 1 (slow)
  - `test_hash_store.py` — 10
  - `test_watchdog_transfer.py` — 14
  - `test_watchdog_audit.py` — 10
  - `test_watchdog_cli.py` — 7

### Architecture after Epic 6
```
User: "find my AT&T invoice"
       |
   corvus ask / corvus chat
       |
   Intent Classifier (LLM — corvus/planner/intent_classifier.py)
       |
   IntentClassification {intent: FETCH_DOCUMENT, fetch_query: "AT&T invoice", confidence: 0.95}
       |
   Orchestrator Router (Python — corvus/orchestrator/router.py)
     - Confidence gate (< 0.7 → ask to clarify)
     - Dispatches to pipeline handler
       |
   Pipeline Handler (corvus/orchestrator/pipelines.py)
     - run_fetch_pipeline() → calls interpret_query + resolve_and_search
       |
   CLI renders result, handles interactive selection if needed
```

### Key design decisions
- **Flat IntentClassification model** — all per-intent params as optional fields; intent field determines which are relevant; simple for LLM structured output
- **Pipeline handlers return data, CLI renders** — both `corvus fetch` and `corvus ask "find..."` call the same handler, enabling future voice/web interfaces
- **Confidence gate at 0.7** — below threshold returns NEEDS_CLARIFICATION, no destructive action taken
- **Existing CLI commands preserved** — `corvus tag/fetch/review/watch/digest/status` work exactly as before
- **Chat is stateless** — each turn classified + dispatched independently; no conversation memory for V1
- **`on_progress` callback pattern** — CLI passes `click.echo`, orchestrator can pass `logger.info` or `None`

### Smoke test results
- `corvus status` — works correctly (0 pending, 10 processed, 10 reviewed)
- `corvus digest --hours 5` — works correctly (no activity in window)
- `corvus ask find latest mortgage statement`:
  - Intent classifier: **correct** — `fetch_document` at 95%, `fetch_query="latest mortgage statement"`
  - Query interpreter: **PROBLEM** — returned `confidence=0.90` but `text=None`, no tags, no correspondent, no type. All structured fields empty despite clear query.
  - Result: Paperless searched with `{}` filters → returned all 95 docs unfiltered
  - Root cause: query interpreter LLM (qwen2.5:7b-instruct) failed to populate any structured fields. Needs prompt tuning or few-shot examples.
- `corvus ask tag my documents` — **works correctly**: intent classifier returned `tag_documents` at 90%, tagged all 95 untagged documents (~5 min), all queued for review. Hit pagination bug at end (see fix below).
- `corvus ask what is the status` — works correctly
- `corvus ask what happened today` — works correctly
- `corvus ask hello how are you` — works correctly (general_chat)
- `corvus ask review my pending items` — works correctly (interactive_required → "use corvus review")
- `corvus ask watch my scan folder` — works correctly (interactive_required → "use corvus watch")
- `corvus chat` — works correctly (REPL loop, multiple intents, quit exits)
- First run hit `httpx.RemoteProtocolError` (Paperless server disconnect) — added try/except in `_ask_async` so it shows clean error instead of raw traceback.

### Bug fixes during smoke testing
1. **Error handling in `_ask_async`** — wraps `dispatch()` in try/except, logs exception, shows clean error message instead of traceback
2. **Pagination bug in `run_tag_pipeline`** — after processing the last page, the loop tried to fetch the next page which Paperless returns as 404. Fixed by breaking when `len(docs) < page_size` (last page detected). Also added defensive 404 handling in `PaperlessClient.list_documents()` to return empty list instead of crashing.

### Known issues
1. **Query interpreter empty fields** — LLM sometimes returns high confidence with no search fields populated. Needs prompt improvement (few-shot examples, post-validation). Filed as S7.1 in backlog.
2. **Paperless connection drops** — transient `RemoteProtocolError` from Paperless server. Needs retry logic or graceful handling in pipeline handlers. Filed as S7.2.

### Next steps
- **Epic 7** (backlog_current.md): S7.1 (query interpreter fix), S7.2 (connection retry), S7.3 (full smoke test)
- After Epic 7: Phase 3 (email pipeline), voice I/O (STT/TTS), or web dashboard
- Consider `corvus chat` conversation memory (multi-turn context)

---

## Session: 2026-02-24 (sessions 7–8)

### What was done
- Completed Epic 3: Paperless Document Retrieval (5 stories: S3.1–S3.5)
- `corvus/schemas/document_retrieval.py` (new): `QueryInterpretation`, `ResolvedSearchParams`, `DeliveryMethod` schemas; `used_fallback: bool` field on `ResolvedSearchParams`
- `corvus/executors/query_interpreter.py` (new): Stateless async `interpret_query()` — LLM parses natural language into structured search params; system prompt includes available correspondents/types/tags + today's date
- `corvus/router/retrieval.py` (new): `resolve_search_params()` resolves names to IDs with warnings; `build_filter_params()` constructs Paperless API filter dict; `resolve_and_search()` orchestrates the full flow with three-tier fallback cascade
- `corvus/integrations/paperless.py`: Added `download_document(doc_id, dest_path) -> Path` (streams original file via API) and `get_document_url(doc_id) -> str` (browser URL)
- `corvus/cli.py`: Added `corvus fetch` command — `nargs=-1` query (no quotes needed), `--method browser|download`, `--download-dir`, `--keep-alive`; interactive numbered list for multiple results; low confidence confirmation; unresolved name warnings; fallback notice
- **Fallback search cascade** (session 8 — live-testing revealed LLM-resolved filters can be wrong or over-restrictive):
  - Unresolved entity names folded into text search with user-visible warnings
  - When structured filters return 0 results, three fallback levels tried in order:
    1. **Per-tag**: tries each resolved tag individually (broadens AND → per-tag; picks smallest/most specific result set)
    2. **Text-only**: drops all structured filters, uses full-text `query` param
    3. **Title/content**: uses Paperless `title_content` filter (case-insensitive substring match on title/content — catches what full-text engine misses, works for untagged docs)
  - `used_fallback` flag + CLI notice: "Structured filters returned no results; showing results from relaxed search."
  - Fallback 3 also triggers for pure text queries (outside the structured-filters gate)
- Tests: 60 new tests across 3 files (8 + 39 + 13)
  - `tests/test_query_interpreter.py` (new) — 8 tests: prompt construction, mocked LLM, keep_alive forwarding, 1 live test
  - `tests/test_retrieval_router.py` (new) — 39 tests: name resolution (15), filter param construction (8), search + fallback cascade (16)
  - `tests/test_cli.py` (extended) — 13 new fetch tests: single result, multi-select, no results, low confidence abort/continue, download, quit, no model, many results truncation
- Updated backlog: Epic 3 moved to archive, current backlog cleared

### Current state
- **Epics 1, 2, 3, 4, 5:** Complete (archived)
- **All tests passing:** 192 total (188 fast, 4 slow/live)
- **Test breakdown:**
  - `test_cli.py` — 29 (all unit, mocked)
  - `test_query_interpreter.py` — 8 (7 unit + 1 slow live)
  - `test_retrieval_router.py` — 39 (all unit)
  - `test_paperless_client.py` — 4
  - `test_ollama_client.py` — 2 (1 slow)
  - `test_document_tagger.py` — 9 (1 slow)
  - `test_tagging_router.py` — 15 (1 slow)
  - `test_review_queue.py` — 19
  - `test_audit_log.py` — 14
  - `test_daily_digest.py` — 11
  - `test_e2e_tagging_pipeline.py` — 1 (slow)
  - `test_hash_store.py` — 10
  - `test_watchdog_transfer.py` — 14
  - `test_watchdog_audit.py` — 10
  - `test_watchdog_cli.py` — 7

### Smoke test results
- `corvus fetch the trust transfer document for howard st property`: LLM extracted `document_type=statement` (wrong match) + text. Initial structured search → 0. Text-only fallback → 4 results. Correct doc ("Trust Transfer Howard", id=27) at position 1. ✓
- `corvus fetch documents related to fy2022 taxes`: LLM extracted `tags=[Taxes, FY 2022]` (both resolved) + text. Initial `tags__id__all` (AND) → 0 (too strict). Per-tag fallback tries both tags, picks smallest result set (most specific).

### Key design decisions
- **LLM interprets, Python searches** — same pattern as tagging pipeline
- **No confidence gates** — retrieval is read-only + interactive; user always picks
- **No audit logging** — read-only, not compliance-critical
- **Reuse name resolution** — imports `resolve_tag/correspondent/document_type` from `corvus.router.tagging`
- **Reuse `list_documents()`** — no new Paperless search method; `filter_params={"query": ..., "correspondent__id": X}` already works
- **`nargs=-1`** for query — no quotes needed: `corvus fetch most recent invoice from AT&T`
- **10-result display cap** — shows first 10 of many, suggests refining query
- **Three-tier fallback** — per-tag (smallest result set wins) → text-only → title/content substring. Each level progressively relaxes constraints. `used_fallback` only set when a fallback actually produces results.
- **`title_content`** — Paperless-ngx custom filter: `Q(title__icontains=value) | Q(content__icontains=value)`. Useful for untagged docs or when full-text tokenization misses substring matches.

### Next steps / known improvements
- Consider next work: Phase 2 (tiered architecture), mobile delivery, or Phase 3 (email)

---

## Session: 2026-02-23 (session 6)

### What was done
- Added `[e]dit` option to `corvus review` (S5.10)
- `corvus/router/tagging.py`: `apply_approved_update()` now accepts `extra_tag_names: list[str] | None` — appends them as `TagSuggestion(confidence=1.0)` entries to a copy of the task before routing
- `corvus/queue/review.py`: Added `modify()` method — sets `ReviewStatus.MODIFIED` (mirrors `approve`/`reject`)
- `corvus/cli.py`: Review prompt now `[a]pprove / [e]dit / [r]eject / [s]kip / [q]uit`; pressing `e` prompts for comma-separated tags, resolves via existing pipeline, marks as MODIFIED with reviewer_notes
- `tests/test_cli.py`: 2 new tests — `test_review_edit_adds_tags`, `test_review_edit_empty_tags_approves_normally`
- Moved Epic 4 from `backlog_current.md` to `backlog_archive.md`

### Current state
- **Epics 1, 2, 4, 5:** Complete (archived)
- **Epic 3:** Backlogged (document retrieval)
- **All tests passing:** 134 total (130 fast, 4 slow/live)
- **Test breakdown:**
  - `test_cli.py` — 16 (all unit, mocked)
  - `test_paperless_client.py` — 4
  - `test_ollama_client.py` — 2 (1 slow)
  - `test_document_tagger.py` — 9 (1 slow)
  - `test_tagging_router.py` — 15 (1 slow)
  - `test_review_queue.py` — 19
  - `test_audit_log.py` — 14
  - `test_daily_digest.py` — 11
  - `test_e2e_tagging_pipeline.py` — 1 (slow)
  - `test_hash_store.py` — 11
  - `test_watchdog_transfer.py` — 14
  - `test_watchdog_audit.py` — 13
  - `test_watchdog_cli.py` — 5

### Next steps
- Smoke test `corvus review` edit flow against real Paperless instance
- Smoke test `corvus watch --once` against real scan directory + Paperless
- Consider Epic 3 (document retrieval) or Phase 2 (tiered architecture)
- Consider systemd service for always-on watchdog monitoring

---

## Session: 2026-02-23 (session 5)

### What was done
- Completed Epic 4: Local Scan Folder Watchdog (9 stories: S4.1–S4.9)
- `pyproject.toml`: Added `watchdog>=6.0` dependency
- `corvus/schemas/watchdog.py` (new): `WatchdogEvent`, `TransferMethod`, `TransferStatus` Pydantic models
- `corvus/config.py`: Added 6 watchdog config variables (`WATCHDOG_SCAN_DIR`, `WATCHDOG_TRANSFER_METHOD`, `WATCHDOG_CONSUME_DIR`, `WATCHDOG_FILE_PATTERNS`, `WATCHDOG_AUDIT_LOG_PATH`, `WATCHDOG_HASH_DB_PATH`)
- `corvus/integrations/paperless.py`: Added `upload_document(file_path, *, title=None) -> str` using httpx multipart POST to `/api/documents/post_document/`
- `corvus/watchdog/` (new package):
  - `hash_store.py`: SQLite-backed SHA-256 duplicate tracker with WAL journal mode
  - `transfer.py`: `compute_file_hash()`, `transfer_by_move()` (with name collision handling), `transfer_by_upload()`, `process_file()` (orchestrates hash→dedup→transfer→audit)
  - `audit.py`: Separate JSONL audit log for watchdog events (different shape from tagging audit)
  - `watcher.py`: `ScanFolderHandler` (FileSystemEventHandler) with `on_closed`/`on_created`, debounce, pattern matching; `scan_existing()` for --once mode; `watch_folder()` for continuous monitoring
- `corvus/cli.py`: Added `corvus watch` command with `--scan-dir`, `--method`, `--consume-dir`, `--patterns`, `--once` options + `_validate_watchdog_config()`
- Tests (4 new files, 43 new tests):
  - `test_hash_store.py` — 11 tests (CRUD, persistence, dedup)
  - `test_watchdog_transfer.py` — 14 tests (hashing, move, upload, process_file with both methods, dedup, error handling)
  - `test_watchdog_audit.py` — 13 tests (JSONL read/write, filtering by since/status/limit, edge cases)
  - `test_watchdog_cli.py` — 5 tests (help, validation, --once with real file operations, dedup on second run)

### Current state
- **Epic 1:** Complete
- **Epic 2:** Complete
- **Epic 4:** Complete (scan folder watchdog)
- **Epic 5:** Complete (CLI entry point)
- **Epic 3:** Backlogged
- **All tests passing:** 128 total (124 fast, 4 slow/live)
- **Test breakdown:**
  - `test_cli.py` — 14 (all unit, mocked)
  - `test_paperless_client.py` — 4
  - `test_ollama_client.py` — 2 (1 slow)
  - `test_document_tagger.py` — 9 (1 slow)
  - `test_tagging_router.py` — 15 (1 slow)
  - `test_review_queue.py` — 19
  - `test_audit_log.py` — 14
  - `test_daily_digest.py` — 11
  - `test_e2e_tagging_pipeline.py` — 1 (slow)
  - `test_hash_store.py` — 11
  - `test_watchdog_transfer.py` — 14
  - `test_watchdog_audit.py` — 13
  - `test_watchdog_cli.py` — 5

### Key design decisions
- Separate JSONL audit log for watchdog events (different shape from tagging audit)
- `on_closed` + `on_created` as dual file detection triggers, with 2s debounce
- SHA-256 hash-based dedup in SQLite prevents re-upload on restart
- `--once` mode scans existing files and exits (useful for catch-up)
- Observer runs in background thread, schedules async `process_file` on event loop
- `transfer_by_move` appends `_1`, `_2`, etc. on name collision
- Paperless upload returns task UUID as plain text (`response.text.strip()`)

### Next steps
- Smoke test `corvus watch --once` against real scan directory + Paperless
- Consider Epic 3 (document retrieval) or Phase 2 (tiered architecture)
- Consider systemd service for always-on watchdog monitoring

---

## Session: 2026-02-23 (session 4)

### What was done
- Completed Epic 5: CLI Entry Point (9 stories)
- `pyproject.toml`: Added `click>=8.0` dependency and `[project.scripts] corvus = "corvus.cli:cli"` entry point
- `corvus/integrations/ollama.py`: Extracted shared `pick_instruct_model()` function (+ `OllamaClient.pick_instruct_model()` method); removed duplicated copies from 3 test files
- `corvus/integrations/paperless.py`: Added `filter_params` kwarg to `list_documents()` — enables `{"tags__isnull": True}` for fetching untagged docs
- `corvus/router/tagging.py`: Added `force_apply` param to `resolve_and_route()` (overrides gate to always apply); added `apply_approved_update()` for re-fetching current Paperless state and applying after human approval
- `corvus/cli.py` (new): Click-based CLI with 4 commands:
  - `corvus tag` — batch-tag with `--limit`, `--model` (auto-detected), `--all`, `--keep-alive`, `--force-queue`; sequential page-by-page processing; per-doc error handling
  - `corvus review` — interactive review: shows details, prompts [a]pprove/[r]eject/[s]kip/[q]uit; approve calls `apply_approved_update()` to write to Paperless
  - `corvus digest` — calls existing `generate_digest()` + `render_text()`
  - `corvus status` — pending count + 24h processed/reviewed counts
- `tests/test_cli.py` (new): 14 tests using Click CliRunner with mocked Paperless/Ollama
- Fixed infinite loop bug: errors now count toward `--limit` to prevent re-fetching the same failing doc

### Current state
- **Epic 1:** Complete
- **Epic 2:** Complete
- **Epic 5:** Complete (CLI entry point)
- **Epics 3 & 4:** Backlogged
- **All tests passing:** 89 total (85 fast, 4 slow/live)
- **Test breakdown:**
  - `test_cli.py` — 14 (all unit, mocked)
  - `test_paperless_client.py` — 4
  - `test_ollama_client.py` — 2 (1 slow)
  - `test_document_tagger.py` — 9 (1 slow)
  - `test_tagging_router.py` — 15 (1 slow)
  - `test_review_queue.py` — 19
  - `test_audit_log.py` — 14
  - `test_daily_digest.py` — 11
  - `test_e2e_tagging_pipeline.py` — 1 (slow)

### Smoke test results
- All 4 commands verified against live Paperless + Ollama
- `corvus tag --limit 1` classified PHH Mortgage Escrow doc (id=94) in ~4s, 100% confidence, queued for review
- `corvus review` approved it — created 2 new tags (escrow-account, statement), 1 new correspondent (PH Mortgage Services), PATCHed document
- `corvus digest` showed full activity trail
- 94 untagged documents remaining in Paperless

### Next steps
- Batch-tag remaining 94 untagged documents: `corvus tag` (or in smaller batches)
- Consider next work: Epic 3 (document retrieval), Epic 4 (scan watchdog), or Phase 2 (tiered architecture)

---

## Session: 2026-02-23 (session 3)

### What was done
- Completed S2.4: Document tagger executor (`corvus/executors/document_tagger.py`)
  - `tag_document()` — stateless async function: takes document + Ollama client + existing Paperless metadata, classifies via LLM, returns `DocumentTaggingTask` + `OllamaResponse`
  - System prompt includes existing tags/correspondents/doc types for reuse
  - Content truncated to 8K chars; snippet (500 chars) stored on task for logging
  - `_compute_overall_confidence()` — weighted average across all suggestions
  - `_determine_gate_action()` — maps confidence to gate thresholds from CLAUDE.md
  - Tests: 8 unit + 1 live integration (9 total)
  - Issue: first model picked (`mythomax` RP model) produced garbage; added `_pick_instruct_model()` to prefer instruct/chat models
- Completed S2.5: Confidence gate logic in router (`corvus/router/tagging.py`)
  - `resolve_and_route()` — deterministic Python (no LLM): resolves tag/correspondent/doc-type names → Paperless IDs (case-insensitive), builds `ProposedDocumentUpdate`, applies or queues based on gate
  - `force_queue=True` (default) — initial posture per CLAUDE.md, overrides gate to always queue
  - When applying: creates missing entities in Paperless first, then PATCHes document (merges existing + new tags)
  - When queuing: resolves what it can, doesn't create anything
  - `RoutingResult` model with proposed_update, applied flag, effective_action
  - Tests: 14 unit + 1 live integration (15 total)
- Completed S2.6: Review queue (`corvus/queue/review.py`)
  - SQLite-backed, WAL journal mode, stdlib sqlite3 (no new dependency)
  - `add()`, `get()`, `list_pending()`, `list_all()`, `approve()`, `reject()`, `count_pending()`
  - Tasks/proposed updates stored as JSON, full Pydantic roundtrip on read
  - Status validation: can't approve/reject already-processed items
  - Tests: 19 unit tests using tmp_path
- Completed S2.7: Audit logging (`corvus/audit/logger.py`)
  - JSONL append-only log (one JSON object per line)
  - Convenience methods: `log_auto_applied()`, `log_queued_for_review()`, `log_review_approved()`, `log_review_rejected()`
  - `read_entries(since=, limit=)` for digest consumption
  - Tests: 14 unit tests using tmp_path
- Completed S2.8: Daily digest (`corvus/digest/daily.py`)
  - `generate_digest(audit_log, review_queue, since=, hours=24)` — groups entries by action type
  - `DailyDigest` model: auto_applied, flagged, queued_for_review, review_approved, review_rejected, pending_review_count
  - `render_text()` — markdown output with summary counts and item details
  - Flagged items (0.7-0.9 confidence) separated from clean auto-applied (>0.9)
  - Tests: 11 unit tests
- Completed S2.9: End-to-end integration test (`tests/test_e2e_tagging_pipeline.py`)
  - Full pipeline: Paperless fetch → LLM classify → route (force_queue) → review queue → audit log → approve → digest
  - Verifies all components wire together correctly against live services

### Current state
- **Epic 1:** Complete (all 6 stories)
- **Epic 2:** Complete (all 9 stories: S2.1–S2.9)
- **Epics 3 & 4:** Backlogged (post-MVP)
- **All tests passing:** 75 total (71 fast, 4 slow/live)
- **Test breakdown:**
  - `test_paperless_client.py` — 4 (Paperless API)
  - `test_ollama_client.py` — 2 (Ollama API, 1 slow)
  - `test_document_tagger.py` — 9 (8 unit + 1 slow live)
  - `test_tagging_router.py` — 15 (14 unit + 1 slow live)
  - `test_review_queue.py` — 19 (SQLite, all unit)
  - `test_audit_log.py` — 14 (JSONL, all unit)
  - `test_daily_digest.py` — 11 (all unit)
  - `test_e2e_tagging_pipeline.py` — 1 (slow, full pipeline)
- **Paperless instance:** `http://athena.eagle-mimosa.ts.net:8000`
- **Ollama:** `qwen2.5:7b-instruct` preferred for classification (also has `mythomax` but unsuitable)

### Decisions made
- Executor is stateless: receives all inputs, returns result, no side effects
- Router `force_queue=True` is default (initial posture — all queued until trust established)
- Review queue uses stdlib sqlite3 (no aiosqlite dependency — fast local I/O)
- Audit log uses JSONL format (append-only, grep-friendly)
- Instruct/chat models preferred over creative/RP models for classification
- New Paperless entities (tags, correspondents, doc types) only created when actually applying (not when queuing)

### Blockers
- None

### Next steps
- Epic 2 is complete. Consider what to tackle next:
  - Wire up a CLI or cron entry point to run the pipeline on untagged documents
  - Build a simple review UI (CLI-based to start?)
  - Start on Epic 3 (document retrieval) or Epic 4 (scan folder watchdog)
  - Or move to Phase 2 (tiered planner/executor architecture refactor)
- The pipeline components are all built and tested but lack a top-level orchestrator to invoke them as a batch job
