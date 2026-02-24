# Corvus — Session Continuity Notes

> Updated at the end of each session or before conversation compacting. Read at the start of each new session.

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
