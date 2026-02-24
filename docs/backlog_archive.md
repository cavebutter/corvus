# Corvus — Backlog Archive

> Completed epics and stories are moved here from `backlog_current.md` for reference.

---

## Epic 1: Project Scaffolding & Infrastructure

**Goal:** Establish the project skeleton, dependencies, config loading, and dev tooling so that Phase 1 work has a foundation to build on.

- [x] **S1.1** Create project directory structure per CLAUDE.md spec
- [x] **S1.2** Create `pyproject.toml` with core dependencies (pydantic, httpx, ruff, pytest)
- [x] **S1.3** Implement `corvus/secrets.py` (SOPS decrypt helper) and `corvus/config.py` (config loader)
- [x] **S1.4** Set up `.gitignore` with full rules from CLAUDE.md (secrets, venv, IDE, etc.)
- [x] **S1.5** Create pre-commit hook to block unencrypted secrets
- [x] **S1.6** Set up basic test scaffolding (`tests/` dir, pytest config, conftest.py)

---

## Epic 2: Paperless-ngx Document Tagging (Phase 1)

**Goal:** Automated document classification and tagging via Paperless-ngx REST API with confidence scoring and human review queue.

- [x] **S2.1** Define Pydantic schemas for document tagging pipeline (`corvus/schemas/`)
- [x] **S2.2** Build async Paperless-ngx API client (`corvus/integrations/paperless.py`)
- [x] **S2.3** Build Ollama LLM client for structured output (`corvus/integrations/ollama.py`)
- [x] **S2.4** Implement document tagger executor (`corvus/executors/document_tagger.py`)
- [x] **S2.5** Implement confidence gate logic in router (`corvus/router/tagging.py`)
- [x] **S2.6** Implement review queue (SQLite-based) (`corvus/queue/review.py`)
- [x] **S2.7** Implement audit logging (`corvus/audit/logger.py`)
- [x] **S2.8** Implement daily digest generation (`corvus/digest/daily.py`)
- [x] **S2.9** End-to-end integration test (`tests/test_e2e_tagging_pipeline.py`)

---

## Epic 5: CLI Entry Point

**Goal:** Provide a CLI to run the tagging pipeline, review pending items, and view digests.

- [x] **S5.1** Add `click` dependency and `corvus` entry point to `pyproject.toml`
- [x] **S5.2** Extract shared `pick_instruct_model()` into `corvus/integrations/ollama.py`
- [x] **S5.3** Add `filter_params` support to `PaperlessClient.list_documents()`
- [x] **S5.4** Add `force_apply` param and `apply_approved_update()` to router
- [x] **S5.5** Implement `corvus tag` — batch-tag documents with progress output
- [x] **S5.6** Implement `corvus review` — interactive approve/reject/skip/quit
- [x] **S5.7** Implement `corvus digest` — show activity summary
- [x] **S5.8** Implement `corvus status` — quick queue/activity overview
- [x] **S5.9** CLI tests (14 tests using Click CliRunner with mocked services)
- [x] **S5.10** Add `[e]dit` option to `corvus review` — user can add extra tags before approving; uses `ReviewStatus.MODIFIED`; 2 new tests

---

## Epic 4: Local Scan Folder Watchdog

**Goal:** Monitor a local folder on the main PC for new scanned documents and automatically move them to the Paperless-ngx intake (consume) directory.

- [x] **S4.1** Pydantic schemas (`corvus/schemas/watchdog.py`) — WatchdogEvent, TransferMethod, TransferStatus
- [x] **S4.2** Config variables in `corvus/config.py` + `secrets/internal.env`
- [x] **S4.3** `upload_document()` on PaperlessClient (httpx multipart POST)
- [x] **S4.4** SHA-256 hash store for duplicate detection (`corvus/watchdog/hash_store.py`)
- [x] **S4.5** File transfer logic — hash, dedup, move/upload (`corvus/watchdog/transfer.py`)
- [x] **S4.6** Watchdog audit logger — JSONL (`corvus/watchdog/audit.py`)
- [x] **S4.7** Filesystem watcher using `watchdog` library (`corvus/watchdog/watcher.py`)
- [x] **S4.8** CLI command `corvus watch` with `--once` flag
- [x] **S4.9** Tests — 43 new tests (hash store, transfer, audit, CLI)
