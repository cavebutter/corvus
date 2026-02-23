# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

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
- [ ] **S2.4** Implement document tagger executor (`corvus/executors/document_tagger.py`)
- [ ] **S2.5** Implement confidence gate logic in router
- [ ] **S2.6** Implement review queue (SQLite-based)
- [ ] **S2.7** Implement audit logging
- [ ] **S2.8** Implement daily digest generation
- [ ] **S2.9** End-to-end integration test with Paperless sandbox

---

## Epic 3: Paperless Document Retrieval (Post-MVP)

**Goal:** Allow users to retrieve documents from Paperless-ngx via natural language queries. Corvus interprets the request, searches Paperless, and delivers the result — opening a browser tab on the local machine or downloading to the requesting device.

**Example:** *"Fetch me the most recent invoice from AT&T"*

- [ ] **S3.1** Define Pydantic schemas for document retrieval requests and results
- [ ] **S3.2** Build LLM-based query interpreter — extract intent (correspondent, doc type, date range, etc.) from natural language
- [ ] **S3.3** Implement Paperless search/filter logic using extracted query fields (`GET /api/documents/?query=...`)
- [ ] **S3.4** Implement result ranking and selection when multiple documents match
- [ ] **S3.5** Implement document delivery — local machine: open browser tab to Paperless PDF view; remote device: download via API
- [ ] **S3.6** Handle ambiguous queries — ask user for clarification when confidence is low or multiple strong matches exist

**Dependencies:** Requires Epic 2 (Paperless API client, Ollama client, schemas foundation).

---

## Epic 4: Local Scan Folder Watchdog (Post-MVP)

**Goal:** Monitor a local folder on the main PC for new scanned documents and automatically move them to the Paperless-ngx intake (consume) directory. Acts as a bridge when the scanner cannot target the Paperless intake path directly — similar to Hazel on macOS.

**Example workflow:** Scanner deposits PDF in `~/Scans/` → watchdog detects new file → moves file to Paperless consume directory (network share or API upload) → optionally notifies user.

- [ ] **S4.1** Define config schema for watchdog (watch path, destination path/method, file patterns, polling vs. inotify)
- [ ] **S4.2** Implement filesystem watcher using `watchdog` library (inotify on Linux)
- [ ] **S4.3** Implement file transfer — direct move (if NAS share is mounted) or upload via Paperless `POST /api/documents/post_document/`
- [ ] **S4.4** Handle edge cases — partial writes (wait for file stability), duplicate detection, unsupported file types
- [ ] **S4.5** Add audit logging for all file movements
- [ ] **S4.6** Optional: run as a systemd service for always-on monitoring

**Notes:**
- Transfer method depends on whether the Paperless consume dir is accessible as a mount. API upload is the more portable option.
- Should be lightweight enough to run alongside the main Corvus pipeline without resource contention.

---

*New epics will be added as phases progress. See `backlog_archive.md` for completed work.*
