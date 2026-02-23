# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 5: CLI Entry Point (Ready to Archive)

**Goal:** Provide a CLI to run the tagging pipeline, review pending items, and view digests — the minimum needed to start using Corvus on real Paperless documents.

- [x] **S5.1** Add `click` dependency and `corvus` entry point to `pyproject.toml`
- [x] **S5.2** Extract shared `pick_instruct_model()` into `corvus/integrations/ollama.py`
- [x] **S5.3** Add `filter_params` support to `PaperlessClient.list_documents()`
- [x] **S5.4** Add `force_apply` param and `apply_approved_update()` to router
- [x] **S5.5** Implement `corvus tag` — batch-tag documents with progress output
- [x] **S5.6** Implement `corvus review` — interactive approve/reject/skip/quit
- [x] **S5.7** Implement `corvus digest` — show activity summary
- [x] **S5.8** Implement `corvus status` — quick queue/activity overview
- [x] **S5.9** CLI tests (14 tests using Click CliRunner with mocked services)

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
