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

- [ ] **S2.1** Define Pydantic schemas for document tagging pipeline (`corvus/schemas/`)
- [ ] **S2.2** Build async Paperless-ngx API client (`corvus/integrations/paperless.py`)
- [ ] **S2.3** Build Ollama LLM client for structured output (`corvus/integrations/ollama.py`)
- [ ] **S2.4** Implement document tagger executor (`corvus/executors/document_tagger.py`)
- [ ] **S2.5** Implement confidence gate logic in router
- [ ] **S2.6** Implement review queue (SQLite-based)
- [ ] **S2.7** Implement audit logging
- [ ] **S2.8** Implement daily digest generation
- [ ] **S2.9** End-to-end integration test with Paperless sandbox

---

*New epics will be added as phases progress. See `backlog_archive.md` for completed work.*
