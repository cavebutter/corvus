# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

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

*New epics will be added as phases progress. See `backlog_archive.md` for completed work.*
