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

## Epic 3: Paperless Document Retrieval (PC-only)

**Goal:** Retrieve documents from Paperless-ngx via natural language queries from the local PC. Corvus interprets the request via LLM, searches Paperless, and delivers the result (browser or download).

- [x] **S3.1** Pydantic schemas (`corvus/schemas/document_retrieval.py`) — QueryInterpretation, ResolvedSearchParams, DeliveryMethod
- [x] **S3.2** Query interpreter executor (`corvus/executors/query_interpreter.py`) — LLM parses natural language into structured search params
- [x] **S3.3** Retrieval router (`corvus/router/retrieval.py`) + `download_document()` + `get_document_url()` on PaperlessClient — resolves names to IDs, builds filter params, searches via existing `list_documents()`
- [x] **S3.4** CLI `corvus fetch` command — interpret → search → interactive selection → deliver (browser or download)
- [x] **S3.5** Tests — 8 query interpreter tests, 24 retrieval router tests, 13 CLI fetch tests (45 new total)

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

---

## Epic 6: Orchestrator Architecture (Phase 2)

**Goal:** Introduce the Planner/Orchestrator tier — an LLM intent classifier + deterministic dispatch router that enables natural language intake across all pipelines. Foundation for future voice, email, and web interfaces.

- [x] **S6.1** Orchestrator schemas (`corvus/schemas/orchestrator.py`) — Intent enum (7 intents), IntentClassification (flat model with per-intent params), pipeline result models (TagPipelineResult, FetchPipelineResult, StatusResult, DigestResult), OrchestratorAction, OrchestratorResponse
- [x] **S6.2** Intent classifier executor (`corvus/planner/intent_classifier.py`) — stateless async `classify_intent()`, calls Ollama with JSON schema constraint, returns (IntentClassification, OllamaResponse); 11 tests
- [x] **S6.3** Pipeline handlers + CLI refactor (`corvus/orchestrator/pipelines.py`) — extracted `run_tag_pipeline()`, `run_fetch_pipeline()`, `run_status_pipeline()`, `run_digest_pipeline()` from CLI; CLI commands now call handlers with `on_progress=click.echo`; 13 tests
- [x] **S6.4** Orchestrator router + `OllamaClient.chat()` (`corvus/orchestrator/router.py`, `corvus/integrations/ollama.py`) — deterministic `dispatch()` function: confidence gate (< 0.7 → clarify), interactive-required for review/watch, pipeline dispatch for tag/fetch/status/digest, free-form chat for general conversation; 11 tests
- [x] **S6.5** CLI: `corvus ask` + `corvus chat` commands — `ask` for single NL query (classify → dispatch → render), `chat` for interactive REPL (stateless per turn); shared `_render_orchestrator_response()` helper; 13 new CLI tests

---

## Epic 7: Search Reliability & Smoke Testing

**Goal:** Fix query interpreter reliability issues discovered during Epic 6 smoke testing. The LLM sometimes returns high confidence but empty search fields, causing Paperless to return all documents unfiltered.

- [x] **S7.1** Improve query interpreter field extraction — added `_has_search_fields()` post-validation: when LLM returns high confidence but no search fields populated, inject original query as `text_search` fallback.
- [x] **S7.2** Fix "most recent" date misinterpretation + richer result display — LLM was setting `date_range_start=today, date_range_end=today` for "most recent" queries. Added Rule 11 to prompt ("do NOT set date ranges for latest/newest"), explicit null dates in few-shot examples, and `_strip_today_only_date_range()` deterministic guardrail. Also enriched fetch results with correspondent, document type, and tags (resolved from already-fetched metadata) and added two-line display format in CLI via `_format_doc_line()` helper.
- [x] **S7.3** Complete ask/chat test coverage — added 7 tests covering all missing intent paths: TAG_DOCUMENTS, SHOW_DIGEST, FETCH_DOCUMENT (0 results), FETCH_DOCUMENT (multi-select), WATCH_FOLDER, chat fetch inline, dispatch error (RemoteProtocolError). Total `test_cli.py` tests: 48.
- [x] **S7.4** Chat model recommendation — **Recommend `qwen2.5:14b-instruct`** for GENERAL_CHAT. Key findings: 14B (Q4_K_M, ~10-11 GB VRAM) offers meaningful conversation quality uplift over 7B for free-form chat while fitting easily in 24 GB. Keep `qwen2.5:7b-instruct` for all structured output tasks. Implementation: add `CHAT_MODEL` config variable, use in `_dispatch_chat()`. No code changes in this story.
- [x] **S7.5** Paperless connection drop handling — added `_retry_on_disconnect()` in `paperless.py` (MAX_RETRIES=2, 1s delay) wrapping `_get`, `_patch`, `list_documents`, `upload_document`, `download_document`. Added try/except for `RemoteProtocolError` and `HTTPStatusError` in CLI. 4 new tests.

---

## Epic 8: Web Search Intent

**Goal:** Add a WEB_SEARCH intent so `corvus ask`/`corvus chat` can answer questions requiring current/external knowledge by searching DuckDuckGo and summarizing results with source citations.

- [x] **S8.1** Add `ddgs>=9.0` dependency (formerly `duckduckgo-search`, renamed upstream)
- [x] **S8.2** Schema updates — `WEB_SEARCH` intent, `search_query` param, `WebSearchSource` + `WebSearchResult` models, union update
- [x] **S8.3** Search integration module — `corvus/integrations/search.py` with async DDG wrapper, `SearchResult`, `SearchError`
- [x] **S8.4** Config — `WEB_SEARCH_MAX_RESULTS` (default 5)
- [x] **S8.5** Intent classifier prompt — web_search as intent #7 with examples, rules for when to classify as web_search vs general_chat
- [x] **S8.6** Search pipeline — `run_search_pipeline()` (DDG search → LLM summarization with citations) + `_search_fallback_chat()` (LLM-only with disclaimer)
- [x] **S8.7** Router dispatch — `_dispatch_search()` with search_query fallback to user_input
- [x] **S8.8** CLI rendering — summary + numbered sources
- [x] **S8.9** Tests — 14 new tests: 5 search integration, 4 pipeline, 2 router, 1 classifier, 2 CLI

---

## Epic 9: CHAT_MODEL Config

**Goal:** Allow `corvus chat` and web search summarization to use a separate (larger) model for free-form text generation while keeping the default model for structured output.

- [x] **S9.1** Config variable — `CHAT_MODEL` in `corvus/config.py` (empty string = same model for everything)
- [x] **S9.2** Thread `chat_model` through router — `dispatch()` gets `chat_model` param, computes `effective_chat_model`, passes to `_dispatch_chat()` and `_dispatch_search()`
- [x] **S9.3** Thread `chat_model` through CLI — `_ask_async()` and `_chat_async()` resolve and pass `CHAT_MODEL`, echo when set
- [x] **S9.4** Tests — 5 new: 4 router (chat/search use chat model, default fallback, tag ignores), 1 CLI (display)

---

## Epic 10: Conversation Memory

**Goal:** Add in-memory conversation history to `corvus chat` so multi-turn references ("the first one", "download it") work. Lost on exit (V1).

- [x] **S10.1** `ConversationHistory` helper — `corvus/orchestrator/history.py` with add/get messages, `get_recent_context()`, `summarize_response()` for all intent types
- [x] **S10.2** Extend `ollama.chat()` — `messages` param inserted between system prompt and current user message
- [x] **S10.3** Thread `conversation_history` through dispatch — `dispatch()` accepts and forwards to `_dispatch_chat()`
- [x] **S10.4** Conversation context for intent classifier — `conversation_context` param, context-aware prompt template, rule 8 for reference resolution
- [x] **S10.5** Wire into `_chat_async()` — `ConversationHistory` instance in REPL loop, context passed to classifier, history passed to dispatch, responses summarized for history
- [x] **S10.6** Tests — 23 new: 14 conversation history, 2 ollama client, 2 intent classifier, 6 router, 4 CLI (history wiring + ask stateless)
- [x] **S10.7** Interactive fetch in chat mode — single result auto-opens in browser, multiple results offer `[1-N, s to skip]` selection prompt (previously chat showed results inline with no delivery). 3 new tests (select, skip, single auto-open), replacing 1 old inline-only test.

---

## Epic 11: Web Search Page Content Fetching

**Goal:** Improve web search answer quality by fetching actual page content from top search results, extracting readable text via trafilatura, and including it in the LLM summarization prompt alongside snippets.

- [x] **S11.1** Add dependency — `trafilatura>=1.6` for HTML-to-text extraction
- [x] **S11.2** Page fetcher — `_fetch_single_page()` + `fetch_page_content()` in `corvus/integrations/search.py`
- [x] **S11.3** Config — `WEB_SEARCH_FETCH_PAGES` (2), `WEB_SEARCH_PAGE_MAX_CHARS` (8000), `WEB_SEARCH_FETCH_TIMEOUT` (10)
- [x] **S11.4** Integrate into search pipeline — fetch pages after DDG search, include in LLM context, updated system prompt
- [x] **S11.5** Tests — 15 new tests across 3 test files (320 total pass)
- [x] **S11.6** Smoke test — confirmed richer answers with actual data from page content
