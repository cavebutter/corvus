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

---

## Epic 12: Conversation Memory Persistence (V2)

**Goal:** Persist `corvus chat` conversation history to SQLite so sessions can be resumed across restarts.

- [x] **S12.1** SQLite schema — conversations table (id, title, created_at, updated_at), messages table (conversation_id, role, content, created_at), index on conversation_id
- [x] **S12.2** Persistence layer — `ConversationStore` class wrapping SQLite (create, add_message, load_messages, get_most_recent, list_conversations, get_conversation with prefix matching)
- [x] **S12.3** Config — `CONVERSATION_DB_PATH` in `corvus/config.py`
- [x] **S12.4** Extend `ConversationHistory` — optional persistence (store + conversation_id), `from_store()` classmethod, `set_persistence()` for deferred creation, `conversation_id` property
- [x] **S12.5** CLI wiring — `--new`, `--list`, `--resume <id>` on chat; default resumes most recent; deferred conversation creation on first message; `_list_conversations()` helper
- [x] **S12.6** Tests — 26 store tests, 6 history persistence tests, 6 CLI tests (38 new, 358 total pass)
- [x] **S12.7** Smoke test — `--list` (empty + populated), `--new` (creates on first msg, not on quit), default resume, `--resume <prefix>`, `--resume bad-id` error, web search in persisted session, message counts update correctly

---

## Epic 13: Voice Engine Foundation

**Goal:** Build the core audio, STT, TTS, and wake word detection modules for voice I/O. All modules are async-ready with `asyncio.to_thread()` bridges for sync libraries.

- [x] **S13.1** Dependencies, config, schemas, and package init
- [x] **S13.2** Audio I/O module (`corvus/voice/audio.py`)
- [x] **S13.3** STT module (`corvus/voice/stt.py`)
- [x] **S13.4** TTS module (`corvus/voice/tts.py`)
- [x] **S13.5** Wake word module (`corvus/voice/wakeword.py`)
- [x] **S13.6** Tests for engine components (52 tests)

---

## Epic 14: Local Voice Assistant

**Goal:** Wire up the voice engine into a full assistant pipeline (wake word → STT → orchestrator → TTS → playback) with a CLI command.

- [x] **S14.1** Voice pipeline orchestration (`corvus/voice/pipeline.py`)
- [x] **S14.2** CLI command `corvus voice`
- [x] **S14.3** Pipeline and CLI tests (23 tests)
- [x] **S14.4** Smoke test on hardware (TTS, STT, full loop verified)

---

## Epic 16: Email Engine Foundation (Phase 3)

**Goal:** Build core email infrastructure — IMAP client, LLM classification/extraction executors, and all Pydantic schemas for the email triage pipeline.

- [x] **S16.1** Dependencies, config, schemas — added `imap-tools>=1.7` to pyproject.toml, EMAIL_* config vars to `corvus/config.py`, created `corvus/schemas/email.py` with 16 models/enums (EmailAccountConfig, EmailEnvelope, EmailMessage, EmailCategory, EmailClassification, EmailAction, EmailTriageTask, InvoiceData, ActionItem, EmailExtractionResult, EmailTriageResult, EmailSummaryResult, EmailReviewQueueItem, EmailAuditEntry)
- [x] **S16.2** IMAP client (`corvus/integrations/imap.py`) — async wrapper around imap-tools using `asyncio.to_thread()`, Gmail COPY+DELETE move support via `is_gmail` flag, folder auto-creation, fetch envelopes (headers-only) and full messages
- [x] **S16.3** Email classifier executor (`corvus/executors/email_classifier.py`) — LLM classification into 9 categories with confidence scoring, category→action mapping (SPAM→DELETE, RECEIPT→MOVE receipts, etc.), confidence gate thresholds (0.9/0.7)
- [x] **S16.4** Email extractor executor (`corvus/executors/email_extractor.py`) — LLM extraction for RECEIPT/INVOICE/ACTION_REQUIRED emails, returns InvoiceData + ActionItem + key dates
- [x] **S16.5** Engine tests — 82 tests across 4 files (schemas 39, imap 17, classifier 19, extractor 7)

---

## Epic 17: Email Triage Pipeline (Phase 3)

**Goal:** Wire the email engine into a full pipeline with review queue, audit log, confidence-gated routing, orchestrator integration, and CLI commands.

- [x] **S17.1** Email review queue + audit log — `corvus/queue/email_review.py` (SQLite, separate table), `corvus/audit/email_logger.py` (separate JSONL file)
- [x] **S17.2** Email router (`corvus/router/email.py`) — confidence gate routing with `force_queue` override, `execute_email_action()` for IMAP operations
- [x] **S17.3** Email triage pipeline (`corvus/orchestrator/email_pipelines.py`) — `run_email_triage()` (fetch → classify → extract → route) and `run_email_summary()` (classify → extract action items → LLM summary)
- [x] **S17.4** Orchestrator integration — EMAIL_TRIAGE + EMAIL_SUMMARY intents in schemas, classifier prompt, router dispatch with `email_account` param
- [x] **S17.5** CLI commands (`corvus email` group) — `triage`, `review`, `summary`, `status`, `accounts` subcommands
- [x] **S17.6** Pipeline and CLI tests — 19 tests across 3 files (router 11, pipeline 11, CLI 8, minus overlap = 30 new)

---

## Epic 18: Email Intelligence (Phase 3)

**Goal:** Add intelligence features on top of the email pipeline — interactive review, inbox summarization, daily digest integration, and voice assistant support.

- [x] **S18.1** Interactive email review CLI — `corvus email review` with approve/reject/skip/quit per queued action, IMAP execution on approve
- [x] **S18.2** Inbox summarization via LLM — `run_email_summary()` classifies all unread, extracts action items, generates natural language summary
- [x] **S18.3** Daily digest integration — added email stats fields to `DailyDigest` model, `generate_digest()` accepts optional email audit/queue, `render_text()` includes email section
- [x] **S18.4** Voice integration — email result handling in `_response_to_speech()` and `summarize_response()` for both EmailTriageResult and EmailSummaryResult

---

## Epic 19: Email Sender Lists and Rules

**Goal:** Deterministic, user-defined handling of known email senders — bypassing LLM classification for blacklisted/vendor/headhunter senders, while ensuring whitelisted humans are always featured in summaries. Follows "deterministic over probabilistic" principle.

- [x] **S19.1** Schemas, core manager, config — `corvus/schemas/sender_lists.py` (SenderListConfig, SenderListsFile, SenderMatch), `corvus/sender_lists.py` (SenderListManager with O(1) lookup, add/remove, rationalize, atomic JSON save, `build_task_from_sender_match()`), `EMAIL_SENDER_LISTS_PATH` config, 31 tests
- [x] **S19.2** Triage pipeline integration — sender list check before LLM in `run_email_triage()`, non-white lists skip LLM and execute immediately (bypass `force_queue`), white list still classifies but forces KEEP + QUEUE_FOR_REVIEW, `sender_list` field on EmailTriageTask, `sender_list_applied` audit action, 7 tests
- [x] **S19.3** Summary pipeline integration — whitelisted senders always added to `important_subjects` and always get action item extraction in `run_email_summary()`, 3 tests
- [x] **S19.4** CLI list management commands — `corvus email lists` (display all), `list-add` (add sender), `list-remove` (remove sender), `rationalize` (dedup + resolve conflicts), passed `sender_lists_path` through triage/summary CLI commands, 8 tests
- [x] **S19.5** Review integration — `[l]ist` option during review prompt (show lists, pick one, add sender, apply action), `[m]ove` option to pick an IMAP folder (lazy-loaded, cached per account), auto-apply pending items whose senders are on a non-white list at review start
- [x] **S19.6** Folder cleanup — `fetch_uids_older_than()` on ImapClient (server-side IMAP BEFORE search), `corvus email cleanup [--account] [--dry-run]` CLI command to delete messages older than `cleanup_days` from folders with retention policies, 4 tests
