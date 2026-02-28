# Corvus — Session Continuity Notes

> Updated at the end of each session or before conversation compacting. Read at the start of each new session.

---

## Session: 2026-02-27 (session 19) — Email Sender Lists and Rules

### What was done
- **Complete Epic 19: Email Sender Lists and Rules** — deterministic sender-based handling before LLM classification
  - Created `corvus/schemas/sender_lists.py` — Pydantic models (SenderListConfig, SenderListsFile, SenderMatch)
  - Created `corvus/sender_lists.py` — SenderListManager with O(1) lookup, add/remove, rationalize, atomic JSON save, `build_task_from_sender_match()`
  - Added `EMAIL_SENDER_LISTS_PATH` config to `corvus/config.py`
  - Added `sender_list: str | None` field to `EmailTriageTask` schema
  - Added `"sender_list_applied"` audit action to `EmailAuditEntry`
  - Added `log_sender_list_applied()` to email audit logger
  - Modified `run_email_triage()` — checks sender lists before LLM; white list still classifies but forces KEEP; black/vendor/headhunter skip LLM and execute directly (bypass force_queue)
  - Modified `run_email_summary()` — whitelisted senders always in important_subjects and always get extraction
  - Added `fetch_uids_older_than()` to ImapClient for server-side date filtering
  - Added CLI commands: `corvus email lists`, `list-add`, `list-remove`, `rationalize`, `cleanup`
  - Enhanced `corvus email review` — `[l]ist` option to add sender during review; `[m]ove` option to pick an IMAP folder (lazy-loaded, cached); auto-applies pending items from known lists at review start
  - Passed `sender_lists_path` through CLI triage and summary commands
  - Fixed pre-existing test gap (JOB_ALERT category count in test_email_schemas.py)
  - Created `tests/test_sender_lists.py` (31 tests)
  - Added 10 triage/summary sender list integration tests to `tests/test_email_pipeline.py`
  - Added 10 CLI tests to `tests/test_email_cli.py`
  - Added 2 IMAP tests to `tests/test_email_imap.py`
  - Full suite: **605 passed**, 0 failed

### Architecture decisions
- JSON file at `data/sender_lists.json` (not SQLite) — user preference, loaded once per run
- Priority order in JSON — user-editable, no code change to reorder
- Sender list matches bypass `force_queue` — explicit user rules, not LLM guesses
- White list is special — only list that does NOT skip LLM (needs classification for summaries/action items)
- All addresses case-insensitive (lowercased on storage and lookup)
- Atomic JSON writes (temp file + rename)

### New files
- `corvus/schemas/sender_lists.py`
- `corvus/sender_lists.py`
- `tests/test_sender_lists.py`

### Modified files
- `corvus/config.py` — added EMAIL_SENDER_LISTS_PATH
- `corvus/schemas/email.py` — sender_list field, sender_list_applied audit action
- `corvus/audit/email_logger.py` — log_sender_list_applied()
- `corvus/orchestrator/email_pipelines.py` — sender list check in triage + summary
- `corvus/integrations/imap.py` — fetch_uids_older_than()
- `corvus/cli.py` — new commands + triage/summary/review updates
- `tests/test_email_schemas.py` — fixed JOB_ALERT test gap
- `tests/test_email_pipeline.py` — sender list integration tests
- `tests/test_email_cli.py` — CLI command tests
- `tests/test_email_imap.py` — fetch_uids_older_than test

### Next steps
- Seed `data/sender_lists.json` with real addresses as trust is established
- Consider web dashboard for list management (Phase 4+)
- Move Epic 19 to backlog_archive.md

---

## Session: 2026-02-27 (session 18) — Email Pipeline (Phase 3)

### What was done
- **Complete Epic 16: Email Engine Foundation**
  - Added `imap-tools>=1.7` dependency to pyproject.toml (core, not optional)
  - Added EMAIL_* config vars to `corvus/config.py` (accounts path, review DB, audit log, batch size)
  - Created `corvus/schemas/email.py` — all email Pydantic models (EmailAccountConfig, EmailEnvelope, EmailMessage, EmailCategory, EmailClassification, EmailAction, EmailTriageTask, InvoiceData, ActionItem, EmailExtractionResult, EmailTriageResult, EmailSummaryResult, EmailReviewQueueItem, EmailAuditEntry)
  - Created `corvus/integrations/imap.py` — async ImapClient wrapping imap-tools (asyncio.to_thread pattern), Gmail COPY+DELETE move support, folder auto-creation
  - Created `corvus/executors/email_classifier.py` — LLM classification with category→action mapping, confidence gate thresholds
  - Created `corvus/executors/email_extractor.py` — LLM extraction for invoices/receipts/action items

- **Complete Epic 17: Email Triage Pipeline**
  - Created `corvus/queue/email_review.py` — SQLite review queue (separate table from doc tagging)
  - Created `corvus/audit/email_logger.py` — JSONL audit log (separate file from doc tagging)
  - Created `corvus/router/email.py` — confidence gate routing with force_queue support
  - Created `corvus/orchestrator/email_pipelines.py` — run_email_triage() and run_email_summary() pipeline handlers
  - Added EMAIL_TRIAGE and EMAIL_SUMMARY intents to schemas/orchestrator.py
  - Added email intents to planner/intent_classifier.py prompt
  - Added _dispatch_email_triage() and _dispatch_email_summary() to orchestrator/router.py
  - Added `corvus email` CLI command group (triage, review, summary, status, accounts)

- **Complete Epic 18: Email Intelligence**
  - Updated `corvus/digest/daily.py` — email stats in daily digest
  - Updated `corvus/orchestrator/history.py` — email result summarization
  - Updated `corvus/voice/pipeline.py` — voice-friendly email result speech

- **112 new email tests** across 7 test files
- **552 total tests pass, 0 failures**
- Updated `backlog_current.md` — Epics 13-14, 16-18 moved to archive
- Updated `backlog_archive.md` — full entries for Epics 13-14, 16-18
- Updated `.gitignore` — added `secrets/email_accounts.json`

### Architecture decisions
- Separate SQLite table + JSONL audit file for email (no mixing with document tagging)
- Account config in JSON file (secrets/email_accounts.json) — structured per-account data
- imap-tools + asyncio.to_thread() — proven library, same async bridge as voice/sounddevice
- Email is core dependency (not optional like voice)
- Extraction runs inline during triage (immediate while body is in memory)
- Gmail handling via `is_gmail` flag (explicit, not auto-detect)

### Git status at compaction
- Commit message written but **not yet committed** — user handles all git ops
- `tools/` was accidentally staged (embedded git repo warning) — fixed: `git rm --cached tools/atlas-voice-training`, added `tools/` to `.gitignore`
- All changes staged except the `git rm --cached` step which user needs to run

### Current state
- User is setting up `secrets/email_accounts.json` for first live test
- First account is a **personal domain on PurelyMail** (standard IMAP, `is_gmail: false`)
- User has existing IMAP folders they want Corvus to file to eventually
- Config includes extra folders beyond the 3 triage uses today (inbox/processed/receipts) for future explicit filing ("file this to the stripe folder")

### Immediate next steps
1. User finishes `secrets/email_accounts.json` setup
2. Live test: `corvus email triage --limit 5`
3. Debug any issues with real IMAP/LLM interaction
4. `corvus email review` to approve/reject queued actions

### Open tasks (deferred)
- Train custom "hey corvus" wake word (tools/atlas-voice-training cloned, nvidia-container-toolkit not yet installed)

---

## Session: 2026-02-27 (session 17)

### What was done
- **openWakeWord now works on Python 3.12** — installed from GitHub `main` branch which replaced `tflite-runtime` with `ai-edge-litert` (PyPI v0.6.0 still has the broken dep)
- Updated `pyproject.toml` to install openwakeword from GitHub main
- Updated `wakeword.py` to support loading models by name (e.g., `hey_jarvis`) in addition to file paths
- Changed default `VOICE_WAKEWORD_MODEL_PATH` from `models/corvus.onnx` to `hey_jarvis` (pre-trained, auto-downloaded)
- Fixed `test_voice_audio.py::TestAudioPlayer::test_play_calls_sounddevice` — scipy (now installed as transitive dep of openwakeword) exposed a mock gap in device rate query
- Cloned `atlas-voice-training` into `tools/` for custom wake word training
- **440 tests pass, 0 failures**

### Open task: Train custom "hey corvus" wake word
- **Tool:** `tools/atlas-voice-training` (Docker-based, ~1 hour on RTX 4090)
- **Prerequisites not yet completed:**
  1. Install `nvidia-container-toolkit` (needs NVIDIA apt repo added first):
     ```bash
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt update && sudo apt install nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```
  2. Ensure user is in `docker` group: `sudo usermod -aG docker $USER && newgrp docker`
  3. Run training: `cd tools/atlas-voice-training && ./train-wakeword.sh --standalone`
     - Enter wake word: `hey corvus`
     - Accept default settings (50k samples, 100k steps)
     - Output: `docker-output/hey_corvus.onnx`
  4. Copy model: `cp tools/atlas-voice-training/docker-output/hey_corvus.onnx models/`
  5. Update `VOICE_WAKEWORD_MODEL_PATH` in config or `secrets/internal.env` to point to the `.onnx` file
- **Current interim:** `hey_jarvis` pre-trained model works as a placeholder

### Decisions made
- openWakeWord installed from GitHub main (`git+https://...@main`) rather than PyPI, because PyPI v0.6.0 still ships `tflite-runtime` which has no Python 3.12 wheel
- Default wake word set to `hey_jarvis` (pre-trained) as interim until custom model is trained
- `WakeWordDetector` now distinguishes file paths (has extension or path separator) from model names (bare string like `hey_jarvis`)

### Known issues
- openwakeword PyPI release still broken for Python 3.12 — pinned to GitHub main in `pyproject.toml`
- `nvidia-container-toolkit` not yet installed on host, blocking Docker GPU training

---

## Session: 2026-02-27 (session 16)

### What was done
- Implemented **Epic 13: Voice Engine Foundation** (S13.1–S13.6, all complete)
- Implemented **Epic 14: Local Voice Assistant** (S14.1–S14.4, all complete including smoke test)

#### Overview
Added voice I/O to Corvus — STT (faster-whisper), TTS (Kokoro), wake word detection (openWakeWord), and audio I/O (sounddevice). Voice is a transport layer: it replaces `click.prompt()` with STT and `click.echo()` with TTS while reusing the existing orchestrator pipeline (`classify_intent()` -> `dispatch()` -> `summarize_response()`) unchanged.

#### New files (10)
- **`corvus/voice/__init__.py`**: Package init
- **`corvus/voice/audio.py`**: `AudioCapture` (async context manager, sounddevice InputStream with asyncio.Queue bridge), `AudioPlayer` (play/stop with automatic resampling to device native rate), `generate_ack_tone()` (sine wave), `_compute_rms()` for silence detection, `_resample()` via scipy
- **`corvus/voice/stt.py`**: `SpeechRecognizer` (async context manager wrapping faster-whisper WhisperModel, returns `TranscriptionResult`, VAD filter enabled by default)
- **`corvus/voice/tts.py`**: `SpeechSynthesizer` (async context manager wrapping Kokoro KPipeline, `synthesize()` yields per-sentence chunks, `synthesize_full()` concatenates)
- **`corvus/voice/wakeword.py`**: `WakeWordDetector` (async context manager wrapping openWakeWord, `process_frame()` returns `WakeWordEvent` or None, gracefully disables if model file missing)
- **`corvus/voice/pipeline.py`**: `VoicePipeline` — main loop: drain -> ack tone -> record_until_silence -> STT -> classify_intent -> dispatch -> response_to_speech -> TTS. Supports `--no-wakeword` mode (Enter key triggers listening). Error recovery per utterance. Conversation persistence via ConversationStore. Prints state prompts (`[Press Enter to speak]`, `Listening...`, `You: <text>`).
- **`corvus/schemas/voice.py`**: `VoiceState` enum, `AudioConfig`, `TranscriptionResult`, `WakeWordEvent` Pydantic models

#### Modified files (3)
- **`pyproject.toml`**: Added `[project.optional-dependencies] voice` group (faster-whisper, kokoro, sounddevice, soundfile, openwakeword, numpy)
- **`corvus/config.py`**: Added 11 `VOICE_*` config variables (STT model/device/compute/beam, TTS voice/lang/speed, wakeword path/threshold, silence/max listen duration)
- **`corvus/cli.py`**: Added `corvus voice` command with `--model`, `--keep-alive`, `--new`, `--resume`, `--voice`, `--no-wakeword`, `--list-voices` options. Includes `_check_voice_deps()` for clear error on missing optional deps and `_list_voices()` to show all 54 available Kokoro voices.

#### Test files (7)
- `tests/test_voice_schemas.py` (12 tests)
- `tests/test_voice_audio.py` (15 tests)
- `tests/test_voice_stt.py` (8 tests)
- `tests/test_voice_tts.py` (7 tests)
- `tests/test_voice_wakeword.py` (7 tests)
- `tests/test_voice_pipeline.py` (14 tests)
- `tests/test_voice_cli.py` (7 tests)

**75 voice tests pass, 433 total tests pass (0 regressions)**

#### Key design decisions
- Voice deps are optional (`pip install corvus[voice]`), core CLI unaffected
- VRAM budget: ~4 GB (STT+TTS) + ~11 GB (Ollama) = ~15 GB / 24 GB — everything stays resident
- No-wakeword mode essential for dev before custom wake word model is trained
- All hardware mocked in tests via MagicMock/AsyncMock + sys.modules patching for uninstalled optional deps
- `_response_to_speech()` converts all OrchestratorAction types to voice-friendly text
- AudioPlayer resamples to device native rate (fixes garbled audio on bluetooth/non-24kHz devices)
- Pipeline uses `synthesize_full()` instead of per-sentence streaming to avoid crackling between chunks
- Queue drain before recording prevents stale audio frames from previous idle period

#### Smoke test results (S14.4)
- TTS: Kokoro loads in ~1s, generates clear speech, 54 voices available via `--list-voices`
- STT: faster-whisper large-v3-turbo loads in ~2s, accurate transcription
- Full loop: `corvus voice --no-wakeword` works end-to-end
- VRAM: ~3.4 GB baseline, well within 24 GB budget with Ollama
- Known issue: openwakeword requires `tflite-runtime` which doesn't support Python 3.12 yet; `--no-wakeword` mode works as intended workaround

### What's next
- **Epic 15**: Mobile Voice Access (PWA + FastAPI WebSocket) — deferred, outline only
- Train custom wake word model for "Corvus" via openWakeWord (when tflite-runtime supports Python 3.12)
- Evaluate alternative wake word solutions if tflite-runtime remains incompatible

### Blockers
- openwakeword blocked by tflite-runtime Python 3.12 incompatibility (non-critical, --no-wakeword covers the use case)

---

## Session: 2026-02-25 (session 15)

### What was done
- Implemented **Epic 12: Conversation Memory Persistence (V2)** (S12.1–S12.7, all complete including smoke test)

#### Overview
`corvus chat` conversation history was previously in-memory (`ConversationHistory`) and lost on exit. Users couldn't resume where they left off or review past conversations. This epic persists conversations to SQLite so sessions survive restarts, with CLI flags for managing sessions.

#### Changes
- **`corvus/config.py`**: Added `CONVERSATION_DB_PATH` (default: `data/conversations.db`)
- **`corvus/orchestrator/conversation_store.py`** (new): SQLite-backed `ConversationStore` class — `conversations` table (id UUID, title, created_at, updated_at) + `messages` table (id autoincrement, conversation_id FK, role, content, created_at) + index on conversation_id. Methods: `create()` (UUID + auto-title from first message truncated to 60 chars), `add_message()`, `load_messages()`, `get_most_recent()`, `list_conversations()`, `get_conversation()` (accepts ID prefix via LIKE query). Context manager pattern, WAL journal mode.
- **`corvus/orchestrator/history.py`**: Extended `ConversationHistory` with optional persistence. New params: `store`, `conversation_id`. New classmethod `from_store()` loads all messages from SQLite, trims to `max_turns * 2` in memory. New method `set_persistence()` for deferred creation pattern. New property `conversation_id`. Backward compatible — all 14 existing tests pass unchanged.
- **`corvus/cli.py`**: New flags on `chat` command: `--new` (fresh conversation), `--list` (show recent conversations and exit), `--resume <id>` (resume specific conversation by ID prefix). Default behavior: resume most recent conversation; if none exist, start new. Deferred creation: conversation only created in DB on first user message (avoids empty conversations if user quits immediately). `_list_conversations()` runs before `_validate_config()` so it works even without Paperless configured. Store opened in `_chat_async()`, closed in `finally` block.

### Test summary
- **38 new tests** across 3 files:
  - `tests/test_conversation_store.py` (new) — 26 tests: _generate_title (6), create (3), add_message (3), load_messages (2), get_most_recent (2), list_conversations (5), get_conversation (4), context manager (1)
  - `tests/test_conversation_history.py` — 6 new: persistence to store, backward compat, from_store loads, from_store trims, set_persistence, conversation_id property
  - `tests/test_cli.py` — 6 new: --list empty, --list with conversations, --new, default resumes, --resume specific, --resume bad id
- All existing chat tests updated with `CONVERSATION_DB_PATH` monkeypatch to tmp_path for isolation
- **All 358 tests passing** (352 fast + 6 slow/deselected)

### Files changed
| File | Change |
|------|--------|
| `corvus/config.py` | `CONVERSATION_DB_PATH` |
| `corvus/orchestrator/conversation_store.py` | **New** — SQLite persistence layer |
| `corvus/orchestrator/history.py` | Optional persistence (store, conversation_id, from_store, set_persistence) |
| `corvus/cli.py` | `--new`, `--list`, `--resume` on chat; store lifecycle; deferred creation |
| `tests/test_conversation_store.py` | **New** — 26 tests |
| `tests/test_conversation_history.py` | 6 new tests |
| `tests/test_cli.py` | 6 new tests + CONVERSATION_DB_PATH isolation in all chat tests |
| `docs/backlog_current.md` | S12.1–S12.6 marked complete |
| `docs/backlog_archive.md` | Epic 12 archived |

### Current state
- **Epics 1–12:** Complete (all archived)
- **All tests passing:** 358 total
- **Test breakdown:**
  - `test_cli.py` — 62
  - `test_conversation_store.py` — 26
  - `test_conversation_history.py` — 20
  - `test_intent_classifier.py` — 13 (1 slow)
  - `test_pipeline_handlers.py` — 22
  - `test_orchestrator_router.py` — 20
  - `test_search_integration.py` — 17
  - `test_ollama_client.py` — 4 (1 slow)
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
  - `test_paperless_client.py` — 6 (4 integration)
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

### Smoke test results (S12.7)
- `corvus chat --list` (with prior data) — shows conversations with title, timestamp, message count, short ID
- `corvus chat --new` + send message + quit — creates conversation with auto-title, 2 messages
- `corvus chat --new` + quit immediately — **no empty conversation created** (deferred creation works)
- `corvus chat` (no flags) — resumes most recent: "Resuming: Hello Corvus, how are you today?"
- `corvus chat --resume f5c4` — prefix match resumes older conversation: "Resuming: hello"
- `corvus chat --resume zzz-bad-id` — "Error: No conversation found matching 'zzz-bad-id'."
- `corvus chat --new` + web search ("weather in New York") — creates new conversation, web search works, 2 msgs persisted
- `corvus chat --list` — all 3 conversations shown, message counts correct, newest first

### Commit status
- **Not yet committed.** All changes staged and ready. Commit message written.
- 10 files: 8 modified + 2 new (`conversation_store.py`, `test_conversation_store.py`)

### Next steps
- Consider next: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard

---

## Session: 2026-02-25 (session 14)

### What was done
- Implemented **Epic 11: Web Search Page Content Fetching** (6 stories, all complete)

#### Overview
The web search pipeline previously only had DDG snippets — short page descriptions, not actual page content. For factual queries (weather, scores, schedules), snippets often lacked the specific data needed. Now, after DDG search returns results, the top N pages are fetched, readable text is extracted via `trafilatura`, truncated to fit context, and included alongside snippets in the LLM summarization prompt. Gracefully degrades to snippet-only if fetches fail.

#### Changes
- **`pyproject.toml`**: Added `trafilatura>=1.6` dependency
- **`corvus/config.py`**: 3 new config variables — `WEB_SEARCH_FETCH_PAGES` (2), `WEB_SEARCH_PAGE_MAX_CHARS` (8000), `WEB_SEARCH_FETCH_TIMEOUT` (10)
- **`corvus/integrations/search.py`**: Added `page_content: str | None = None` field to `SearchResult`. New functions: `_truncate_on_word_boundary()`, `_fetch_single_page()` (GET with User-Agent, check 2xx + text/html, trafilatura.extract via to_thread, truncate, returns None on any failure), `fetch_page_content()` (creates httpx.AsyncClient, fetches top N concurrently via asyncio.gather, populates page_content on each result)
- **`corvus/orchestrator/pipelines.py`**: `run_search_pipeline()` now accepts `fetch_pages`, `page_max_chars`, `fetch_timeout` params. After DDG search, fetches page content if `fetch_pages > 0`. Context builder includes `Page content:` section when available. System prompt updated: "search result snippets" → "search results", added "Prefer data from page content over snippets when both are available."
- **`corvus/orchestrator/router.py`**: `_dispatch_search()` imports and passes the 3 new config values to `run_search_pipeline()`

#### Graceful degradation
- Individual page failure → `None`, other pages unaffected
- All pages fail → context identical to snippet-only format
- `fetch_pages=0` → skips fetching entirely (config-disabled)
- Non-HTML responses → skipped (content-type check)
- trafilatura returns None → treated as failure, skipped

### Test summary
- **17 new tests** across 3 files:
  - `tests/test_search_integration.py` — 12 new: truncation (2), _fetch_single_page (extracts, truncates, timeout, HTTP error, non-HTML, extraction failure), fetch_page_content (top N only, partial failures, all failures, empty list)
  - `tests/test_pipeline_handlers.py` — 4 new: page content in LLM context, snippet-only fallback on all failures, skips fetch when disabled, fetch progress messages
  - `tests/test_orchestrator_router.py` — 1 new: _dispatch_search passes fetch config values
- **All 320 tests passing** (314 fast + 6 slow/skipped)

### Files changed
| File | Change |
|------|--------|
| `pyproject.toml` | Added `trafilatura>=1.6` |
| `corvus/config.py` | 3 new config variables |
| `corvus/integrations/search.py` | `page_content` field, `_fetch_single_page()`, `fetch_page_content()` |
| `corvus/orchestrator/pipelines.py` | Page fetching integration, updated system prompt |
| `corvus/orchestrator/router.py` | Pass fetch config values in `_dispatch_search()` |
| `tests/test_search_integration.py` | 12 new tests |
| `tests/test_pipeline_handlers.py` | 4 new tests |
| `tests/test_orchestrator_router.py` | 1 new test |
| `docs/backlog_current.md` | Epic 11 moved to archive |
| `docs/backlog_archive.md` | Epic 11 archived |

### Current state
- **Epics 1–11:** Complete (all archived)
- **All tests passing:** 320 total
- **Test breakdown:**
  - `test_cli.py` — 56
  - `test_conversation_history.py` — 14
  - `test_intent_classifier.py` — 13 (1 slow)
  - `test_pipeline_handlers.py` — 22
  - `test_orchestrator_router.py` — 20
  - `test_search_integration.py` — 17
  - `test_ollama_client.py` — 4 (1 slow)
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
  - `test_paperless_client.py` — 6 (4 integration)
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
- Factual queries (weather, etc.) now return richer answers with actual data from page content
- `corvus ask "hello"` — general_chat still works, no regression
- Snippet-only behavior preserved when all fetches fail or `WEB_SEARCH_FETCH_PAGES=0`

### Next steps
- Consider next: Epic 12 (conversation memory persistence V2), Phase 3 (email pipeline), voice I/O, web dashboard

---

## Session: 2026-02-25 (session 13)

### What was done
- Implemented **Epic 9: CHAT_MODEL Config** (4 stories, all complete)
- Implemented **Epic 10: Conversation Memory** (6 stories, all complete)

#### Epic 9 — CHAT_MODEL Config
Added a `CHAT_MODEL` config variable that allows `corvus chat` and web search summarization to use a separate (larger) LLM model for free-form text generation while keeping the default model for structured output tasks (classification, extraction).

- **`corvus/config.py`**: Added `CHAT_MODEL` (empty string = use same model for everything)
- **`corvus/orchestrator/router.py`**: `dispatch()` accepts `chat_model` param, computes `effective_chat_model = chat_model or model`, passes to `_dispatch_chat()` and `_dispatch_search()`. All other dispatch functions (tag, fetch, status, digest) keep using `model`.
- **`corvus/cli.py`**: Both `_ask_async()` and `_chat_async()` resolve `CHAT_MODEL` and pass to `dispatch()`. Echo chat model when set.

#### Epic 10 — Conversation Memory
The `corvus chat` REPL was completely stateless. Each turn was independently classified and dispatched with no prior context. Now has in-memory conversation history (lost on exit, V1).

- **`corvus/orchestrator/history.py`** (new): `ConversationHistory` class — stores user/assistant message pairs, provides `get_messages()` (Ollama-compatible list) and `get_recent_context()` (formatted text for classifier). `summarize_response()` creates brief summaries for non-chat intents (fetch → "Found N document(s): ...", tag → "Tagged N ...", etc.).
- **`corvus/integrations/ollama.py`**: `chat()` accepts optional `messages` param — inserted between system prompt and current user message. Backwards-compatible (defaults to None).
- **`corvus/planner/intent_classifier.py`**: `classify_intent()` accepts `conversation_context` param. Context-aware prompt template includes recent conversation. Added rule 8: "resolve ambiguous references like 'the first one', 'that document', 'download it'".
- **`corvus/orchestrator/router.py`**: `dispatch()` accepts `conversation_history`, forwards to `_dispatch_chat()` which passes to `ollama.chat(messages=...)`.
- **`corvus/cli.py`**: `_chat_async()` creates `ConversationHistory(max_turns=20)`, passes context to classifier, passes message history to dispatch, records user/assistant turns after each exchange. `_ask_async()` remains stateless.

#### S10.7 — Interactive fetch in chat mode (smoke test fix)
Smoke testing revealed that `corvus chat` could resolve document references via conversation memory (e.g., "tell me about the first one" correctly became a search for "PHH Mortgage Escrow"), but the results were display-only — no way to actually open or download the document. Fixed by adding interactive fetch selection to the chat REPL:
- **Single result**: auto-opens in browser (same as `corvus ask`/`corvus fetch`)
- **Multiple results**: numbered list + `[1-N, s to skip]` prompt; pick a number to open, press Enter to skip (default `s`) and continue chatting
- Replaced old `test_chat_fetch_inline` with 3 new tests: `test_chat_fetch_select`, `test_chat_fetch_skip`, `test_chat_fetch_single_auto_opens`

### Test summary
- **30 new tests** (net +30, replaced 1 old test with 3 new):
  - `tests/test_conversation_history.py` (new) — 14 tests: history basics (empty, add, trim, copy), get_recent_context (empty, formatting, max_turns), summarize_response (chat, fetch, fetch-empty, tag, status, web_search, clarification)
  - `tests/test_orchestrator_router.py` — 6 new: chat_model (chat uses, search uses, default fallback, tag ignores), conversation_history (forwarded, None)
  - `tests/test_ollama_client.py` — 2 new: messages builds correct payload, without messages backwards compatible
  - `tests/test_intent_classifier.py` — 2 new: with conversation context, without context unchanged
  - `tests/test_cli.py` — 6 new: chat_model display, history to dispatch, context to classifier, ask no history, chat fetch select, chat fetch skip, chat fetch single auto-open (replaced 1 old inline-only test)
- **All 303 tests passing** (297 fast + 6 slow/skipped)

### Files changed
| File | Change |
|------|--------|
| `corvus/config.py` | Added `CHAT_MODEL` |
| `corvus/orchestrator/router.py` | `chat_model` + `conversation_history` params on `dispatch()` and `_dispatch_chat()` |
| `corvus/integrations/ollama.py` | `messages` param on `chat()` |
| `corvus/planner/intent_classifier.py` | `conversation_context` param + rule 8 + context-aware prompt template |
| `corvus/orchestrator/history.py` | **New** — ConversationHistory + summarize_response |
| `corvus/cli.py` | Wire CHAT_MODEL + history into `_ask_async()` and `_chat_async()`; interactive fetch selection in chat mode |
| `tests/test_conversation_history.py` | **New** — 14 tests |
| `tests/test_orchestrator_router.py` | 6 new tests |
| `tests/test_ollama_client.py` | 2 new tests |
| `tests/test_intent_classifier.py` | 2 new tests |
| `tests/test_cli.py` | 7 new tests (replaced 1 old) |
| `docs/backlog_current.md` | Epics 9 + 10 |

### Current state
- **Epics 1–10:** Complete (Epics 1–7 archived, 8–10 in backlog_current)
- **All tests passing:** 303 total
- **Test breakdown:**
  - `test_cli.py` — 56
  - `test_conversation_history.py` — 14
  - `test_intent_classifier.py` — 13 (1 slow)
  - `test_pipeline_handlers.py` — 18
  - `test_orchestrator_router.py` — 19
  - `test_search_integration.py` — 5
  - `test_ollama_client.py` — 4 (1 slow)
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
  - `test_paperless_client.py` — 6 (4 integration)
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

### VRAM note
When `CHAT_MODEL=qwen2.5:14b-instruct` and default is `qwen2.5:7b-instruct`, both may coexist in VRAM during chat (7B ~4.5GB + 14B ~10GB ≈ 14.5GB, fits 24GB RTX 4090). Ollama's `keep_alive` handles lifecycle.

### Smoke test results
- `CHAT_MODEL=qwen2.5:14b-instruct` — auto-selected and displayed correctly
- "hello" → "What did I just say?" → correctly recalled "hello" (conversation memory working)
- "find documents related to my mortgage" → 31 results → "tell me about the first one" → classifier resolved to "PHH Mortgage Escrow", searched and found 3 docs
- "Show me the first document" / "Fetch me the first document" → also resolved correctly
- "Tell me about document number 1" → **did not resolve** (literal "document number 1" passed as query). Natural references ("the first one") work better than index-style ("number 1") — inherent V1 limitation of text-based context summary
- "What is the status" → correctly switched to show_status intent mid-conversation
- Interactive fetch in chat was missing (display-only, no delivery) → fixed with S10.7

### Re-smoke-test results (S10.7 fix confirmed)
- CHAT_MODEL display: confirmed
- Conversation memory recall: confirmed
- Multi-result fetch + interactive selection: confirmed (number opens browser)
- Multi-result fetch + skip (Enter/s): confirmed (returns to prompt)
- Single-result fetch auto-open: confirmed
- Reference resolution + fetch delivery: confirmed
- Intent switching mid-conversation: confirmed
- Stateless `corvus ask`: confirmed (no history, exits immediately)

### Next steps
- Consider next: Phase 3 (email pipeline), voice I/O, web dashboard, web search page content fetching, conversation memory persistence (V2)

---

## Session: 2026-02-25 (session 12)

### What was done
- Implemented **Epic 8: Web Search Intent** (9 stories, all complete)

#### Overview
Added `WEB_SEARCH` as a new intent for `corvus ask`/`corvus chat`. Users can now ask questions requiring current/external knowledge (weather, news, factual lookups). Corvus searches DuckDuckGo, summarizes results via LLM with numbered source citations, and gracefully falls back to LLM-only answers when search fails.

#### Changes
- **`pyproject.toml`**: Added `ddgs>=9.0` dependency (formerly `duckduckgo-search`, renamed upstream)
- **`corvus/schemas/orchestrator.py`**: `WEB_SEARCH` intent enum, `search_query` on `IntentClassification`, `WebSearchSource` + `WebSearchResult` models, updated `OrchestratorResponse.result` union
- **`corvus/integrations/search.py`** (new): Async DDG wrapper — `SearchResult`, `SearchError`, `_search_sync()` (lazy DDGS import), `web_search()` (asyncio.to_thread). Uses `backend="duckduckgo"` by default (library defaults to `"auto"` which rotates through Bing/Google/etc.)
- **`corvus/config.py`**: `WEB_SEARCH_MAX_RESULTS` (default 5)
- **`corvus/planner/intent_classifier.py`**: web_search as intent #7 with examples; rules 6 (when to classify as web_search) and 7 (prefer general_chat when ambiguous)
- **`corvus/orchestrator/pipelines.py`**: `run_search_pipeline()` (DDG search → LLM summarization prompt with citations, temperature=0.3) + `_search_fallback_chat()` (LLM-only with "search unavailable" disclaimer). Summarization prompt instructs LLM to extract specific facts from snippets, give direct answers, and never redirect users to visit websites.
- **`corvus/orchestrator/router.py`**: `_dispatch_search()` with search_query fallback to user_input; WEB_SEARCH branch before GENERAL_CHAT
- **`corvus/cli.py`**: `WebSearchResult` rendering — summary + numbered sources with title + URL

### Test summary
- **14 new tests**:
  - `tests/test_search_integration.py` (new) — 5 tests: _search_sync DDGS call, error wrapping, param forwarding, web_search async, empty results
  - `tests/test_pipeline_handlers.py` — 4 new: search summary, no results fallback, SearchError fallback, progress messages
  - `tests/test_orchestrator_router.py` — 2 new: dispatch with search_query, fallback to user_input
  - `tests/test_intent_classifier.py` — 1 new: web_search intent classification
  - `tests/test_cli.py` — 2 new: ask web search rendering, chat web search
- **All 273 tests passing** (267 fast + 6 slow/skipped)

### Files changed
| File | Change |
|------|--------|
| `pyproject.toml` | Added `ddgs>=9.0` |
| `corvus/schemas/orchestrator.py` | WEB_SEARCH intent, search_query, WebSearchSource, WebSearchResult, union |
| `corvus/integrations/search.py` | **New** — async DDG wrapper |
| `corvus/config.py` | WEB_SEARCH_MAX_RESULTS |
| `corvus/planner/intent_classifier.py` | web_search intent #7 + rules 6-7 |
| `corvus/orchestrator/pipelines.py` | run_search_pipeline, _search_fallback_chat |
| `corvus/orchestrator/router.py` | _dispatch_search, WEB_SEARCH branch |
| `corvus/cli.py` | WebSearchResult rendering |
| `tests/test_search_integration.py` | **New** — 5 tests |
| `tests/test_pipeline_handlers.py` | 4 new search tests |
| `tests/test_orchestrator_router.py` | 2 new search tests |
| `tests/test_intent_classifier.py` | 1 new web_search test |
| `tests/test_cli.py` | 2 new web search tests |
| `docs/backlog_current.md` | Epic 8 added |

### Current state
- **Epics 1–7:** Complete (archived/ready to archive)
- **Epic 8:** Complete (web search intent)
- **All tests passing:** 273 total
- **Test breakdown:**
  - `test_cli.py` — 50
  - `test_intent_classifier.py` — 12 (1 slow)
  - `test_pipeline_handlers.py` — 18
  - `test_orchestrator_router.py` — 13
  - `test_search_integration.py` — 5
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
  - `test_paperless_client.py` — 6 (4 integration)
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

### Next steps
- Smoke test: `corvus ask what is the weather in NYC` — should classify as web_search, return summary + sources
- Smoke test: `corvus ask hello` — should still classify as general_chat
- Archive Epic 7, mark Epic 8 as complete in backlog
- Consider next: Phase 3 (email pipeline), `CHAT_MODEL` config, conversation memory, voice I/O

---

## Session: 2026-02-25 (session 11)

### What was done
- Completed **Epic 7** (S7.3, S7.4, S7.5) — Search Reliability & Smoke Testing is now fully done

#### S7.5 — Paperless Connection Drop Handling
- **Retry logic** (`corvus/integrations/paperless.py`): Added `_retry_on_disconnect()` helper — retries async calls up to `MAX_RETRIES=2` times with 1s delay on `httpx.RemoteProtocolError`. Applied to 5 methods: `_get`, `_patch`, `list_documents`, `upload_document`, `download_document` (each refactored into public method + `_raw` internal, with retry wrapping the raw call). `_get_all_pages` inherits retry via `_get`.
- **CLI error handling** (`corvus/cli.py`): Added `import httpx` and try/except blocks to `_tag_async`, `_fetch_async`, `_review_async`, `_watch_async` catching `RemoteProtocolError` ("Lost connection to Paperless") and `HTTPStatusError` (shows status code). All exit with code 1 and clean error message.

#### S7.3 — Complete ask/chat Test Coverage
- 7 new tests in `tests/test_cli.py`:
  - `test_ask_tag_intent` — TAG_DOCUMENTS returns TagPipelineResult, asserts "Tagging complete" + counts
  - `test_ask_digest_intent` — SHOW_DIGEST returns DigestResult, asserts rendered text
  - `test_ask_fetch_no_results` — FETCH_DOCUMENT with 0 docs, asserts "0 document(s)"
  - `test_ask_fetch_multi_select` — FETCH_DOCUMENT with 3 docs, user selects "2", asserts webbrowser.open with doc 2
  - `test_ask_watch_folder_intent` — WATCH_FOLDER returns INTERACTIVE_REQUIRED, asserts "corvus watch"
  - `test_chat_fetch_inline` — Chat with FETCH_DOCUMENT, docs displayed inline (no selection), then quit
  - `test_ask_dispatch_error` — dispatch raises RemoteProtocolError, asserts "Failed to complete request" + exit 1
- 2 new S7.5 CLI error tests:
  - `test_tag_paperless_connection_error` — run_tag_pipeline raises RemoteProtocolError, clean error + exit 1
  - `test_fetch_paperless_connection_error` — run_fetch_pipeline raises RemoteProtocolError, clean error + exit 1

#### S7.4 — Chat Model Recommendation
- Researched Ollama-available models; recommended `qwen2.5:14b-instruct` for GENERAL_CHAT
- Rationale: meaningful conversation quality uplift over 7B, ~10-11 GB VRAM (Q4_K_M), same model family as existing instruct model, ~5-10s cold load on RTX 4090
- Keep `qwen2.5:7b-instruct` for structured output tasks (classification, extraction)
- Alternatives evaluated: 32b (tight VRAM), gemma2:27b (verbose), mistral-small:22b (trails Qwen), phi-4:14b (stiff), Llama 3.x (no 14B option)
- No code changes — recommendation documented in backlog

### Test summary
- **11 new tests**: 9 in `test_cli.py` (7 S7.3 + 2 S7.5), 2 in `test_paperless_client.py` (retry)
- **All 265 tests passing** (259 fast + 6 slow/skipped)
- Note: `test_paperless_client.py` restructured — unit tests (retry) always run, integration tests use per-test `@_skip_no_paperless` marker instead of module-level `pytestmark`

### Files changed
| File | Change |
|------|--------|
| `corvus/integrations/paperless.py` | `_retry_on_disconnect()`, 5 methods wrapped with retry |
| `corvus/cli.py` | `import httpx`, try/except in 4 async entry points |
| `tests/test_cli.py` | 9 new tests (7 S7.3 + 2 S7.5) |
| `tests/test_paperless_client.py` | 2 new retry tests, restructured skip markers |
| `docs/backlog_current.md` | S7.3, S7.4, S7.5 marked complete |

### Current state
- **Epics 1–7:** Complete (Epic 7 ready to archive)
- **All tests passing:** 265 total
- **Test breakdown:**
  - `test_cli.py` — 48
  - `test_intent_classifier.py` — 11 (1 slow)
  - `test_pipeline_handlers.py` — 14
  - `test_orchestrator_router.py` — 11
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
  - `test_paperless_client.py` — 6 (4 integration)
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

### Next steps
- Archive Epic 7 to `docs/backlog_archive.md`
- Implement `CHAT_MODEL` config variable + `qwen2.5:14b-instruct` for `GENERAL_CHAT` (follow-up from S7.4)
- Consider next epic: Phase 3 (email pipeline), voice I/O, web dashboard, or `corvus chat` conversation memory

---

## Session: 2026-02-25 (session 10)

### What was done
- Completed **S7.2**: Fix "most recent" date misinterpretation + richer result display

#### Fix A — "most recent" date filter (`corvus/executors/query_interpreter.py`)
- **A1**: Added Rule 11 to system prompt — "most recent"/"latest"/"newest" means `sort_order: "newest"`, do NOT set date ranges
- **A2**: Updated few-shot example for "latest mortgage statement" with explicit `date_range_start: null, date_range_end: null`
- **A3**: Added `_strip_today_only_date_range()` deterministic guardrail — strips date range when both start and end equal today (called after `_has_search_fields` check)

#### Fix B — Richer result display
- **B1** (`corvus/orchestrator/pipelines.py`): Build ID→name lookup dicts from already-fetched tags/correspondents/doc_types; doc_dicts now include `correspondent`, `document_type`, `tags` (resolved names, no new API calls)
- **B2** (`corvus/cli.py`): Added `_format_doc_line()` helper producing two-line output per result (line 1: index/date/title/id; line 2: correspondent | doc type | tags). Updated all 4 display sites (fetch single, fetch multi, ask interactive, chat mode)
- **B3** (`corvus/schemas/orchestrator.py`): Updated `documents` field description

### Test summary
- **6 new tests**: 5 in `test_query_interpreter.py` (date-stripping: today-only stripped, explicit preserved, no-range noop, partial preserved; prompt rule assertion), 1 in `test_pipeline_handlers.py` (`test_fetch_pipeline_resolves_metadata_names` with populated correspondent/type/tags)
- Updated `test_pipeline_handlers.py` existing fetch test to assert new doc_dict keys
- Updated `test_cli.py` `test_ask_fetch_intent` doc_dict to include new keys
- **All 254 tests passing** (75 in affected files + 179 remaining)

### Files changed
| File | Change |
|------|--------|
| `corvus/executors/query_interpreter.py` | Rule 11 in prompt; explicit null dates in few-shot; `_strip_today_only_date_range()` post-validation |
| `corvus/orchestrator/pipelines.py` | ID→name lookups; enriched doc_dicts with correspondent/type/tags |
| `corvus/cli.py` | `_format_doc_line()` helper; 4 display sites updated |
| `corvus/schemas/orchestrator.py` | Updated `documents` field description |
| `tests/test_query_interpreter.py` | 5 new tests (date-stripping + prompt rule) |
| `tests/test_pipeline_handlers.py` | 1 new test + updated assertions |
| `tests/test_cli.py` | Updated doc_dict in ask test |

### Current state
- **Epics 1–6:** Complete (archived)
- **Epic 7:** S7.1 ✓, S7.2 ✓, S7.3–S7.5 pending
- **All tests passing:** 254 total
- **Test breakdown:**
  - `test_cli.py` — 39
  - `test_intent_classifier.py` — 11 (1 slow)
  - `test_pipeline_handlers.py` — 14
  - `test_orchestrator_router.py` — 11
  - `test_query_interpreter.py` — 27 (1 slow)
  - `test_retrieval_router.py` — 39
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

### Next steps
- **S7.3**: Full smoke test `corvus ask find the most recent mortgage statement` to verify date fix + richer display
- **S7.4**: Chat model research
- **S7.5**: Paperless connection drop handling
- After Epic 7: Phase 3 (email pipeline), voice I/O, web dashboard

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
