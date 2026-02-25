# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---


## Epic 9: CHAT_MODEL Config *(Complete)*

**Goal:** Allow `corvus chat` and web search summarization to use a separate (larger) model for free-form text generation while keeping the default model for structured output.

- [x] **S9.1** Config variable — `CHAT_MODEL` in `corvus/config.py` (empty string = same model for everything)
- [x] **S9.2** Thread `chat_model` through router — `dispatch()` gets `chat_model` param, computes `effective_chat_model`, passes to `_dispatch_chat()` and `_dispatch_search()`
- [x] **S9.3** Thread `chat_model` through CLI — `_ask_async()` and `_chat_async()` resolve and pass `CHAT_MODEL`, echo when set
- [x] **S9.4** Tests — 5 new: 4 router (chat/search use chat model, default fallback, tag ignores), 1 CLI (display)

---

## Epic 10: Conversation Memory *(Complete)*

**Goal:** Add in-memory conversation history to `corvus chat` so multi-turn references ("the first one", "download it") work. Lost on exit (V1).

- [x] **S10.1** `ConversationHistory` helper — `corvus/orchestrator/history.py` with add/get messages, `get_recent_context()`, `summarize_response()` for all intent types
- [x] **S10.2** Extend `ollama.chat()` — `messages` param inserted between system prompt and current user message
- [x] **S10.3** Thread `conversation_history` through dispatch — `dispatch()` accepts and forwards to `_dispatch_chat()`
- [x] **S10.4** Conversation context for intent classifier — `conversation_context` param, context-aware prompt template, rule 8 for reference resolution
- [x] **S10.5** Wire into `_chat_async()` — `ConversationHistory` instance in REPL loop, context passed to classifier, history passed to dispatch, responses summarized for history
- [x] **S10.6** Tests — 23 new: 14 conversation history, 2 ollama client, 2 intent classifier, 6 router, 4 CLI (history wiring + ask stateless)
- [x] **S10.7** Interactive fetch in chat mode — single result auto-opens in browser, multiple results offer `[1-N, s to skip]` selection prompt (previously chat showed results inline with no delivery). 3 new tests (select, skip, single auto-open), replacing 1 old inline-only test.

---

## Epic 8: Web Search Intent *(Complete)*

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

*Candidates for future work: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard, web search page content fetching (see below), conversation memory persistence (V2).*

---

## Future Enhancements

### Web search: fetch page content for richer answers
**Context:** The current web search pipeline only has access to search result snippets (page descriptions), not actual page content. For queries like weather, sports scores, or detailed factual lookups, snippets often lack the specific data needed (e.g. game times, current temperatures). The LLM summarizes correctly from what it has, but the input is inherently thin.

**Proposed improvement:** After the initial search, fetch the HTML content of the top 1-2 result pages, extract readable text, truncate to fit the LLM context window, and include it alongside the snippets in the summarization prompt. This would give the LLM access to actual temperatures, game schedules, article text, etc.

**Trade-offs:** Adds ~2-3s latency per page fetch, requires HTML-to-text extraction (e.g. `readability-lxml` or `trafilatura`), needs content truncation logic to avoid exceeding LLM context limits. Significantly improves answer quality for current-events and factual queries.

### Conversation memory persistence (V2)
**Context:** Conversation history (Epic 10) is currently in-memory and lost on exit. A future version could persist history to SQLite, allowing `corvus chat` to resume conversations across sessions and enabling "what did we talk about yesterday?" queries.
