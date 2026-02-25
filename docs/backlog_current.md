# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

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

*Candidates for future work: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard, `corvus chat` conversation memory, implement `CHAT_MODEL` config for separate chat model, web search page content fetching (see below).*

---

## Future Enhancements

### Web search: fetch page content for richer answers
**Context:** The current web search pipeline only has access to search result snippets (page descriptions), not actual page content. For queries like weather, sports scores, or detailed factual lookups, snippets often lack the specific data needed (e.g. game times, current temperatures). The LLM summarizes correctly from what it has, but the input is inherently thin.

**Proposed improvement:** After the initial search, fetch the HTML content of the top 1-2 result pages, extract readable text, truncate to fit the LLM context window, and include it alongside the snippets in the summarization prompt. This would give the LLM access to actual temperatures, game schedules, article text, etc.

**Trade-offs:** Adds ~2-3s latency per page fetch, requires HTML-to-text extraction (e.g. `readability-lxml` or `trafilatura`), needs content truncation logic to avoid exceeding LLM context limits. Significantly improves answer quality for current-events and factual queries.
