# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 7: Search Reliability & Smoke Testing

**Goal:** Fix query interpreter reliability issues discovered during Epic 6 smoke testing. The LLM sometimes returns high confidence but empty search fields, causing Paperless to return all documents unfiltered.

- [x] **S7.1** Improve query interpreter field extraction — added `_has_search_fields()` post-validation: when LLM returns high confidence but no search fields populated, inject original query as `text_search` fallback.
- [x] **S7.2** Fix "most recent" date misinterpretation + richer result display — LLM was setting `date_range_start=today, date_range_end=today` for "most recent" queries. Added Rule 11 to prompt ("do NOT set date ranges for latest/newest"), explicit null dates in few-shot examples, and `_strip_today_only_date_range()` deterministic guardrail. Also enriched fetch results with correspondent, document type, and tags (resolved from already-fetched metadata) and added two-line display format in CLI via `_format_doc_line()` helper.
- [ ] **S7.3** Full smoke test suite for `corvus ask` and `corvus chat` — test all intent paths against live Ollama + Paperless, document results.
- [ ] **S7.4** Research and recommend a chat model for `corvus chat` — evaluate local models available via Ollama for general conversation quality (GENERAL_CHAT intent). Consider: response quality, VRAM fit (24 GB ceiling), personality/tone suitability for the Corvus persona, and whether a separate model from the instruct model (qwen2.5:7b-instruct) is warranted. Deliver a recommendation with rationale.
- [ ] **S7.5** Add error handling for Paperless connection drops — `corvus fetch` crashes with raw traceback on `httpx.RemoteProtocolError` (server disconnect). Wrap Paperless API calls in retry/graceful-error paths.

---

*Candidates for future work: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard, `corvus chat` conversation memory.*
