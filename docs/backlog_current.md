# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 7: Search Reliability & Smoke Testing

**Goal:** Fix query interpreter reliability issues discovered during Epic 6 smoke testing. The LLM sometimes returns high confidence but empty search fields, causing Paperless to return all documents unfiltered.

- [ ] **S7.1** Improve query interpreter field extraction — LLM returns `confidence=0.90` with `text=None`, no tags, no correspondent, no type for "latest mortgage statement". Investigate: prompt tuning, few-shot examples in system prompt, or post-validation that detects "high confidence but no filters" as suspicious and forces `text_search` fallback.
- [ ] **S7.2** Add error handling for Paperless connection drops — `corvus fetch` crashes with raw traceback on `httpx.RemoteProtocolError` (server disconnect). Wrap Paperless API calls in retry/graceful-error paths.
- [ ] **S7.3** Full smoke test suite for `corvus ask` and `corvus chat` — test all intent paths against live Ollama + Paperless, document results.

---

*Candidates for future work: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard, `corvus chat` conversation memory.*
