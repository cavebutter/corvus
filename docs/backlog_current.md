# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 12: Conversation Memory Persistence (V2)

**Goal:** Persist `corvus chat` conversation history to SQLite so sessions can be resumed across restarts. Enable "what did we talk about yesterday?" queries.

- [ ] **S12.1** SQLite schema — conversations table (id, created_at), messages table (conversation_id, role, content, timestamp)
- [ ] **S12.2** Persistence layer — `ConversationStore` class wrapping SQLite, load/save/list conversations
- [ ] **S12.3** Extend `ConversationHistory` — optional persistence backend, auto-save on each turn
- [ ] **S12.4** CLI wiring — `corvus chat` resumes most recent conversation by default, `--new` flag for fresh start
- [ ] **S12.5** Conversation listing — `corvus chat --list` or `corvus history` to show past sessions
- [ ] **S12.6** Tests
- [ ] **S12.7** Smoke test

---

*Candidates for future work: Phase 3 (email pipeline), voice I/O (STT/TTS), mobile delivery, web dashboard.*
