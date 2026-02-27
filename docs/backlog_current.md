# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 13: Voice Engine Foundation — COMPLETE

- [x] S13.1 — Dependencies, config, schemas, and package init
- [x] S13.2 — Audio I/O module (`corvus/voice/audio.py`)
- [x] S13.3 — STT module (`corvus/voice/stt.py`)
- [x] S13.4 — TTS module (`corvus/voice/tts.py`)
- [x] S13.5 — Wake word module (`corvus/voice/wakeword.py`)
- [x] S13.6 — Tests for engine components (52 tests)

## Epic 14: Local Voice Assistant — COMPLETE

- [x] S14.1 — Voice pipeline orchestration (`corvus/voice/pipeline.py`)
- [x] S14.2 — CLI command `corvus voice`
- [x] S14.3 — Pipeline and CLI tests (23 tests)
- [x] S14.4 — Smoke test on hardware (TTS, STT, full loop verified)

## Epic 15: Mobile Voice Access (deferred, outline only)

- [ ] S15.1 — FastAPI WebSocket server
- [ ] S15.2 — PWA frontend (HTML + JS, tap-to-talk)
- [ ] S15.3 — CLI command `corvus serve`
- [ ] S15.4 — Audio codec handling (Opus/AAC)
- [ ] S15.5 — Tests + Cloudflare Tunnel integration

---

*Candidates for future work: Phase 3 (email pipeline), mobile delivery, web dashboard.*
