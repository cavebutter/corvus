# Corvus — Active Backlog

> Organized by epic. Stories use `[ ]` (pending), `[~]` (in progress), `[x]` (done, ready to archive).

---

## Epic 21: PWA Mobile Interface

**Goal:** Expose Corvus as a Progressive Web App served from the homelab, accessible on mobile via Cloudflare Tunnel. Dashboard, review queues, and voice — all from the phone.

### Architecture

```
Phone (PWA)  ──Cloudflare Tunnel──>  corvus serve (FastAPI)
                                        ├─ REST API (/api/...)
                                        ├─ WebSocket (/ws/voice)
                                        └─ Static files (HTML/JS/CSS)
```

- **Backend:** FastAPI (async, matches existing httpx/asyncio stack)
- **Frontend:** Vanilla HTML + JS (personal tool, no framework overhead). CSS via Pico or similar classless/minimal framework.
- **Auth:** API key in header, validated per request. Key set via config. Cloudflare Zero Trust provides the outer layer.
- **Voice:** Browser MediaRecorder API → WebSocket → server-side STT → orchestrator → TTS → WebSocket → browser Audio API playback.
- **Notifications:** Ntfy (self-hostable, free, has iOS/Android apps, simple HTTP POST to send).

### Stories

- [x] **S21.1** FastAPI server foundation
  - FastAPI app with lifespan (initialize shared clients: Ollama, Paperless)
  - Static file serving for frontend assets
  - Health endpoint (`GET /api/health`)
  - CORS middleware (Cloudflare Tunnel origins)
  - `corvus serve` CLI command (`--host`, `--port`, `--reload`)
  - API key auth dependency (from config, `X-API-Key` header)
  - Config: `API_KEY`, `SERVE_HOST`, `SERVE_PORT`

- [x] **S21.2** Status & audit API endpoints
  - `GET /api/status` — pipeline stats (pending counts, 24h activity), same data as `corvus status` + `corvus email status`
  - `GET /api/audit/documents?since=&limit=` — document audit log entries
  - `GET /api/audit/email?since=&limit=` — email audit log entries
  - `GET /api/audit/watchdog?since=&limit=` — watchdog audit log entries
  - Pydantic response models (reuse existing schemas where possible)

- [x] **S21.3** Review queue API endpoints
  - `GET /api/review/documents` — pending document review items
  - `POST /api/review/documents/{id}/approve` — approve with optional notes
  - `POST /api/review/documents/{id}/reject` — reject with optional notes
  - `GET /api/review/email` — pending email review items
  - `POST /api/review/email/{id}/approve` — approve (executes IMAP action)
  - `POST /api/review/email/{id}/reject` — reject with optional notes
  - Error handling for already-reviewed items

- [x] **S21.4** Dashboard frontend
  - Single-page layout: status cards (pending counts, 24h stats), recent activity feed
  - Auto-refresh via polling or SSE (simple first, upgrade to SSE if needed)
  - Mobile-first responsive layout
  - PWA manifest (`manifest.json`) + service worker (offline shell, cache static assets)
  - Install-to-homescreen support (icons, theme color, standalone display)

- [x] **S21.5** Review interface frontend
  - Document review: show task details (title, tags, correspondent, confidence, reasoning), approve/reject buttons
  - Email review: show sender, subject, category, proposed action, confidence, approve/reject/delete buttons
  - Swipe or tap interactions for quick review
  - Optimistic UI updates (mark resolved immediately, reconcile on error)

- [ ] **S21.6** Voice over WebSocket
  - WebSocket endpoint (`/ws/voice`) with auth
  - Client: MediaRecorder API captures audio → sends binary frames over WS
  - Server: receives audio → STT (faster-whisper) → intent classification → orchestrator dispatch → TTS → sends audio frames back
  - Client: receives audio → plays via Audio API
  - Tap-to-talk button on frontend (no wake word on mobile — browser limitations)
  - Conversation context maintained per WS connection

- [ ] **S21.7** Push notifications via Ntfy
  - Config: `NTFY_TOPIC`, `NTFY_SERVER` (default: `https://ntfy.sh` or self-hosted)
  - Notify on: new items queued for review, daily digest ready, pipeline errors
  - `corvus/integrations/ntfy.py` — simple async httpx POST
  - Hook into review queue `.add()` and digest generation

- [ ] **S21.8** Cloudflare Tunnel setup & docs
  - Document tunnel configuration for `corvus serve`
  - Cloudflare Zero Trust access policy (email-based or one-time PIN)
  - HTTPS termination at Cloudflare edge (FastAPI serves HTTP internally)
  - Test from mobile on cellular (not LAN)

- [ ] **S21.10** UI polish pass
  - Nav logo sizing (too small at 2rem, needs to be visible on mobile)
  - Overall spacing, typography, and color tuning
  - Dark theme consistency check
  - Mobile layout testing (status cards, activity feed, review items)
  - Tap target sizes for mobile review actions

- [ ] **S21.9** Tests
  - API endpoint tests (TestClient, mocked services)
  - WebSocket voice round-trip test
  - Auth tests (valid key, missing key, wrong key)
  - Frontend: manual smoke test checklist (install, dashboard, review, voice)

### Dependencies
- `fastapi`, `uvicorn[standard]` (server)
- `python-multipart` (file uploads if needed later)
- Existing: `faster-whisper`, `kokoro` (voice), all current pipeline deps

### Out of scope for this epic
- Multi-user auth (personal tool — single API key is sufficient)
- Offline voice processing on mobile (browser API only)
- Email triage trigger from mobile (use `corvus email triage` via SSH/cron for now)

---

## Future: Calendar & Task Integration (Phase 4)

**Goal:** Push Corvus-extracted action items and deadlines to real calendar/todo apps on the user's phone.

**Approach:**
- **Calendar:** CalDAV to a self-hosted Radicale server (Docker). Subscribe from iCloud as an additional calendar. Corvus writes events (invoice due dates, extracted deadlines) via `caldav` Python library.
- **Todo:** Microsoft Graph API for Microsoft To-Do. OAuth2 refresh token flow. Corvus pushes action items extracted from emails/documents as tasks.

**Stories (outline only — not yet decomposed):**
- [ ] Radicale CalDAV server setup (Docker compose)
- [ ] CalDAV integration client (`corvus/integrations/caldav.py`)
- [ ] Microsoft Graph To-Do integration (`corvus/integrations/ms_todo.py`)
- [ ] Pipeline hooks: email action items → To-Do, invoice due dates → Calendar
- [ ] CLI commands: `corvus calendar`, `corvus tasks`
- [ ] PWA integration: upcoming deadlines on dashboard

---

## Future: Smart Home — Philips Hue

**Goal:** Voice and text control of Hue lighting via the local bridge REST API.

**Approach:** Direct integration with the Hue Bridge v2 API (local LAN, no cloud). One-time button press for API key. Control lights, rooms, scenes, schedules. Natural fit for voice commands ("make the living room cozy", "turn off all lights").

- Hue Bridge v2 REST API (local, HTTPS with self-signed cert)
- `corvus/integrations/hue.py` — async client
- `SMART_HOME` intent in classifier
- Confidence gate: "turn off all lights" auto-executes, "set color to red in all rooms" queues for review
- Start with: list rooms/lights, on/off, brightness, scenes

---

## Future: Smart Home — Ecobee Thermostats

**Goal:** Voice and text control of Ecobee thermostats.

**Approach:** Ecobee cloud API (OAuth2, similar to Microsoft Graph pattern). Two thermostats. Natural commands: "set the house to 72", "what's the temperature?", "switch to away mode".

- Ecobee API (cloud, OAuth2 + PIN auth flow)
- `corvus/integrations/ecobee.py` — async client
- Reuses `SMART_HOME` intent
- Higher confidence threshold for temperature changes (reversible but uncomfortable if wrong)

---

## Future: Smart Home — TP-Link/Kasa Smart Plugs

**Goal:** Voice and text control of smart plugs via local network.

**Approach:** `python-kasa` library — fully local, no cloud dependency, async-native. On/off, energy monitoring if supported.

- `python-kasa` library (local, async)
- `corvus/integrations/kasa.py` — async client with device discovery
- Reuses `SMART_HOME` intent
- Auto-execute for on/off (low risk, easily reversible)

---

*Completed epics (1–14, 16–20) are in `backlog_archive.md`.*
