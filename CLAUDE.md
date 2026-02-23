# CLAUDE.md — Corvus: Local Limited-Autonomous Agent System

## Project Purpose

Corvus is a personal homelab AI system consisting of limited-autonomous local LLM agents that manage recurring digital chores. A single orchestrator agent handles all user-facing intake and output, delegating to specialized sub-agents. Human review is required for ambiguous or high-risk actions.

## Identity & Persona

- **Name:** Corvus (from Latin *corvus*, raven — a symbol of intelligence, memory, and message-carrying)
- **Wake word:** "Corvus" (designed for eventual use as an audible trigger phrase)
- **Voice:** Female (target persona for TTS output)
- **Visual identity:** Raven motif — lends itself to logo and icon design

---

## Architecture

### Three-Tier Design

```
User (typed or spoken) → Orchestrator Agent
                              ↓
                    Python Router (deterministic)
                         ↙        ↘
               Planner Agent    Executor Agents (specialized)
```

**Tier 1 — Planner (LLM)**
- Receives structured task descriptions
- Decomposes complex tasks into steps
- Emits structured JSON plans with confidence scores
- Must self-assess confidence on every decision

**Tier 2 — Router / Orchestrator (Python, NOT LLM)**
- Parses and validates planner output against Pydantic schemas
- Applies confidence-based approval gates (see below)
- Dispatches to the correct executor
- Maintains audit log of all actions
- Generates daily digest of auto-executed and queued items

**Tier 3 — Executors (LLM, task-specific)**
- Lightweight, tightly-prompted models
- One job each: classify, extract, summarize, etc.
- Emit structured data only — never free-form text
- Constrained output schemas enforced via Pydantic

### Confidence Gate

| Confidence | Action | Review |
|------------|--------|--------|
| > 0.9 | Execute automatically | Logged, in daily digest |
| 0.7 – 0.9 | Execute automatically | Flagged in daily digest |
| < 0.7 | Queued only | Requires manual approval |

**Initial posture:** All tasks queued for review. Lower thresholds per-pipeline as trust is established.

---

## Hardware & Runtime Constraints

| Component | Spec |
|-----------|------|
| CPU | Intel i9 |
| RAM | 64 GB |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| OS | Pop_OS (COSMIC desktop) |
| LLM Runtime | Ollama (pipeline) + LM Studio (eval only) |
| Storage | Unraid NAS, Synology NAS |
| Network | Cloudflare Tunnels / Zero Trust |

- **24 GB VRAM is the hard ceiling.** Never design for simultaneous model loading.
- Use Ollama `keep_alive` to control model lifecycle (sequential loading pattern).
- LM Studio is for manual experimentation only — never integrated into the automated pipeline.
- All models run locally. No external LLM API calls in the agent pipeline.

---

## Tech Stack

- **Language:** Python 3.x
- **Validation:** Pydantic (all inter-agent communication uses validated schemas)
- **HTTP client:** httpx (async preferred)
- **Task queue:** File-based or SQLite to start; Celery only if concurrency requires it
- **Async:** asyncio for multi-pipeline parallelism
- **LLM runtime:** Ollama (REST API)
- **Image generation:** ComfyUI (future integration, out of scope for now)
- **IDE:** PyCharm Pro
- **Virtual environment:** `~/virtual-envs/corvus/.venv` (Python 3.12.3)

**Explicitly avoided:**
- CrewAI, AutoGen (too much abstraction, obscure failures)
- LangGraph (revisit only if state machines become genuinely complex)
- Any agent framework that obscures the router logic

---

## Git Policy

- **Claude will never `git add`, `git commit`, or `git push`.** The user performs all git operations manually.
- Claude **will** write commit messages when asked (to be copied by the user).
- Claude **may** run `git diff` and `git status` freely without asking permission.
- Claude **may** resolve merge conflicts **only after asking for and receiving explicit permission**.

---

## Project Documents (Maintained by Claude)

Claude is responsible for creating and maintaining the following documents in the project root under `docs/`:

| Document | Purpose |
|----------|---------|
| `docs/backlog_current.md` | Active backlog organized by epics and stories. May be split into multiple files (e.g., `backlog_phase1.md`, `backlog_phase2.md`) for clarity and context conservation. |
| `docs/backlog_archive.md` | Completed epics and stories moved here for reference. Serves as a historical record. |
| `docs/continuity.md` | Session notes updated at the end of each session or before conversation compacting. Referred to at the start of each new session to restore context. |

**Rules:**
- Stories and epics move from `backlog_current` to `backlog_archive` upon completion.
- `continuity.md` must capture: current work in progress, decisions made, blockers, and next steps.
- Claude should proactively update these documents — do not wait to be asked.

---

## Coding Principles

1. **Deterministic over probabilistic.** Prefer Python logic over LLM reasoning wherever possible. LLMs classify and extract; Python decides and routes.
2. **Structured output is non-negotiable.** All planner→executor communication uses validated Pydantic JSON. No free-form text crossing agent boundaries.
3. **Human in the loop.** No destructive actions without review until confidence thresholds are proven. When in doubt, queue it.
4. **All actions are logged and auditable.** Every agent action, decision, and confidence score must be written to the audit log.
5. **Stateless executors.** Executors receive everything they need in the task payload. They do not maintain state between calls.
6. **One pipeline at a time.** Prove out Phase 1 fully before expanding. Resist the urge to build Phase 3 while Phase 1 is unproven.
7. **Fail loudly.** Raise exceptions on schema violations, unexpected LLM output, or failed API calls. Never silently swallow errors and continue.

---

## Pydantic Schemas (Canonical)

All task types must have a defined schema. Examples:

```python
from pydantic import BaseModel
from typing import Literal

class DocumentTaggingTask(BaseModel):
    task_type: Literal["tag_document"]
    file_path: str
    suggested_tags: list[str]
    confidence: float
    requires_review: bool

class EmailTriageTask(BaseModel):
    task_type: Literal["email_triage"]
    message_id: str
    action: Literal["delete", "archive", "flag_for_review", "file_receipt"]
    category: str
    reasoning: str

class InvoiceExtractionTask(BaseModel):
    task_type: Literal["extract_invoice"]
    vendor: str
    amount: float
    currency: str
    due_date: str | None
    confidence: float
```

New task types require a schema before implementation begins.

---

## Agent Capabilities (Target Use Cases)

### Document Management (Paperless-ngx) — Phase 1
- Monitor processed Paperless document directory (cron or on-demand)
- Auto-tag scanned documents (receipts, invoices, correspondence)
- Assign correspondents and document types
- Apply metadata from OCR content
- REST API: `GET /api/documents/`, `PATCH /api/documents/{id}/`, `/api/tags/`, `/api/correspondents/`

### Email Inbox Management — Phase 3
- Identify and delete spam
- File receipts, newsletters, package notices
- Extract invoice fields and queue for review/payment
- Flag action items for calendar or task list
- Summarize individual messages or full inbox
- Support multiple inboxes

### Calendar & Task Management — Phase 4
- Extract action items from emails, documents, and direct orchestrator input
- Create/suggest calendar entries
- Surface upcoming deadlines from invoices and email

### Chatbot — Ongoing
- General-purpose conversational interface via the orchestrator (typed and spoken)

### STT/TTS I/O — MVP Requirement
- Both typed and spoken input/output must be supported at MVP
- Whisper for STT; TTS solution TBD (Coqui XTTS-v2 or similar)

### Future (Out of Scope Now)
- ComfyUI image generation integration
- SillyTavern or similar RP/writing integration
- Outgoing email composition and sending

---

## Implementation Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Paperless document tagging — single model, confidence scoring, review queue, daily digest | **Start here** |
| 2 | Refactor to tiered architecture — separate planner/executor, deterministic router, audit logging | Upcoming |
| 3 | Email pipeline — IMAP, spam, receipts, invoice extraction, action items | Future |
| 4 | Calendar & task integration — cross-pipeline aggregation, deadline tracking | Future |

---

## Secrets Management

### Threat Model

| Threat | Likelihood | Mitigation |
|--------|------------|------------|
| Accidental git commit of secrets | High | `.gitignore` + pre-commit hook |
| Local filesystem read after intrusion | Medium | Encrypted secrets at rest (SOPS) |
| Secrets leaked in outbound requests | Low | Scope separation + audit logging |
| Secrets in logs or tracebacks | Medium | Never log secret values; sanitize exceptions |

### Toolchain: SOPS + age

**SOPS** (Mozilla) encrypts secrets files using **age** keypairs. The encrypted file is safe to commit to git. Decryption happens at runtime using a private key stored locally with strict permissions. No daemon, no cloud dependency, no server.

```bash
# One-time setup
sudo apt install age sops

# Generate keypair (do this once, back up the private key securely)
age-keygen -o ~/.config/sops/age/keys.txt   # private key, chmod 600 automatically
# Public key is printed to stdout — copy it

# Create .sops.yaml in project root (tells SOPS which key to use)
# age: <your-public-key>
```

**.sops.yaml** (committed to git):
```yaml
creation_rules:
  - path_regex: secrets/.*\.env
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Encrypt a secrets file:**
```bash
sops --encrypt secrets/internal.env > secrets/internal.env.enc
```

**Decrypt at runtime (in Python):**
```python
import subprocess
import os
from dotenv import dotenv_values

def load_secrets(encrypted_path: str) -> dict:
    result = subprocess.run(
        ["sops", "--decrypt", encrypted_path],
        capture_output=True, text=True, check=True
    )
    # Parse the decrypted dotenv output
    from io import StringIO
    return dotenv_values(stream=StringIO(result.stdout))
```

### Secret Scope Separation

Maintain two separate secrets files by scope. This enforces discipline around what talks outside the network and makes rotation easier.

```
secrets/
├── internal.env.enc      # LAN-only services — encrypted at rest
└── external.env.enc      # Anything that leaves the network — encrypted at rest
```

**`secrets/internal.env`** (never committed unencrypted):
```ini
# Paperless-ngx
PAPERLESS_BASE_URL=http://192.168.x.x:8000
PAPERLESS_API_TOKEN=your_token_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Internal queue / DB
QUEUE_DB_PATH=/path/to/queue.db
AUDIT_LOG_PATH=/path/to/audit.log
```

**`secrets/external.env`** (never committed unencrypted):
```ini
# Email (IMAP — Phase 3)
EMAIL_HOST=imap.example.com
EMAIL_USER=you@example.com
EMAIL_PASSWORD=your_password_here

# Google Calendar API (Phase 4)
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REFRESH_TOKEN=your_refresh_token

# Notification channel (Pushover, Telegram, etc.)
NOTIFICATION_TOKEN=your_token_here
```

### Loading Pattern

```python
# corvus/config.py — single source of truth for all config
from corvus.secrets import load_secrets

_internal = load_secrets("secrets/internal.env.enc")
_external = load_secrets("secrets/external.env.enc")

# Internal
PAPERLESS_BASE_URL: str = _internal["PAPERLESS_BASE_URL"]
PAPERLESS_API_TOKEN: str = _internal["PAPERLESS_API_TOKEN"]
OLLAMA_BASE_URL: str = _internal.get("OLLAMA_BASE_URL", "http://localhost:11434")

# External (only loaded when needed)
EMAIL_HOST: str = _external.get("EMAIL_HOST", "")
GOOGLE_CLIENT_ID: str = _external.get("GOOGLE_CLIENT_ID", "")
```

All modules import from `corvus.config`, never from `os.environ` directly. This centralizes secret access and makes it easy to audit what touches what.

### Git Protection

**.gitignore** must include:
```
secrets/*.env          # Never commit plaintext secrets
secrets/*.env.bak
.env
.env.*
!secrets/*.env.enc     # Encrypted files ARE safe to commit
!.sops.yaml            # SOPS config is safe to commit
```

**Pre-commit hook** (`.git/hooks/pre-commit`):
```bash
#!/bin/bash
# Block commits of unencrypted secret files
if git diff --cached --name-only | grep -E 'secrets/.*\.env$|^\.env'; then
    echo "ERROR: Attempt to commit unencrypted secrets file. Aborting."
    exit 1
fi
```

### Operational Rules

1. **Never log secret values.** Sanitize exceptions before writing to the audit log. Use `repr()` carefully — Pydantic models may include field values in their string representation.
2. **Never pass secrets as CLI arguments.** They appear in `ps aux` and shell history.
3. **Never hardcode secrets** in source files, even for tests. Use a separate `secrets/test.env.enc`.
4. **Rotate external credentials** if the machine is ever accessed unexpectedly.
5. **Back up the age private key** out-of-band (e.g., encrypted note in a password manager). Losing `~/.config/sops/age/keys.txt` means losing access to all encrypted secrets.

### Development Without SOPS

For initial dev/testing before SOPS is wired up, a plain `.env` file is acceptable **only if**:
- Permissions are `chmod 600`
- It is in `.gitignore`
- It contains no real external credentials (use placeholder values)

Migrate to SOPS before connecting any real email, calendar, or notification credentials.

---

## Open Questions (Decisions Pending)

- Email provider/protocol (IMAP, Gmail API, etc.)
- Calendar system (Google Calendar API, CalDAV, etc.)
- Notification channel for review queue (email digest, Pushover, Telegram bot, web UI)
- Review/approval interface (web dashboard vs. CLI vs. Telegram)
- STT/TTS final model selection

---

## Project Structure (Recommended)

```
corvus/
├── CLAUDE.md
├── .sops.yaml            # SOPS encryption config (committed)
├── .gitignore
├── secrets/
│   ├── internal.env.enc  # LAN-only credentials (committed, encrypted)
│   └── external.env.enc  # External credentials (committed, encrypted)
├── corvus/
│   ├── config.py         # Single source of truth for all config/secrets
│   ├── secrets.py        # SOPS decrypt helper
│   ├── orchestrator/     # User-facing intake and output
│   ├── planner/          # Reasoning agent(s)
│   ├── router/           # Deterministic Python dispatch logic
│   ├── executors/        # One module per task type
│   │   ├── document_tagger.py
│   │   ├── email_triage.py
│   │   └── invoice_extractor.py
│   ├── schemas/          # All Pydantic models live here
│   ├── integrations/     # Paperless, IMAP, calendar API clients
│   ├── queue/            # Task queue (SQLite or file-based)
│   ├── audit/            # Audit log writer/reader
│   └── digest/           # Daily digest generation
└── tests/
```
