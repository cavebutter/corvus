# Corvus

Local limited-autonomous agent system for homelab digital chores. Corvus uses local LLM inference (via Ollama) to manage recurring tasks — document tagging, email triage, voice interaction — with a human-in-the-loop review system for anything the AI isn't confident about.

## What it does

### Document management (Paperless-ngx)
- **Auto-tag scanned documents** -- classifies documents via LLM, suggests tags/correspondents/document types, routes through a confidence gate
- **Natural language document retrieval** -- ask for documents in plain English, get structured search with fallback cascade
- **Scan folder watchdog** -- monitors a directory for new files and ingests them into Paperless (move or API upload, with SHA-256 dedup)

### Email inbox management (IMAP)
- **Email triage** -- classifies unread emails (spam, newsletters, receipts, personal, etc.) and applies actions (delete, move, keep, flag)
- **Sender lists** -- deterministic rules for known senders (whitelist, blacklist, vendor, headhunter) that bypass LLM classification
- **Invoice/receipt extraction** -- pulls structured data (vendor, amount, due date) from financial emails
- **Email summary** -- per-account inbox summary with important messages, action items, and category breakdown
- **Multi-account support** -- configure multiple IMAP accounts, run commands against one or all

### Voice interface
- **STT** -- Whisper-based speech-to-text via whisper.cpp
- **TTS** -- Piper neural TTS with voice-optimized response formatting
- **Wake word** -- optional "hey corvus" activation (openWakeWord)

### Cross-cutting
- **Review queue** -- all low-confidence or flagged actions are queued for human approval before anything is changed
- **Daily digest** -- summary of what was auto-applied, what needs review, and what was approved/rejected
- **Audit logging** -- every agent action, decision, and confidence score logged (JSONL, append-only)

## Architecture

```
User (CLI / voice) --> Intent Classifier (LLM)
                            |
                     Orchestrator Router (Python, deterministic)
                          /    \
                 Planner       Executor Agents (LLM, task-specific)
                   |                |
             Structured plans    Structured output (Pydantic JSON)
                   |                |
             Confidence Gate (Python)
                /        \
          Auto-execute   Queue for review
          (> 0.9)        (< 0.9)
```

Key principle: **LLMs classify and extract; Python decides and routes.** All inter-agent communication uses validated Pydantic schemas. No free-form text crosses agent boundaries.

## Requirements

| Component | Spec |
|-----------|------|
| Python | 3.12+ |
| LLM Runtime | [Ollama](https://ollama.com/) running locally |
| Document System | [Paperless-ngx](https://docs.paperless-ngx.com/) instance |
| GPU | NVIDIA GPU recommended (tested on RTX 4090) |
| Voice (optional) | whisper.cpp, Piper TTS, PortAudio (`libportaudio2`) |

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> corvus
cd corvus
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Optional: install voice dependencies
pip install -e ".[voice]"
```

## Configuration

Corvus reads configuration from a `.env` file (for development) or SOPS-encrypted secrets (for production). Create `secrets/internal.env`:

```ini
PAPERLESS_BASE_URL=http://your-paperless-host:8000
PAPERLESS_API_TOKEN=your_api_token_here
OLLAMA_BASE_URL=http://localhost:11434
```

For email, create `secrets/email_accounts.json`:

```json
[
  {
    "name": "My Account",
    "server": "imap.example.com",
    "email": "me@example.com",
    "password": "app-password-here",
    "folders": {
      "inbox": "INBOX",
      "processed": "Processed",
      "receipts": "Receipts"
    },
    "is_gmail": false
  }
]
```

See `CLAUDE.md` for the full secrets management setup with SOPS + age encryption.

## Usage

### Document tagging

```bash
corvus tag              # Process all untagged documents
corvus tag -n 5         # Process up to 5 documents
corvus tag --all        # Include already-tagged documents
corvus tag -m gemma3    # Use a specific Ollama model
```

### Document review

Interactively approve, edit, or reject queued tagging suggestions:

```bash
corvus review
```

Each item shows the LLM's suggestions (tags, correspondent, document type, confidence score). Options: **[a]pprove**, **[e]dit** (add tags), **[r]eject**, **[s]kip**, **[q]uit**.

### Fetch a document

```bash
corvus fetch latest mortgage statement
corvus fetch AT&T invoice from last month
corvus fetch tax documents for 2022 --method download
```

### Email triage

```bash
corvus email triage                          # Triage all accounts
corvus email triage -a me@example.com        # Single account
corvus email triage --limit 10               # Cap messages per account
corvus email triage --force-queue            # Queue everything for review
```

### Email review

```bash
corvus email review
```

Options: **[a]pprove**, **[r]eject**, **[s]kip**, **[l]ist** (add sender to a list), **[m]ove** (pick IMAP folder), **[q]uit**.

### Email summary

```bash
corvus email summary                         # Summarize all accounts
corvus email summary -a me@example.com       # Single account
```

### Sender lists

Deterministic rules applied before LLM classification:

```bash
corvus email lists                           # Show all lists and members
corvus email list-add black spam@junk.com    # Add to blacklist
corvus email list-remove vendor old@co.com   # Remove from a list
corvus email rationalize                     # Remove cross-list duplicates
corvus email cleanup --dry-run               # Preview aged-out message deletion
```

### Email status and accounts

```bash
corvus email status                          # Pipeline statistics
corvus email accounts                        # List configured accounts
```

### Ask (single query via orchestrator)

```bash
corvus ask find my most recent mortgage statement
corvus ask what is the status
corvus ask tag my documents
```

### Chat (interactive REPL)

```bash
corvus chat
```

### Voice assistant

```bash
corvus voice                                 # Start voice session (STT + TTS)
```

### Watch a scan folder

```bash
corvus watch --scan-dir /path/to/scans --method move --consume-dir /path/to/consume
corvus watch --scan-dir /path/to/scans --method upload
corvus watch --scan-dir /path/to/scans --once    # Scan existing files and exit
```

### Status and digest

```bash
corvus status               # Quick overview: pending, processed, reviewed counts
corvus digest               # Full activity digest (last 24h)
corvus digest --hours 48    # Custom lookback period
```

## Project structure

```
corvus/
  cli.py                        # Click CLI entry point
  config.py                     # Centralized config/secrets loading
  secrets.py                    # SOPS decrypt helper
  sender_lists.py               # Deterministic sender-based email rules
  orchestrator/
    router.py                   # Deterministic dispatch (confidence gate)
    pipelines.py                # Document pipeline handlers
    email_pipelines.py          # Email triage + summary pipeline handlers
    history.py                  # Conversation history management
  planner/
    intent_classifier.py        # LLM intent classification
  executors/
    document_tagger.py          # LLM document classification
    query_interpreter.py        # LLM natural language -> search params
    email_classifier.py         # LLM email classification
    email_extractor.py          # LLM invoice/receipt/action item extraction
  router/
    tagging.py                  # Name resolution + confidence routing
    retrieval.py                # Search param resolution + fallback cascade
    email.py                    # Email confidence gate routing
  schemas/
    orchestrator.py             # Intent, pipeline results, response schemas
    document_tagging.py         # Tagging task/result schemas
    document_retrieval.py       # Query interpretation schemas
    email.py                    # Email triage/classification schemas
    sender_lists.py             # Sender list config schemas
    paperless.py                # Paperless API data models
    watchdog.py                 # Watchdog event schemas
  integrations/
    paperless.py                # Paperless-ngx REST API client
    ollama.py                   # Ollama REST API client
    imap.py                     # Async IMAP client (imap-tools)
  queue/
    review.py                   # SQLite-backed review queue (documents)
    email_review.py             # SQLite-backed review queue (email)
  audit/
    logger.py                   # JSONL audit log (documents)
    email_logger.py             # JSONL audit log (email)
  digest/
    daily.py                    # Daily digest generation + rendering
  voice/
    pipeline.py                 # Voice session orchestration
    stt.py                      # Whisper STT
    tts.py                      # Piper TTS
    audio.py                    # Audio I/O (sounddevice)
    wakeword.py                 # Wake word detection (openWakeWord)
  watchdog/
    watcher.py                  # Filesystem monitoring (watchdog library)
    transfer.py                 # File move/upload logic
    hash_store.py               # SHA-256 dedup (SQLite)
    audit.py                    # Watchdog-specific audit log
tests/                          # 605 tests (pytest + pytest-asyncio)
```

## Testing

```bash
pytest                          # Run all fast tests
pytest -m slow                  # Run only live integration tests (requires Ollama + Paperless)
pytest -v                       # Verbose output
```

## Confidence gate

All LLM decisions include a confidence score. The orchestrator routes based on thresholds:

| Confidence | Action | Review |
|------------|--------|--------|
| > 0.9 | Auto-execute | Logged in daily digest |
| 0.7 -- 0.9 | Auto-execute | Flagged in daily digest |
| < 0.7 | Queued only | Requires manual approval |

Current posture: **all actions queued for review** (`--force-queue` default). Thresholds are lowered per-pipeline as trust is established.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Paperless-ngx document tagging, retrieval, scan folder watchdog | Complete |
| 2 | Tiered architecture refactor (planner/executor split, orchestrator router) | Complete |
| 3 | Email inbox management (IMAP, triage, sender lists, invoice extraction) | Complete |
| 4 | Calendar and task integration | Upcoming |
| -- | Voice I/O (STT/TTS, wake word) | Complete |
| -- | Web dashboard, ComfyUI integration | Future |

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) -- free for personal use, research, education, and hobby projects. Commercial use is prohibited. See [LICENSE](LICENSE) for the full terms.
