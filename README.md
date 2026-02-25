# Corvus

Local limited-autonomous agent system for homelab digital chores. Corvus uses local LLM inference (via Ollama) to manage recurring tasks like document tagging, retrieval, and file ingestion, with a human-in-the-loop review system for anything the AI isn't confident about.

Currently focused on **Paperless-ngx document management** (Phase 1).

## What it does

- **Auto-tag scanned documents** -- classifies documents via LLM, suggests tags/correspondents/document types, routes through a confidence gate
- **Natural language document retrieval** -- ask for documents in plain English, get structured search with fallback cascade
- **Scan folder watchdog** -- monitors a directory for new files and ingests them into Paperless (move or API upload, with SHA-256 dedup)
- **Review queue** -- all low-confidence or flagged actions are queued for human approval before anything is changed
- **Daily digest** -- summary of what was auto-applied, what needs review, and what was approved/rejected

## Architecture

```
User (CLI) --> Intent Classifier (LLM)
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

## Installation

```bash
# Clone and set up virtual environment
git clone <repo-url> corvus
cd corvus
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Corvus reads configuration from a `.env` file (for development) or SOPS-encrypted secrets (for production). Create `secrets/internal.env`:

```ini
PAPERLESS_BASE_URL=http://your-paperless-host:8000
PAPERLESS_API_TOKEN=your_api_token_here
OLLAMA_BASE_URL=http://localhost:11434
```

See `CLAUDE.md` for the full secrets management setup with SOPS + age encryption.

## Usage

### Tag documents

Classify and tag untagged documents in Paperless via LLM:

```bash
corvus tag              # Process all untagged documents
corvus tag -n 5         # Process up to 5 documents
corvus tag --all        # Include already-tagged documents
corvus tag -m gemma3    # Use a specific Ollama model
```

### Review queue

Interactively approve, edit, or reject queued tagging suggestions:

```bash
corvus review
```

Each item shows the LLM's suggestions (tags, correspondent, document type, confidence score). Options: **[a]pprove**, **[e]dit** (add tags), **[r]eject**, **[s]kip**, **[q]uit**.

### Fetch a document

Retrieve a document using natural language:

```bash
corvus fetch latest mortgage statement
corvus fetch AT&T invoice from last month
corvus fetch tax documents for 2022 --method download
```

### Ask (single query via orchestrator)

Route a natural language request through the full orchestrator pipeline:

```bash
corvus ask find my most recent mortgage statement
corvus ask what is the status
corvus ask tag my documents
```

### Chat (interactive REPL)

```bash
corvus chat
```

### Watch a scan folder

Monitor a directory and ingest new files into Paperless:

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
  orchestrator/
    router.py                   # Deterministic dispatch (confidence gate)
    pipelines.py                # Reusable pipeline handlers
  planner/
    intent_classifier.py        # LLM intent classification
  executors/
    document_tagger.py          # LLM document classification
    query_interpreter.py        # LLM natural language -> search params
  router/
    tagging.py                  # Name resolution + confidence routing
    retrieval.py                # Search param resolution + fallback cascade
  schemas/
    orchestrator.py             # Intent, pipeline results, response schemas
    document_tagging.py         # Tagging task/result schemas
    document_retrieval.py       # Query interpretation schemas
    paperless.py                # Paperless API data models
    watchdog.py                 # Watchdog event schemas
  integrations/
    paperless.py                # Paperless-ngx REST API client
    ollama.py                   # Ollama REST API client
  queue/
    review.py                   # SQLite-backed review queue
  audit/
    logger.py                   # JSONL append-only audit log
  digest/
    daily.py                    # Daily digest generation + rendering
  watchdog/
    watcher.py                  # Filesystem monitoring (watchdog library)
    transfer.py                 # File move/upload logic
    hash_store.py               # SHA-256 dedup (SQLite)
    audit.py                    # Watchdog-specific audit log
tests/                          # 254 tests (pytest + pytest-asyncio)
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

- **Phase 1** (current): Paperless-ngx document tagging, retrieval, scan folder watchdog
- **Phase 2**: Email inbox management (IMAP, spam filtering, receipt filing, invoice extraction)
- **Phase 3**: Calendar and task integration
- **Future**: STT/TTS voice interface, web dashboard, ComfyUI integration

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) -- free for personal use, research, education, and hobby projects. Commercial use is prohibited. See [LICENSE](LICENSE) for the full terms.
