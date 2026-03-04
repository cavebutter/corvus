# Corvus Paperless-ngx Guide

Corvus integrates with Paperless-ngx to auto-tag scanned documents. It fetches untagged documents from your Paperless instance, uses a local LLM to suggest tags, correspondents, and document types, and queues the proposals for your review. It can also watch a scan folder for new files and transfer them into Paperless automatically.

---

## Setup

### 1. Configure Paperless credentials

Add your Paperless-ngx connection details to `secrets/internal.env`:

```ini
PAPERLESS_BASE_URL=http://192.168.x.x:8000
PAPERLESS_API_TOKEN=your_token_here
```

Get your API token from the Paperless-ngx admin panel under **Settings > Auth Tokens**, or generate one via the Django shell.

### 2. Configure Ollama

Ollama must be running locally. The default URL is `http://localhost:11434`. Override it if needed:

```ini
OLLAMA_BASE_URL=http://localhost:11434
```

Corvus auto-detects the best available instruct model. To force a specific model, use the `--model` flag on any command.

### 3. Configure the watchdog (optional)

If you want Corvus to monitor a scan folder and transfer files into Paperless automatically, add these to `secrets/internal.env`:

```ini
WATCHDOG_SCAN_DIR=/path/to/scan/folder
WATCHDOG_TRANSFER_METHOD=upload
```

**Transfer methods:**

| Method | How it works | When to use |
|--------|-------------|-------------|
| `upload` | Uploads via the Paperless REST API (`POST /api/documents/post_document/`). | Paperless is on a different machine or you want API-level tracking. |
| `move` | Moves the file into Paperless's consume directory. | Paperless is on the same machine and you want the fastest path. |

If using `move`, also set:

```ini
WATCHDOG_CONSUME_DIR=/path/to/paperless/consume
```

**File patterns** default to `*.pdf,*.png,*.jpg,*.jpeg,*.tiff,*.tif`. Override with:

```ini
WATCHDOG_FILE_PATTERNS=*.pdf,*.png
```

---

## Daily workflow

The typical flow is: **watch** (or manually scan) for new documents, **tag** them, then **review** what Corvus queued.

### Step 1: Tag documents

```
corvus tag
```

This connects to Paperless, fetches untagged documents, and processes each one:

1. **Fetch metadata** -- pulls all existing tags, correspondents, and document types from Paperless so the LLM knows what's available.
2. **LLM classification** -- for each document, sends the title, filename, and first 8,000 characters of OCR content to the LLM. The LLM suggests tags, a correspondent, a document type, and a confidence score for each suggestion.
3. **Confidence gate** -- computes an overall confidence score (weighted average across all suggestions) and assigns a gate action.
4. **Queue** -- by default, all proposals are queued for your review regardless of confidence (`--force-queue` is on by default as the initial safe posture).

**Output example:**

```
Fetching documents from Paperless-ngx...
Found 12 untagged documents.

[1/12] Invoice from Acme Corp - March 2026.pdf
  Tags: invoice, acme-corp (confidence=94%)
  Correspondent: Acme Corp
  Document type: Invoice
  -> Queued for review

[2/12] Homeowners Insurance Policy.pdf
  Tags: insurance, state-farm (confidence=87%)
  Correspondent: State Farm
  Document type: Policy
  -> Queued for review
```

**Options:**

| Flag | Description |
|------|-------------|
| `--limit`, `-n` | Max documents to process. Default: all untagged documents. |
| `--model`, `-m` | Ollama model name. Auto-detected if omitted. |
| `--all` | Include already-tagged documents (re-classify everything). |
| `--keep-alive` | Ollama keep_alive duration. Default: `5m`. |
| `--no-force-queue` | Allow high-confidence items to auto-apply instead of queuing. Use with caution. |

### Step 2: Review

```
corvus review
```

This shows each queued proposal and lets you decide what to do:

```
--- Pending review items: 12 ---

[1/12] Document #142: Invoice from Acme Corp - March 2026.pdf
  Confidence: 94%
  Suggested tags: invoice, acme-corp
  Correspondent: Acme Corp
  Document type: Invoice
  Reasoning: Document contains invoice number, billing address, and line items
    from Acme Corp dated March 2026.
  Content: "INVOICE #10042 — Acme Corp — 123 Main St..."
  Action [a]pprove / [e]dit / [r]eject / [s]kip / [q]uit:
```

**Review actions:**

| Key | Action | What happens |
|-----|--------|-------------|
| `a` | Approve | Applies the suggested tags, correspondent, and document type to the document in Paperless. |
| `e` | Edit | Approve with additional tags. You're prompted for comma-separated tag names. |
| `r` | Reject | Discards the proposal. The document stays untagged. You can add optional notes. |
| `s` | Skip | Leave it in the queue for next time. |
| `q` | Quit | Stop reviewing. Remaining items stay in the queue. |

On approve or edit, Corvus creates any tags, correspondents, or document types that don't already exist in Paperless, then PATCHes the document. New tags are merged with any existing tags on the document.

### Step 3: Check status (optional)

```
corvus status
```

Shows pipeline activity -- how many documents are pending review, how many were processed in the last 24 hours, and how many were reviewed.

### Step 4: Daily digest (optional)

```
corvus digest [--hours 24]
```

Generates a markdown summary of recent activity, broken into sections:

- **Auto-applied** -- items that executed automatically (only when `--no-force-queue` is used).
- **Flagged** -- medium-confidence items that auto-applied but deserve a second look.
- **Queued for review** -- items waiting for your decision.
- **Approved / Rejected** -- items you've already reviewed.
- **Pending count** -- how many are still in the queue.

---

## Confidence gate

Every LLM suggestion includes a confidence score between 0 and 1. Corvus computes an overall confidence as a weighted average across all suggestions (tags, correspondent, document type), then maps it to a gate action:

| Confidence | Gate action | Behavior |
|-----------|-------------|----------|
| >= 0.9 | Auto-execute | Apply immediately, logged in daily digest. |
| 0.7 -- 0.89 | Flag in digest | Apply immediately, flagged for your attention in the digest. |
| < 0.7 | Queue for review | Held for manual approval. |

**Important:** The `--force-queue` flag (on by default) overrides the gate and queues everything regardless of confidence. This is the recommended posture until you've built trust in the LLM's suggestions. Disable it with `--no-force-queue` once you're comfortable.

---

## Watching a scan folder

### Continuous watch

```
corvus watch
```

Monitors your scan folder using filesystem events (inotify). When a new file appears:

1. Computes a SHA-256 hash of the file.
2. Checks the hash store for duplicates. If it's been seen before, logs it as a duplicate and skips.
3. Transfers the file to Paperless (upload or move, depending on config).
4. Records the hash to prevent future duplicates.
5. Logs the event to the watchdog audit log.

Runs until you press Ctrl+C.

### One-shot scan

```
corvus watch --once
```

Scans all matching files currently in the folder, transfers them, and exits with a summary:

```
Scan complete: 5 transferred, 1 duplicate, 0 errors.
```

**Options:**

| Flag | Description |
|------|-------------|
| `--scan-dir` | Directory to watch. Default: `WATCHDOG_SCAN_DIR` from config. |
| `--method` | Transfer method: `upload` or `move`. Default: `WATCHDOG_TRANSFER_METHOD` from config. |
| `--consume-dir` | Paperless consume directory (required for `move` method). |
| `--patterns` | Comma-separated file patterns. Default: `*.pdf,*.png,*.jpg,*.jpeg,*.tiff,*.tif`. |
| `--once` | Scan existing files once and exit instead of watching continuously. |

### Duplicate detection

The watchdog uses a SQLite-backed hash store (`data/watchdog_hashes.db`) to track every file it has transferred. If you scan the same document twice (even with a different filename), Corvus recognizes the duplicate by its SHA-256 hash and skips it.

---

## Fetching documents

### Quick fetch

```
corvus fetch "my acme invoice from last month"
```

Searches Paperless using natural language. Corvus interprets your query via the LLM, resolves names to Paperless IDs, and searches. If the initial search returns no results, it automatically tries broader fallback queries.

Results are displayed as a numbered list. Pick one to open it in your browser or download it.

**Options:**

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Ollama model name. |
| `--method` | `browser` (default) or `download`. |
| `--download-dir` | Where to save downloaded files. Default: `~/Downloads`. |
| `--keep-alive` | Ollama keep_alive duration. Default: `5m`. |

### Through the orchestrator

```
corvus ask "find my latest insurance policy"
```

The `ask` command classifies your intent first. If it's a document fetch, it runs the same pipeline as `corvus fetch`. This also works in `corvus chat` for interactive sessions.

---

## Chat and voice

### Interactive chat

```
corvus chat
```

An interactive REPL that maintains conversation history (stored in SQLite). Each input is classified by intent and dispatched through the orchestrator -- so you can tag documents, fetch files, check status, or just have a conversation, all from one prompt.

**Options:**

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Ollama model name. |
| `--keep-alive` | Ollama keep_alive duration. Default: `10m`. |
| `--new` | Start a fresh conversation (ignore previous history). |
| `--list` | Show recent conversations and exit. |
| `--resume ID` | Resume a specific conversation by ID prefix. |

### Intent classification

Every input (whether from `ask`, `chat`, or a future voice interface) is classified into one of these intents:

| Intent | Dispatched to |
|--------|---------------|
| Tag documents | `run_tag_pipeline()` |
| Fetch document | `run_fetch_pipeline()` |
| Review queue | Interactive review (CLI only) |
| Show digest | `run_digest_pipeline()` |
| Show status | `run_status_pipeline()` |
| Watch folder | Interactive watch (CLI only) |
| General chat | Free-form LLM conversation |

If the intent classifier's confidence is below 0.7, Corvus asks for clarification instead of guessing.

---

## Maintenance

```
corvus maintain [--days 90] [--dry-run]
```

Purges old data from all stores:

- Document audit log (`data/audit.log`)
- Watchdog audit log (`data/watchdog_audit.log`)
- Document review queue (`data/queue.db`)

Entries older than the retention period (default: 90 days, configurable via `RETENTION_DAYS`) are permanently deleted. Resolved review queue items (approved, rejected, modified) are purged; pending items are always kept.

Use `--dry-run` to preview what would be deleted without actually deleting.

---

## Data files

Corvus stores its working data in the `data/` directory:

| File | Purpose |
|------|---------|
| `data/queue.db` | SQLite review queue for document tagging proposals. |
| `data/audit.log` | JSONL audit log of all tagging actions (auto-applied, queued, approved, rejected). |
| `data/watchdog_audit.log` | JSONL audit log of watchdog file transfers. |
| `data/watchdog_hashes.db` | SQLite hash store for duplicate detection. |
| `data/conversations.db` | SQLite store for chat conversation history. |

All paths are configurable via `secrets/internal.env`.

---

## Configuration reference

All variables go in `secrets/internal.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERLESS_BASE_URL` | *(required)* | Paperless-ngx server URL. |
| `PAPERLESS_API_TOKEN` | *(required)* | API authentication token. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `QUEUE_DB_PATH` | `data/queue.db` | Review queue database path. |
| `AUDIT_LOG_PATH` | `data/audit.log` | Tagging audit log path. |
| `WATCHDOG_SCAN_DIR` | *(empty)* | Scan folder to watch for new files. |
| `WATCHDOG_TRANSFER_METHOD` | `upload` | How to transfer files: `upload` or `move`. |
| `WATCHDOG_CONSUME_DIR` | *(empty)* | Paperless consume directory (for `move` method). |
| `WATCHDOG_FILE_PATTERNS` | `*.pdf,*.png,*.jpg,...` | Comma-separated file patterns to match. |
| `WATCHDOG_AUDIT_LOG_PATH` | `data/watchdog_audit.log` | Watchdog audit log path. |
| `WATCHDOG_HASH_DB_PATH` | `data/watchdog_hashes.db` | Hash store database path. |
| `RETENTION_DAYS` | `90` | Days to keep audit logs and resolved queue items. |

---

## Command reference

| Command | Description |
|---------|-------------|
| `corvus tag` | Classify and tag untagged documents. |
| `corvus review` | Review queued tagging proposals. |
| `corvus digest` | Generate a digest of recent activity. |
| `corvus status` | Show pipeline statistics. |
| `corvus watch` | Watch a scan folder and transfer new files to Paperless. |
| `corvus fetch` | Search for and retrieve a document by natural language query. |
| `corvus ask` | Ask a question or give a command via the orchestrator. |
| `corvus chat` | Interactive conversation with full orchestrator routing. |
| `corvus maintain` | Purge old audit logs and resolved review queue items. |
