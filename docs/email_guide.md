# Corvus Email Guide

Corvus manages your email inbox by classifying messages, applying rules, and letting you review what happened. It connects to your accounts via IMAP, uses a local LLM to classify messages it hasn't seen before, and queues proposed actions for your approval.

---

## Setup

### 1. Configure an email account

Create `secrets/email_accounts.json` with your account details:

```json
[
  {
    "name": "Personal",
    "server": "imap.example.com",
    "email": "you@example.com",
    "password": "your-app-password",
    "port": 993,
    "ssl": true,
    "folders": {
      "inbox": "INBOX",
      "receipts": "Corvus/Receipts",
      "approved_ads": "Corvus/Ads",
      "headhunt": "Corvus/Headhunt",
      "processed": "Corvus/Processed"
    }
  }
]
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Display name for the account. |
| `server` | Yes | IMAP server hostname. |
| `email` | Yes | Your email address. |
| `password` | Yes | App password (not your main password). |
| `port` | No | IMAP port. Default: `993`. |
| `ssl` | No | Use SSL. Default: `true`. |
| `is_gmail` | No | Set to `true` for Gmail (uses COPY+DELETE instead of MOVE). |
| `folders` | Yes | Map of logical folder names to IMAP folder paths. |

The `folders` map tells Corvus where to file messages. The keys are logical names you reference in sender list rules (e.g. `receipts`, `approved_ads`). The values are the actual IMAP folder paths on your server. Corvus creates any missing folders automatically.

Verify your config:

```
corvus email accounts
```

### 2. Set up sender lists (recommended)

Sender lists let you define rules for known senders. Messages from listed senders are handled instantly without LLM classification — faster and more predictable.

Corvus starts with no sender lists. You build them over time as you triage and review your inbox.

---

## Daily workflow

The typical flow is: **triage** your inbox, then **review** what Corvus queued.

### Step 1: Triage

```
corvus email triage
```

This connects to your inbox, fetches unread messages, and processes each one:

1. **Sender list check** — if the sender is on a list (other than white), the rule executes immediately (delete, move, or keep). No LLM involved.
2. **Whitelist check** — whitelisted senders still go through LLM classification (so you get summaries and action items), but the action is forced to KEEP and queued for review.
3. **LLM classification** — for unknown senders, Corvus classifies the message into a category (spam, receipt, newsletter, important, action required, etc.), proposes an action, and assigns a confidence score.
4. **Queue** — by default, all LLM-classified actions are queued for your review.

**Output example:**

```
[1/15] Weekly Newsletter from Amazon
  From: deals@amazon.com
  Sender list: vendor -> move

[2/15] Your invoice from Acme Corp
  From: billing@acme.com
  Category: receipt (confidence=92%)
  Summary: Invoice #4521 for $149.00 due 2026-03-15
  Action: move (queued for review)
```

**Options:**

| Flag | Description |
|------|-------------|
| `--account`, `-a` | Process a specific account (by email address). Default: all accounts. |
| `--limit`, `-n` | Max emails to process. Default: 50. |
| `--model`, `-m` | Ollama model name. Auto-detected if omitted. |
| `--apply` | Apply actions immediately instead of queuing. Use with caution. |

### Step 2: Review

```
corvus email review
```

This shows each queued action and lets you decide what to do:

```
--- [1/5] Email from billing@acme.com ---
  Subject: Your invoice from Acme Corp
  Account: you@example.com
  Category: receipt
  Confidence: 92%
  Proposed: move
  Target folder: Corvus/Receipts
  Summary: Invoice #4521 for $149.00 due 2026-03-15
  Reasoning: Contains invoice number, dollar amount, and due date.
  Action [a]pprove / [r]eject / [s]kip / [l]ist / [m]ove / [q]uit:
```

**Review actions:**

| Key | Action | What happens |
|-----|--------|-------------|
| `a` | Approve | Executes the proposed action (delete, move, or keep). |
| `r` | Reject | Discards the proposal. The email stays where it is. |
| `s` | Skip | Leave it in the queue for next time. |
| `l` | Add to list | Add this sender to a sender list. If the list has a non-white action, executes it immediately. |
| `m` | Move | Choose an IMAP folder to move the email to (overrides the proposed action). |
| `q` | Quit | Stop reviewing. Remaining items stay in the queue. |

**Auto-apply on review start:** When you run `corvus email review`, Corvus first checks if any queued senders have been added to a list since they were queued. If so, it auto-applies the list rule before showing you the remaining items.

### Step 3: Check status (optional)

```
corvus email status
```

Shows pipeline activity for the last 24 hours — how many emails are pending review, how many were auto-applied, approved, or rejected.

### Getting a quick summary (optional)

```
corvus email summary
```

Classifies your unread emails and generates a natural-language summary highlighting important messages and action items. Does not move or delete anything.

---

## Sender lists

Sender lists are the core of Corvus's email rules. They provide deterministic, instant handling for known senders — no LLM required.

### How they work

Each list has a **name**, an **action**, and a list of **email addresses**. When Corvus sees an email from a listed sender, it applies the action immediately.

| Action | What happens |
|--------|-------------|
| `keep` | Leave the email in the inbox. Used for the whitelist. |
| `move` | Move the email to a specific folder. Requires a `--folder` key. |
| `delete` | Delete the email. |

Lists are checked in **priority order**. If the same address appears in two lists, the higher-priority list wins.

### Viewing lists

```
corvus email lists
```

Shows all lists, their actions, priority order, and addresses.

### Creating a list

```
corvus email list-create <name> --action <keep|move|delete> [options]
```

**Examples:**

```bash
# Create a list that moves emails to a receipts folder
corvus email list-create finance --action move --folder receipts --description "Finance and billing"

# Create a list that deletes matching emails
corvus email list-create junk --action delete --description "Known junk senders"

# Create a list with auto-cleanup after 90 days
corvus email list-create promo --action move --folder approved_ads --cleanup-days 90
```

| Option | Description |
|--------|-------------|
| `--action` | Required. One of `keep`, `move`, `delete`. |
| `--folder` | Required when action is `move`. Must match a key in your account's `folders` map. |
| `--cleanup-days` | Optional. Automatically delete messages older than N days from the target folder (used with `corvus email cleanup`). |
| `--description` | Optional. Human-readable description of the list. |

### Adding and removing senders

```bash
# Add a sender to a list
corvus email list-add finance billing@acme.com

# Remove a sender from a list
corvus email list-remove finance billing@acme.com
```

You can also add senders during review by pressing `l` at the review prompt.

### Deleting a list

```
corvus email list-delete <name> [--yes]
```

Shows how many addresses are in the list and asks for confirmation. Use `--yes` to skip the prompt.

```
corvus email list-delete promo --yes
```

### Fixing duplicates and conflicts

```
corvus email rationalize
```

Deduplicates addresses within each list and removes addresses that appear in multiple lists (the higher-priority list wins).

### Cleanup

```
corvus email cleanup [--account EMAIL] [--dry-run]
```

Deletes messages older than `cleanup_days` from the target folders of lists that have retention policies. Use `--dry-run` to preview what would be deleted.

---

## The white list

The white list has special behavior. Senders on the white list are **not** skipped by the LLM — they still get classified, summarized, and checked for action items. But their proposed action is always forced to **keep** and queued for review.

This ensures that emails from people you care about are always highlighted in summaries and never auto-deleted or auto-filed.

---

## Folder mapping

Corvus uses logical folder names internally, which map to actual IMAP paths via your account config. This lets you use the same sender list rules across accounts with different folder structures.

**Example:** Your sender list has `folder_key: "receipts"`. In your account config, `"receipts": "Corvus/Receipts"`. When Corvus moves an email from that list, it moves it to `Corvus/Receipts` on the IMAP server.

If a folder doesn't exist, Corvus creates it automatically on first use.

---

## Command reference

| Command | Description |
|---------|-------------|
| `corvus email triage` | Classify and triage unread emails. |
| `corvus email review` | Review queued email actions. |
| `corvus email summary` | Summarize unread emails. |
| `corvus email status` | Show pipeline statistics (last 24h). |
| `corvus email accounts` | List configured email accounts. |
| `corvus email lists` | Display all sender lists. |
| `corvus email list-create` | Create a new sender list. |
| `corvus email list-delete` | Delete a sender list. |
| `corvus email list-add` | Add a sender to a list. |
| `corvus email list-remove` | Remove a sender from a list. |
| `corvus email rationalize` | Dedup and resolve cross-list conflicts. |
| `corvus email cleanup` | Delete old messages from folders with retention policies. |
