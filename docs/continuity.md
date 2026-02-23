# Corvus — Session Continuity Notes

> Updated at the end of each session or before conversation compacting. Read at the start of each new session.

---

## Session: 2026-02-23

### What was done
- Read and reviewed CLAUDE.md (full project spec)
- Added Git Policy and Project Documents sections to CLAUDE.md
- Added venv location to CLAUDE.md Tech Stack section
- Created `docs/backlog_current.md`, `backlog_archive.md`, `continuity.md`
- **Completed Epic 1: Project Scaffolding** (all 6 stories):
  - Created full package directory structure with `__init__.py` files
  - Created `pyproject.toml` — pydantic, httpx, python-dotenv, pytest, pytest-asyncio, ruff
  - Implemented `corvus/secrets.py` (SOPS decrypt + dotenv fallback) and `corvus/config.py`
  - Expanded `.gitignore` to full ruleset
  - Created pre-commit hook blocking unencrypted secrets
  - Set up `tests/conftest.py` with `_no_sops` fixture
  - Created `secrets/internal.env` (placeholder, chmod 600)
  - Installed all deps into venv via `pip install -e ".[dev]"`
- Verified: config loads correctly, ruff passes clean, pytest discovers test dir

### Current state
- **Epic 1 complete.** All scaffolding in place, not yet committed.
- **Venv** at `~/virtual-envs/corvus/.venv` — Python 3.12.3, all deps installed.
- **Ruff** configured: py312, 100 char line length, E/W/F/I/N/UP/B/SIM/RUF rules.
- **Config** defaults to plain `.env` fallback (set `CORVUS_USE_SOPS=true` for production).

### Decisions made
- Git operations (add/commit/push) are user-only; Claude writes commit messages.
- Ruff for linting (user-requested).
- Dev-mode config uses plain dotenv; SOPS toggle via `CORVUS_USE_SOPS` env var.

### Blockers
- None.

### Next steps
- User to commit the scaffolding.
- Begin Epic 2: Paperless-ngx Document Tagging (start with S2.1 Pydantic schemas).
