"""Single source of truth for all configuration and secrets.

All modules import from here — never from os.environ directly.

During early development (before SOPS is wired up), set USE_SOPS=false
and provide a plain .env file at secrets/internal.env with chmod 600.
"""

import os
from pathlib import Path

from corvus.secrets import load_dotenv_fallback, load_secrets

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Toggle SOPS vs plain .env (default: plain .env for early dev)
USE_SOPS = os.environ.get("CORVUS_USE_SOPS", "false").lower() == "true"


def _load(scope: str) -> dict[str, str | None]:
    """Load secrets for a given scope (internal or external)."""
    if USE_SOPS:
        return load_secrets(PROJECT_ROOT / f"secrets/{scope}.env.enc")
    return load_dotenv_fallback(PROJECT_ROOT / f"secrets/{scope}.env")


_internal = _load("internal")

# --- Internal (LAN-only) ---
PAPERLESS_BASE_URL: str = _internal.get("PAPERLESS_BASE_URL", "")
PAPERLESS_API_TOKEN: str = _internal.get("PAPERLESS_API_TOKEN", "")
OLLAMA_BASE_URL: str = _internal.get("OLLAMA_BASE_URL", "http://localhost:11434")
QUEUE_DB_PATH: str = _internal.get("QUEUE_DB_PATH", str(PROJECT_ROOT / "data" / "queue.db"))
AUDIT_LOG_PATH: str = _internal.get("AUDIT_LOG_PATH", str(PROJECT_ROOT / "data" / "audit.log"))

# --- Watchdog (scan folder → Paperless intake) ---
WATCHDOG_SCAN_DIR: str = _internal.get("WATCHDOG_SCAN_DIR", "")
WATCHDOG_TRANSFER_METHOD: str = _internal.get("WATCHDOG_TRANSFER_METHOD", "upload")
WATCHDOG_CONSUME_DIR: str = _internal.get("WATCHDOG_CONSUME_DIR", "")
WATCHDOG_FILE_PATTERNS: str = _internal.get(
    "WATCHDOG_FILE_PATTERNS", "*.pdf,*.png,*.jpg,*.jpeg,*.tiff,*.tif"
)
WATCHDOG_AUDIT_LOG_PATH: str = _internal.get(
    "WATCHDOG_AUDIT_LOG_PATH", str(PROJECT_ROOT / "data" / "watchdog_audit.log")
)
WATCHDOG_HASH_DB_PATH: str = _internal.get(
    "WATCHDOG_HASH_DB_PATH", str(PROJECT_ROOT / "data" / "watchdog_hashes.db")
)


# --- Web Search ---
WEB_SEARCH_MAX_RESULTS: int = int(_internal.get("WEB_SEARCH_MAX_RESULTS", "5"))


def load_external() -> dict[str, str | None]:
    """Load external secrets on demand. Only call when needed (Phase 3+)."""
    return _load("external")
