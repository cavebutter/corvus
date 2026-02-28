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
CONVERSATION_DB_PATH: str = _internal.get(
    "CONVERSATION_DB_PATH", str(PROJECT_ROOT / "data" / "conversations.db")
)

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
WEB_SEARCH_FETCH_PAGES: int = int(_internal.get("WEB_SEARCH_FETCH_PAGES", "2"))
WEB_SEARCH_PAGE_MAX_CHARS: int = int(_internal.get("WEB_SEARCH_PAGE_MAX_CHARS", "8000"))
WEB_SEARCH_FETCH_TIMEOUT: int = int(_internal.get("WEB_SEARCH_FETCH_TIMEOUT", "10"))

# --- Chat ---
CHAT_MODEL: str = _internal.get("CHAT_MODEL", "")


# --- Voice I/O ---
VOICE_STT_MODEL: str = _internal.get("VOICE_STT_MODEL", "large-v3-turbo")
VOICE_STT_COMPUTE_TYPE: str = _internal.get("VOICE_STT_COMPUTE_TYPE", "int8_float16")
VOICE_STT_DEVICE: str = _internal.get("VOICE_STT_DEVICE", "cuda")
VOICE_STT_BEAM_SIZE: int = int(_internal.get("VOICE_STT_BEAM_SIZE", "5"))
VOICE_TTS_VOICE: str = _internal.get("VOICE_TTS_VOICE", "af_heart")
VOICE_TTS_LANG_CODE: str = _internal.get("VOICE_TTS_LANG_CODE", "a")
VOICE_TTS_SPEED: float = float(_internal.get("VOICE_TTS_SPEED", "1.0"))
VOICE_WAKEWORD_MODEL_PATH: str = _internal.get(
    "VOICE_WAKEWORD_MODEL_PATH", "hey_jarvis"
)
VOICE_WAKEWORD_THRESHOLD: float = float(_internal.get("VOICE_WAKEWORD_THRESHOLD", "0.5"))
VOICE_SILENCE_DURATION: float = float(_internal.get("VOICE_SILENCE_DURATION", "1.5"))
VOICE_MAX_LISTEN_DURATION: float = float(_internal.get("VOICE_MAX_LISTEN_DURATION", "30.0"))


# --- Email ---
EMAIL_ACCOUNTS_PATH: str = _internal.get(
    "EMAIL_ACCOUNTS_PATH", str(PROJECT_ROOT / "secrets" / "email_accounts.json")
)
EMAIL_REVIEW_DB_PATH: str = _internal.get(
    "EMAIL_REVIEW_DB_PATH", str(PROJECT_ROOT / "data" / "email_queue.db")
)
EMAIL_AUDIT_LOG_PATH: str = _internal.get(
    "EMAIL_AUDIT_LOG_PATH", str(PROJECT_ROOT / "data" / "email_audit.log")
)
EMAIL_BATCH_SIZE: int = int(_internal.get("EMAIL_BATCH_SIZE", "50"))


def load_email_accounts() -> list[dict]:
    """Load email account configs from JSON file."""
    path = Path(EMAIL_ACCOUNTS_PATH)
    if not path.exists():
        return []
    import json

    return json.loads(path.read_text())


def load_external() -> dict[str, str | None]:
    """Load external secrets on demand. Only call when needed (Phase 3+)."""
    return _load("external")
