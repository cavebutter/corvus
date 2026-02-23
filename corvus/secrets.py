"""SOPS decrypt helper for loading encrypted secrets files."""

import subprocess
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values


def load_secrets(encrypted_path: str | Path) -> dict[str, str | None]:
    """Decrypt a SOPS-encrypted .env file and return its key-value pairs.

    Args:
        encrypted_path: Path to the encrypted .env.enc file.

    Returns:
        Dictionary of decrypted key-value pairs.

    Raises:
        FileNotFoundError: If the encrypted file does not exist.
        subprocess.CalledProcessError: If SOPS decryption fails.
    """
    path = Path(encrypted_path)
    if not path.exists():
        raise FileNotFoundError(f"Encrypted secrets file not found: {path}")

    result = subprocess.run(
        ["sops", "--decrypt", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(dotenv_values(stream=StringIO(result.stdout)))


def load_dotenv_fallback(dotenv_path: str | Path) -> dict[str, str | None]:
    """Load a plain .env file directly. For development use only.

    Args:
        dotenv_path: Path to an unencrypted .env file.

    Returns:
        Dictionary of key-value pairs.

    Raises:
        FileNotFoundError: If the .env file does not exist.
    """
    path = Path(dotenv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dotenv file not found: {path}")

    return dict(dotenv_values(path))
