"""Schemas for the scan folder watchdog pipeline (Epic 4).

Covers file detection, transfer, dedup, and audit logging.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TransferMethod(StrEnum):
    """How files are delivered to Paperless-ngx."""

    MOVE = "move"
    UPLOAD = "upload"


class TransferStatus(StrEnum):
    """Outcome of a file transfer attempt."""

    SUCCESS = "success"
    DUPLICATE = "duplicate"
    ERROR = "error"


class WatchdogEvent(BaseModel):
    """An audit record for a single watchdog file event."""

    timestamp: datetime
    source_path: str = Field(description="Original file path in the scan directory")
    file_name: str
    file_hash: str = Field(description="SHA-256 hex digest of the file content")
    transfer_method: TransferMethod
    transfer_status: TransferStatus
    destination: str = Field(
        default="",
        description="Destination path (move) or Paperless task UUID (upload)",
    )
    error_message: str = Field(default="", description="Error details if status is error")
    file_size_bytes: int = Field(default=0, ge=0)
