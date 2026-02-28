"""Schemas for sender list management (Epic 19).

Sender lists allow deterministic, user-defined handling of known email
addresses — bypassing LLM classification entirely for blacklisted senders,
vendors, and headhunters, while ensuring whitelisted humans are always
featured in summaries.
"""

from pydantic import BaseModel, Field


class SenderListConfig(BaseModel):
    """Configuration for a single sender list."""

    description: str = ""
    action: str  # "keep", "move", "delete"
    folder_key: str | None = None  # logical folder name for MOVE actions
    cleanup_days: int | None = None  # auto-delete after N days (None = no cleanup)
    addresses: list[str] = Field(default_factory=list)


class SenderListsFile(BaseModel):
    """Top-level schema for the sender_lists.json file."""

    lists: dict[str, SenderListConfig] = Field(default_factory=dict)
    priority: list[str] = Field(default_factory=list)


class SenderMatch(BaseModel):
    """Result of looking up an address in the sender lists."""

    list_name: str
    address: str  # normalized (lowercased)
    action: str  # "keep", "move", "delete"
    folder_key: str | None = None
    cleanup_days: int | None = None
