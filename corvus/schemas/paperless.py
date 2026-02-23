"""Pydantic models mirroring Paperless-ngx API response shapes."""

from pydantic import BaseModel


class PaperlessTag(BaseModel):
    """A tag from the Paperless-ngx API."""

    id: int
    name: str
    slug: str


class PaperlessCorrespondent(BaseModel):
    """A correspondent from the Paperless-ngx API."""

    id: int
    name: str
    slug: str


class PaperlessDocumentType(BaseModel):
    """A document type from the Paperless-ngx API."""

    id: int
    name: str
    slug: str


class PaperlessDocument(BaseModel):
    """A document from the Paperless-ngx API.

    Only includes fields relevant to the tagging pipeline.
    Extend as needed for other use cases.
    """

    id: int
    title: str
    content: str
    tags: list[int]
    correspondent: int | None = None
    document_type: int | None = None
    created: str
    added: str
    original_filename: str = ""


class PaperlessPaginatedResponse(BaseModel):
    """Paginated list response wrapper from Paperless-ngx API."""

    count: int
    next: str | None = None
    previous: str | None = None
    results: list[dict]
