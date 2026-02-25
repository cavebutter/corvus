"""Async client for the Paperless-ngx REST API."""

import asyncio
import logging
from pathlib import Path

import httpx

from httpx import RemoteProtocolError

from corvus.schemas.paperless import (
    PaperlessCorrespondent,
    PaperlessDocument,
    PaperlessDocumentType,
    PaperlessTag,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 1.0


async def _retry_on_disconnect(coro_fn, *args, **kwargs):
    """Retry an async call on ``RemoteProtocolError`` (server disconnect).

    Retries up to ``MAX_RETRIES`` times with a fixed delay between attempts.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except RemoteProtocolError:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Connection dropped (attempt %d/%d), retrying...",
                attempt + 1,
                MAX_RETRIES,
            )
            await asyncio.sleep(RETRY_DELAY)


class PaperlessClient:
    """Async HTTP client for Paperless-ngx.

    Usage::

        async with PaperlessClient(base_url, token) as client:
            docs = await client.list_documents()
    """

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Token {token}",
                "Accept": "application/json; version=5",
            },
            timeout=30.0,
        )

    async def __aenter__(self) -> "PaperlessClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET a JSON endpoint. Retries on connection drop."""
        return await _retry_on_disconnect(self._get_raw, path, params=params)

    async def _get_raw(self, path: str, params: dict | None = None) -> dict:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def _patch(self, path: str, payload: dict) -> dict:
        """PATCH a JSON endpoint. Retries on connection drop."""
        return await _retry_on_disconnect(self._patch_raw, path, payload)

    async def _patch_raw(self, path: str, payload: dict) -> dict:
        response = await self._client.patch(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def _get_all_pages(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages of a paginated endpoint."""
        params = dict(params) if params else {}
        results: list[dict] = []
        page = 1

        while True:
            params["page"] = page
            data = await self._get(path, params=params)
            results.extend(data["results"])
            if data.get("next") is None:
                break
            page += 1

        return results

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def list_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        ordering: str = "-added",
        filter_params: dict | None = None,
    ) -> tuple[list[PaperlessDocument], int]:
        """Fetch a single page of documents.

        Args:
            page: Page number (1-indexed).
            page_size: Number of documents per page.
            ordering: Sort order field.
            filter_params: Additional query parameters for filtering
                (e.g. ``{"tags__isnull": True}`` to fetch untagged documents).

        Returns:
            Tuple of (documents, total_count).
        """
        params: dict = {"page": page, "page_size": page_size, "ordering": ordering}
        if filter_params:
            params.update(filter_params)
        return await _retry_on_disconnect(self._list_documents_raw, params)

    async def _list_documents_raw(self, params: dict) -> tuple[list[PaperlessDocument], int]:
        response = await self._client.get("/api/documents/", params=params)
        # Paperless returns 404 for pages beyond the last — treat as empty.
        if response.status_code == 404:
            return [], 0
        response.raise_for_status()
        data = response.json()
        docs = [PaperlessDocument.model_validate(r) for r in data["results"]]
        return docs, data["count"]

    async def get_document(self, doc_id: int) -> PaperlessDocument:
        """Fetch a single document by ID."""
        data = await self._get(f"/api/documents/{doc_id}/")
        return PaperlessDocument.model_validate(data)

    async def upload_document(
        self,
        file_path: str | Path,
        *,
        title: str | None = None,
    ) -> str:
        """Upload a document to Paperless-ngx for ingestion.

        Uses ``POST /api/documents/post_document/`` with multipart form data.
        Paperless returns the task UUID as plain text on success.

        Args:
            file_path: Path to the file to upload.
            title: Optional title override. Defaults to the filename.

        Returns:
            The Paperless task UUID (string).
        """
        return await _retry_on_disconnect(self._upload_document_raw, file_path, title=title)

    async def _upload_document_raw(
        self, file_path: str | Path, *, title: str | None = None
    ) -> str:
        path = Path(file_path)
        data = {}
        if title:
            data["title"] = title

        with path.open("rb") as f:
            response = await self._client.post(
                "/api/documents/post_document/",
                data=data,
                files={"document": (path.name, f, "application/octet-stream")},
            )
        response.raise_for_status()
        return response.text.strip()

    async def download_document(self, doc_id: int, dest_path: str | Path) -> Path:
        """Download a document's original file from Paperless-ngx.

        Uses ``GET /api/documents/{id}/download/`` to fetch the original file.

        Args:
            doc_id: Paperless document ID.
            dest_path: Directory or file path to save to. If a directory,
                the filename from the Content-Disposition header is used.

        Returns:
            Path to the downloaded file.
        """
        return await _retry_on_disconnect(self._download_document_raw, doc_id, dest_path)

    async def _download_document_raw(self, doc_id: int, dest_path: str | Path) -> Path:
        dest = Path(dest_path)
        response = await self._client.get(f"/api/documents/{doc_id}/download/")
        response.raise_for_status()

        if dest.is_dir():
            # Extract filename from Content-Disposition header
            cd = response.headers.get("content-disposition", "")
            filename = f"document_{doc_id}"
            if "filename=" in cd:
                # Parse: attachment; filename="somefile.pdf"
                parts = cd.split("filename=")
                if len(parts) > 1:
                    filename = parts[1].strip().strip('"')
            dest = dest / filename

        dest.write_bytes(response.content)
        logger.info("Downloaded document %d to %s", doc_id, dest)
        return dest

    async def update_document(self, doc_id: int, payload: dict) -> PaperlessDocument:
        """PATCH a document. Only include fields you want to change.

        Example payload: {"tags": [1, 3], "correspondent": 5}
        """
        data = await self._patch(f"/api/documents/{doc_id}/", payload)
        return PaperlessDocument.model_validate(data)

    def get_document_url(self, doc_id: int) -> str:
        """Return the browser URL for viewing a document in the Paperless-ngx UI."""
        return f"{self._base_url}/documents/{doc_id}/details"

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    async def list_tags(self) -> list[PaperlessTag]:
        """Fetch all tags (all pages)."""
        results = await self._get_all_pages("/api/tags/")
        return [PaperlessTag.model_validate(r) for r in results]

    async def create_tag(self, name: str) -> PaperlessTag:
        """Create a new tag."""
        response = await self._client.post("/api/tags/", json={"name": name})
        response.raise_for_status()
        return PaperlessTag.model_validate(response.json())

    # ------------------------------------------------------------------
    # Correspondents
    # ------------------------------------------------------------------

    async def list_correspondents(self) -> list[PaperlessCorrespondent]:
        """Fetch all correspondents (all pages)."""
        results = await self._get_all_pages("/api/correspondents/")
        return [PaperlessCorrespondent.model_validate(r) for r in results]

    async def create_correspondent(self, name: str) -> PaperlessCorrespondent:
        """Create a new correspondent."""
        response = await self._client.post("/api/correspondents/", json={"name": name})
        response.raise_for_status()
        return PaperlessCorrespondent.model_validate(response.json())

    # ------------------------------------------------------------------
    # Document Types
    # ------------------------------------------------------------------

    async def list_document_types(self) -> list[PaperlessDocumentType]:
        """Fetch all document types (all pages)."""
        results = await self._get_all_pages("/api/document_types/")
        return [PaperlessDocumentType.model_validate(r) for r in results]

    async def create_document_type(self, name: str) -> PaperlessDocumentType:
        """Create a new document type."""
        response = await self._client.post("/api/document_types/", json={"name": name})
        response.raise_for_status()
        return PaperlessDocumentType.model_validate(response.json())
