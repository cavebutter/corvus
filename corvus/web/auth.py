"""API key authentication for the Corvus web interface."""

from fastapi import HTTPException, Request

from corvus.config import API_KEY


async def require_api_key(request: Request) -> None:
    """FastAPI dependency that validates the X-API-Key header.

    Raises 401 if the key is missing or incorrect.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY not configured on server.",
        )
    key = request.headers.get("X-API-Key")
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
