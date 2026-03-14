"""FastAPI application for the Corvus PWA.

Serves the REST API, WebSocket voice endpoint, and static frontend assets.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from corvus.web.routes import router
from corvus.web.voice_ws import voice_models, voice_websocket_endpoint

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared resources."""
    yield
    # Clean up voice models if they were loaded
    await voice_models.cleanup()


app = FastAPI(
    title="Corvus",
    description="Local AI agent system — PWA interface",
    lifespan=lifespan,
)

# CORS — allow Cloudflare Tunnel origins and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tightened per-deployment via config if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health (public, no auth) ─────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Voice WebSocket (auth via query param) ───────────────────────────

@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    """Voice interaction over WebSocket."""
    await voice_websocket_endpoint(websocket)


# ── Protected API routes (from router) ───────────────────────────────

app.include_router(router)


# ── Static files (frontend) ─────────────────────────────────────────
# Mounted last so /api/* routes take precedence.

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
