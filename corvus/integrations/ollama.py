"""Async client for the Ollama REST API with structured output support."""

import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaResponse(BaseModel):
    """Raw response from Ollama's /api/chat endpoint (non-streaming)."""

    model: str
    message: dict
    done: bool
    done_reason: str = ""
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaClient:
    """Async HTTP client for Ollama with structured output support.

    Uses /api/chat with a JSON schema in the ``format`` parameter
    to constrain LLM output to a Pydantic model.

    Usage::

        async with OllamaClient(base_url) as client:
            result = await client.generate_structured(
                model="gemma3",
                schema_class=MyPydanticModel,
                system="You are a document classifier.",
                prompt="Classify this document: ...",
            )
    """

    def __init__(self, base_url: str, *, default_keep_alive: str = "5m") -> None:
        self._base_url = base_url.rstrip("/")
        self._default_keep_alive = default_keep_alive
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,
        )

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def generate_structured(
        self,
        model: str,
        schema_class: type[T],
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        keep_alive: str | None = None,
    ) -> tuple[T, OllamaResponse]:
        """Generate a structured response constrained to a Pydantic schema.

        Args:
            model: Ollama model name (e.g., "gemma3", "llama3.2").
            schema_class: Pydantic model class. Its JSON schema is sent to Ollama
                as the ``format`` parameter to constrain output.
            system: System prompt with instructions for the LLM.
            prompt: User prompt (e.g., document content to classify).
            temperature: Sampling temperature. Lower = more deterministic.
            keep_alive: How long to keep the model loaded after this request.
                Defaults to the client's default_keep_alive.

        Returns:
            Tuple of (parsed Pydantic model instance, raw OllamaResponse).

        Raises:
            httpx.HTTPStatusError: On non-2xx response from Ollama.
            pydantic.ValidationError: If LLM output doesn't match the schema
                (shouldn't happen with format constraint, but fail loudly).
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema_class.model_json_schema(),
            "stream": False,
            "keep_alive": keep_alive or self._default_keep_alive,
            "options": {
                "temperature": temperature,
            },
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()

        raw = OllamaResponse.model_validate(response.json())
        content = raw.message.get("content", "")

        parsed = schema_class.model_validate(json.loads(content))

        logger.debug(
            "Ollama %s: %d prompt tokens, %d eval tokens, %.1fs total",
            model,
            raw.prompt_eval_count,
            raw.eval_count,
            raw.total_duration / 1e9,
        )

        return parsed, raw

    async def list_models(self) -> list[dict]:
        """List models available on the Ollama server."""
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    async def pick_instruct_model(self) -> str | None:
        """Auto-detect the best instruct/chat model available on the server.

        Prefers models with 'instruct', 'chat', 'qwen', or 'gemma' in the name.
        Falls back to the first available model if none match.

        Returns:
            Model name string, or None if no models are available.
        """
        models = await self.list_models()
        return pick_instruct_model(models)

    async def unload_model(self, model: str) -> None:
        """Explicitly unload a model from VRAM."""
        response = await self._client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [],
                "keep_alive": 0,
            },
        )
        response.raise_for_status()
        logger.info("Unloaded model %s", model)


def pick_instruct_model(models: list[dict]) -> str | None:
    """Select the best instruct/chat model from a list of Ollama models.

    Prefers models with 'instruct', 'chat', 'qwen', or 'gemma' in the name.
    Falls back to the first available model if none match.

    Args:
        models: List of model dicts from Ollama's /api/tags endpoint.

    Returns:
        Model name string, or None if the list is empty.
    """
    for m in models:
        name = m["name"].lower()
        if "instruct" in name or "chat" in name or "qwen" in name or "gemma" in name:
            return m["name"]
    return models[0]["name"] if models else None
