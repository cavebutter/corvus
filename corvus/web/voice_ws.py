"""WebSocket endpoint for voice interaction over the browser.

Handles the full voice pipeline: receive PCM audio from the browser,
transcribe via faster-whisper, classify intent, dispatch, synthesize
TTS response, and send audio back.

Audio protocol (all binary frames are float32 PCM, mono):
  Client → Server:
    {"type": "audio_start"}          — begin recording
    <binary float32 PCM @ 16 kHz>    — audio chunks
    {"type": "audio_end"}            — stop recording, trigger processing

  Server → Client:
    {"type": "ready"}                — connection established
    {"type": "listening"}            — ack audio_start
    {"type": "transcription", "text": "..."} — STT result
    {"type": "thinking"}             — processing intent
    {"type": "response", "text": "..."} — text response
    {"type": "audio_start", "sample_rate": 24000} — TTS audio follows
    <binary float32 PCM @ 24 kHz>    — TTS audio chunks
    {"type": "audio_end"}            — all TTS audio sent
    {"type": "error", "message": "..."} — error
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from corvus.config import API_KEY

if TYPE_CHECKING:
    from corvus.voice.stt import SpeechRecognizer
    from corvus.voice.tts import SpeechSynthesizer

logger = logging.getLogger(__name__)

# Maximum audio duration in seconds (safety limit)
MAX_AUDIO_SECONDS = 30
SAMPLE_RATE_IN = 16000  # Expected input sample rate from client


class VoiceModels:
    """Lazy-loaded shared STT and TTS models.

    Loads on first use, stays loaded until explicit cleanup.
    Thread-safe via asyncio (single event loop).
    """

    def __init__(self) -> None:
        self._stt: SpeechRecognizer | None = None
        self._tts: SpeechSynthesizer | None = None
        self._loading = False

    async def get_stt(self) -> SpeechRecognizer:
        if self._stt is None:
            await self._load()
        return self._stt  # type: ignore[return-value]

    async def get_tts(self) -> SpeechSynthesizer:
        if self._tts is None:
            await self._load()
        return self._tts  # type: ignore[return-value]

    async def _load(self) -> None:
        if self._loading:
            # Wait for another coroutine to finish loading
            while self._loading:
                await asyncio.sleep(0.1)
            return

        self._loading = True
        try:
            from corvus.config import (
                VOICE_STT_BEAM_SIZE,
                VOICE_STT_COMPUTE_TYPE,
                VOICE_STT_DEVICE,
                VOICE_STT_MODEL,
                VOICE_TTS_LANG_CODE,
                VOICE_TTS_SPEED,
                VOICE_TTS_VOICE,
            )
            from corvus.voice.stt import SpeechRecognizer
            from corvus.voice.tts import SpeechSynthesizer

            logger.info("Loading voice models for WebSocket...")

            stt = SpeechRecognizer(
                model_size=VOICE_STT_MODEL,
                device=VOICE_STT_DEVICE,
                compute_type=VOICE_STT_COMPUTE_TYPE,
                beam_size=VOICE_STT_BEAM_SIZE,
            )
            self._stt = await stt.__aenter__()

            tts = SpeechSynthesizer(
                lang_code=VOICE_TTS_LANG_CODE,
                voice=VOICE_TTS_VOICE,
                speed=VOICE_TTS_SPEED,
            )
            self._tts = await tts.__aenter__()

            logger.info("Voice models loaded")
        finally:
            self._loading = False

    async def cleanup(self) -> None:
        if self._stt is not None:
            await self._stt.__aexit__(None, None, None)
            self._stt = None
        if self._tts is not None:
            await self._tts.__aexit__(None, None, None)
            self._tts = None
        logger.info("Voice models unloaded")


# Singleton — attached to app.state in app.py
voice_models = VoiceModels()


async def voice_websocket_endpoint(websocket: WebSocket) -> None:
    """Handle a voice WebSocket connection.

    Auth is via query param: ws://host/ws/voice?api_key=xxx
    """
    # Auth check
    api_key_param = websocket.query_params.get("api_key", "")
    if not API_KEY or api_key_param != API_KEY:
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return

    await websocket.accept()
    logger.info("Voice WebSocket connected")

    try:
        await websocket.send_json({"type": "ready"})
        await _handle_session(websocket)
    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
    except Exception:
        logger.exception("Voice WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass


async def _handle_session(websocket: WebSocket) -> None:
    """Main session loop — receive audio, process, respond."""
    from corvus.orchestrator.conversation_store import ConversationStore
    from corvus.orchestrator.history import ConversationHistory, summarize_response

    from corvus.config import CONVERSATION_DB_PATH

    store = ConversationStore(CONVERSATION_DB_PATH)
    history = ConversationHistory(max_turns=20, store=store)

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "audio_start":
                await websocket.send_json({"type": "listening"})
                audio_buffer = await _receive_audio(websocket)

                if audio_buffer.size == 0:
                    await websocket.send_json(
                        {"type": "transcription", "text": ""}
                    )
                    continue

                # Transcribe
                stt = await voice_models.get_stt()
                result = await stt.transcribe(audio_buffer)
                text = result.text.strip()
                logger.info("WS transcription: %r", text)

                await websocket.send_json(
                    {"type": "transcription", "text": text}
                )

                if not text:
                    continue

                # Deferred conversation creation
                if history.conversation_id is None:
                    conv_id = store.create(text)
                    history.set_persistence(store, conv_id)

                # Process through orchestrator
                await websocket.send_json({"type": "thinking"})
                response = await _process_text(text, history)

                # Convert response to speech-friendly text
                from corvus.voice.pipeline import _response_to_speech
                speech_text = _response_to_speech(response)

                await websocket.send_json(
                    {"type": "response", "text": speech_text}
                )

                # Synthesize and send TTS audio
                await _send_tts_audio(websocket, speech_text)

                # Update conversation history
                history.add_user_message(text)
                history.add_assistant_message(summarize_response(response))

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    finally:
        store.close()


async def _receive_audio(websocket: WebSocket) -> np.ndarray:
    """Receive binary PCM audio frames until audio_end message.

    Returns float32 numpy array at 16 kHz.
    """
    chunks: list[np.ndarray] = []
    total_samples = 0
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE_IN

    while True:
        message = await websocket.receive()

        if "text" in message:
            import json
            data = json.loads(message["text"])
            if data.get("type") == "audio_end":
                break
            continue

        if "bytes" in message:
            audio = np.frombuffer(message["bytes"], dtype=np.float32)
            total_samples += audio.size
            if total_samples > max_samples:
                logger.warning("Audio exceeded max duration, truncating")
                break
            chunks.append(audio)

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks)


async def _process_text(text: str, history) -> object:
    """Run intent classification and dispatch for transcribed text."""
    import httpx

    from corvus.config import (
        AUDIT_LOG_PATH,
        CHAT_MODEL,
        OLLAMA_BASE_URL,
        PAPERLESS_API_TOKEN,
        PAPERLESS_BASE_URL,
        QUEUE_DB_PATH,
    )
    from corvus.integrations.ollama import OllamaClient
    from corvus.integrations.paperless import PaperlessClient
    from corvus.orchestrator.router import dispatch
    from corvus.planner.intent_classifier import classify_intent

    async with (
        PaperlessClient(PAPERLESS_BASE_URL, PAPERLESS_API_TOKEN) as paperless,
        OllamaClient(OLLAMA_BASE_URL) as ollama,
    ):
        # Auto-detect model
        model = await ollama.pick_instruct_model()
        if model is None:
            from corvus.schemas.orchestrator import OrchestratorAction, OrchestratorResponse
            return OrchestratorResponse(
                action=OrchestratorAction.CHAT_RESPONSE,
                intent=None,
                confidence=0.0,
                message="No models available on the Ollama server.",
            )

        conversation_context = history.get_recent_context(max_turns=5)
        classification, _raw = await classify_intent(
            text,
            ollama=ollama,
            model=model,
            keep_alive="5m",
            conversation_context=conversation_context or None,
        )

        response = await dispatch(
            classification,
            user_input=text,
            paperless=paperless,
            ollama=ollama,
            model=model,
            keep_alive="5m",
            queue_db_path=QUEUE_DB_PATH,
            audit_log_path=AUDIT_LOG_PATH,
            chat_model=CHAT_MODEL or model,
            conversation_history=history.get_messages(),
        )

    return response


async def _send_tts_audio(websocket: WebSocket, text: str) -> None:
    """Synthesize text to speech and send audio over WebSocket."""
    if not text:
        return

    tts = await voice_models.get_tts()
    audio, sample_rate = await tts.synthesize_full(text)

    if audio.size == 0:
        return

    await websocket.send_json(
        {"type": "audio_start", "sample_rate": sample_rate}
    )

    # Send audio in chunks (64KB each) to avoid large single frames
    chunk_size = 16384  # 16K float32 samples = 64KB
    for i in range(0, audio.size, chunk_size):
        chunk = audio[i : i + chunk_size]
        await websocket.send_bytes(chunk.astype(np.float32).tobytes())

    await websocket.send_json({"type": "audio_end"})
