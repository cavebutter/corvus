"""Voice pipeline — main orchestration loop for the voice assistant.

Ties together wake word detection, STT, the existing orchestrator pipeline
(classify_intent -> dispatch -> summarize_response), and TTS playback.
"""

from __future__ import annotations

import asyncio
import logging

from corvus.orchestrator.conversation_store import ConversationStore
from corvus.orchestrator.history import ConversationHistory, summarize_response
from corvus.orchestrator.router import dispatch
from corvus.planner.intent_classifier import classify_intent
from corvus.schemas.email import EmailSummaryResult, EmailTriageResult
from corvus.schemas.orchestrator import OrchestratorAction
from corvus.schemas.voice import AudioConfig, VoiceState
from corvus.voice.audio import AudioCapture, AudioPlayer, generate_ack_tone
from corvus.voice.stt import SpeechRecognizer
from corvus.voice.tts import SpeechSynthesizer
from corvus.voice.wakeword import WakeWordDetector

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Main voice assistant pipeline.

    Manages all voice components and the main event loop:
    wake word -> STT -> classify_intent -> dispatch -> TTS -> play.
    """

    def __init__(
        self,
        *,
        # Orchestrator deps
        ollama,
        paperless,
        model: str,
        keep_alive: str = "30m",
        chat_model: str | None = None,
        queue_db_path: str,
        audit_log_path: str,
        # Voice config
        audio_config: AudioConfig | None = None,
        stt_model: str = "large-v3-turbo",
        stt_device: str = "cuda",
        stt_compute_type: str = "int8_float16",
        stt_beam_size: int = 5,
        tts_lang_code: str = "a",
        tts_voice: str = "af_heart",
        tts_speed: float = 1.0,
        wakeword_model_path: str = "",
        wakeword_threshold: float = 0.5,
        silence_duration: float = 1.5,
        max_listen_duration: float = 30.0,
        # Conversation
        store: ConversationStore | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._ollama = ollama
        self._paperless = paperless
        self._model = model
        self._keep_alive = keep_alive
        self._chat_model = chat_model or model
        self._queue_db_path = queue_db_path
        self._audit_log_path = audit_log_path
        self._silence_duration = silence_duration
        self._max_listen_duration = max_listen_duration

        self._audio_config = audio_config or AudioConfig()
        self._recognizer = SpeechRecognizer(
            model_size=stt_model,
            device=stt_device,
            compute_type=stt_compute_type,
            beam_size=stt_beam_size,
        )
        self._synthesizer = SpeechSynthesizer(
            lang_code=tts_lang_code,
            voice=tts_voice,
            speed=tts_speed,
        )
        self._detector = WakeWordDetector(
            model_path=wakeword_model_path,
            threshold=wakeword_threshold,
        )
        self._player = AudioPlayer()
        self._ack_tone = generate_ack_tone()

        # Conversation persistence
        self._store = store
        self._history: ConversationHistory
        if conversation_id and store:
            self._history = ConversationHistory.from_store(store, conversation_id)
        else:
            self._history = ConversationHistory(max_turns=20, store=store)

        self._state = VoiceState.IDLE
        self._running = False

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def history(self) -> ConversationHistory:
        return self._history

    async def run(self, *, no_wakeword: bool = False) -> None:
        """Run the voice pipeline loop until stopped.

        Args:
            no_wakeword: If True, skip wake word detection and use
                Enter key to trigger listening (for dev/testing).
        """
        self._running = True

        async with (
            self._recognizer,
            self._synthesizer,
            self._detector,
            AudioCapture(self._audio_config) as capture,
        ):
            logger.info("Voice pipeline ready")

            if no_wakeword:
                await self._run_no_wakeword(capture)
            else:
                await self._run_with_wakeword(capture)

    async def _run_with_wakeword(self, capture: AudioCapture) -> None:
        """Main loop with wake word detection."""
        async for frame in capture.frames():
            if not self._running:
                break

            self._state = VoiceState.IDLE
            event = self._detector.process_frame(frame)

            if event is not None:
                self._detector.reset()
                await self._process_utterance(capture)

    async def _run_no_wakeword(self, capture: AudioCapture) -> None:
        """Main loop without wake word — waits for Enter key."""
        loop = asyncio.get_running_loop()

        while self._running:
            self._state = VoiceState.IDLE
            print("\n[Press Enter to speak]", flush=True)
            try:
                await loop.run_in_executor(None, input)
            except (EOFError, KeyboardInterrupt):
                break
            await self._process_utterance(capture)

    async def _process_utterance(self, capture: AudioCapture) -> None:
        """Handle a single wake-to-response cycle."""
        try:
            # 1. Drain stale audio frames from queue before recording
            capture.drain()

            # 2. Play acknowledgment tone
            await self._player.play(self._ack_tone, self._audio_config.sample_rate_playback)
            print("Listening...", flush=True)

            # 3. Record audio until silence
            self._state = VoiceState.LISTENING
            audio = await capture.record_until_silence(
                silence_duration=self._silence_duration,
                max_duration=self._max_listen_duration,
            )

            if audio.size == 0:
                return

            # 4. Transcribe
            self._state = VoiceState.TRANSCRIBING
            result = await self._recognizer.transcribe(audio)
            logger.info("Transcribed: %r", result.text)

            if not result.text.strip():
                print("(no speech detected)", flush=True)
                return

            # Check for stop commands
            text = result.text.strip()
            print(f"You: {text}", flush=True)
            if text.lower() in ("stop", "quit", "exit", "goodbye", "bye"):
                await self._speak("Goodbye.")
                self.stop()
                return

            # 5. Deferred conversation creation
            if self._history.conversation_id is None and self._store:
                conv_id = self._store.create(text)
                self._history.set_persistence(self._store, conv_id)

            # 6. Classify intent
            self._state = VoiceState.PROCESSING
            conversation_context = self._history.get_recent_context(max_turns=5)

            classification, _raw = await classify_intent(
                text,
                ollama=self._ollama,
                model=self._model,
                keep_alive=self._keep_alive,
                conversation_context=conversation_context or None,
            )
            logger.info(
                "Intent: %s (%.0f%%)", classification.intent.value, classification.confidence * 100
            )

            # 7. Dispatch
            response = await dispatch(
                classification,
                user_input=text,
                paperless=self._paperless,
                ollama=self._ollama,
                model=self._model,
                keep_alive=self._keep_alive,
                queue_db_path=self._queue_db_path,
                audit_log_path=self._audit_log_path,
                chat_model=self._chat_model,
                conversation_history=self._history.get_messages(),
            )

            # 8. Convert response to speech-friendly text
            speech_text = _response_to_speech(response)

            # 9. Speak the response
            await self._speak(speech_text)

            # 10. Update conversation history
            self._history.add_user_message(text)
            self._history.add_assistant_message(summarize_response(response))

        except Exception:
            logger.exception("Error processing utterance")
            try:
                await self._speak("Sorry, I ran into an error.")
            except Exception:
                logger.exception("Error speaking error message")

    async def _speak(self, text: str) -> None:
        """Synthesize and play text via TTS.

        Uses synthesize_full() to concatenate all chunks before playback,
        avoiding crackling/popping from gaps between per-sentence sd.play() calls.
        """
        self._state = VoiceState.SPEAKING
        audio, sr = await self._synthesizer.synthesize_full(text)
        if audio.size > 0:
            await self._player.play(audio, sr)

    def stop(self) -> None:
        """Signal the pipeline to stop after the current cycle."""
        self._running = False


def _response_to_speech(response) -> str:
    """Convert an OrchestratorResponse to voice-friendly text."""
    if response.action == OrchestratorAction.CHAT_RESPONSE:
        return response.message

    if response.action == OrchestratorAction.NEEDS_CLARIFICATION:
        return response.message

    if response.action == OrchestratorAction.INTERACTIVE_REQUIRED:
        return "That requires the terminal. Please use the Corvus CLI."

    # Specific handling for email results before generic DISPATCHED
    if response.result is not None:
        if isinstance(response.result, EmailTriageResult):
            r = response.result
            return (
                f"Processed {r.processed} emails. "
                f"{r.auto_acted} auto-applied, {r.queued} queued for review."
            )
        if isinstance(response.result, EmailSummaryResult):
            return response.result.summary

    if response.action == OrchestratorAction.DISPATCHED:
        return summarize_response(response)

    return response.message or "Done."
