"""Speech-to-text via faster-whisper.

SpeechRecognizer is an async context manager that loads a Whisper model
on enter and provides a ``transcribe()`` method returning structured
TranscriptionResult objects.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from corvus.schemas.voice import TranscriptionResult

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Async context manager wrapping faster-whisper for STT.

    Usage::

        async with SpeechRecognizer() as recognizer:
            result = await recognizer.transcribe(audio_buffer)
            print(result.text)
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model = None

    async def __aenter__(self) -> SpeechRecognizer:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model %s (device=%s, compute=%s)",
            self._model_size, self._device, self._compute_type,
        )
        self._model = await asyncio.to_thread(
            WhisperModel,
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model loaded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe an audio buffer to text.

        Args:
            audio: Float32 numpy array of audio samples at 16 kHz.

        Returns:
            TranscriptionResult with transcribed text and metadata.
        """
        if self._model is None:
            raise RuntimeError("SpeechRecognizer not entered as context manager")

        if audio.size == 0:
            return TranscriptionResult(text="", language="en", duration_seconds=0.0)

        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio,
            beam_size=self._beam_size,
            vad_filter=True,
        )

        # Consume the generator to collect all segments
        segment_list = await asyncio.to_thread(list, segments)

        text = " ".join(seg.text.strip() for seg in segment_list if seg.text.strip())

        # Compute aggregate no-speech probability
        no_speech_probs = [seg.no_speech_prob for seg in segment_list]
        avg_no_speech = (
            sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
        )

        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=info.duration,
            no_speech_probability=avg_no_speech,
        )
