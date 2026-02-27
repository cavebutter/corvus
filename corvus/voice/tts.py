"""Text-to-speech via Kokoro.

SpeechSynthesizer is an async context manager that loads a Kokoro
pipeline on enter and provides streaming ``synthesize()`` and
convenience ``synthesize_full()`` methods.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Async context manager wrapping Kokoro TTS pipeline.

    Usage::

        async with SpeechSynthesizer() as synth:
            async for audio, sr in synth.synthesize("Hello world"):
                play(audio, sr)
    """

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> None:
        self._lang_code = lang_code
        self._voice = voice
        self._speed = speed
        self._pipeline = None

    async def __aenter__(self) -> SpeechSynthesizer:
        from kokoro import KPipeline

        logger.info("Loading Kokoro TTS pipeline (lang=%s, voice=%s)", self._lang_code, self._voice)
        self._pipeline = await asyncio.to_thread(KPipeline, lang_code=self._lang_code)
        logger.info("Kokoro TTS pipeline loaded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._pipeline = None

    async def synthesize(self, text: str) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Synthesize text to audio, yielding chunks per sentence.

        Kokoro's pipeline natively chunks by sentence, so each yield
        is one sentence ready for immediate playback.

        Yields:
            Tuples of (audio_array, sample_rate) for each sentence chunk.
        """
        if self._pipeline is None:
            raise RuntimeError("SpeechSynthesizer not entered as context manager")

        queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue()

        def _generate():
            for _gs, _ps, audio in self._pipeline(
                text, voice=self._voice, speed=self._speed
            ):
                queue.put_nowait((audio, 24000))
            queue.put_nowait(None)  # sentinel

        # Run the blocking Kokoro generator in a thread
        gen_task = asyncio.get_running_loop().run_in_executor(None, _generate)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        # Ensure the thread is done
        await gen_task

    async def synthesize_full(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to a single concatenated audio array.

        Convenience method for short utterances where streaming is not needed.

        Returns:
            Tuple of (concatenated_audio_array, sample_rate).
        """
        chunks: list[np.ndarray] = []
        sample_rate = 24000

        async for audio, sr in self.synthesize(text):
            chunks.append(audio)
            sample_rate = sr

        if not chunks:
            return np.array([], dtype=np.float32), sample_rate

        return np.concatenate(chunks), sample_rate
