"""Audio capture and playback via sounddevice.

AudioCapture wraps an InputStream as an async context manager, yielding
frames through an asyncio.Queue.  AudioPlayer provides simple async
play/stop around sounddevice blocking calls.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from corvus.schemas.voice import AudioConfig

logger = logging.getLogger(__name__)

# RMS threshold below which audio is considered silence (16-bit normalized to float32)
_SILENCE_RMS_THRESHOLD = 0.01


class AudioCapture:
    """Async context manager wrapping a sounddevice InputStream.

    Usage::

        async with AudioCapture(config) as capture:
            async for frame in capture.frames():
                process(frame)
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> AudioCapture:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate_capture,
            channels=self._config.channels,
            dtype="float32",
            blocksize=self._config.chunk_samples,
            device=self._config.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in a separate thread."""
        if status:
            logger.warning("Audio input status: %s", status)
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, indata.copy()
            )

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield audio frames as they arrive from the microphone."""
        if self._queue is None:
            raise RuntimeError("AudioCapture not entered as context manager")
        while True:
            frame = await self._queue.get()
            yield frame

    def drain(self) -> None:
        """Discard all queued audio frames.

        Call before recording to avoid capturing stale audio
        that accumulated while the pipeline was idle.
        """
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def record_until_silence(
        self,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
    ) -> np.ndarray:
        """Record audio until silence or max duration is reached.

        Args:
            silence_duration: Seconds of continuous silence to stop recording.
            max_duration: Hard ceiling on recording time in seconds.

        Returns:
            Concatenated audio buffer as a float32 numpy array.
        """
        if self._queue is None:
            raise RuntimeError("AudioCapture not entered as context manager")

        chunks: list[np.ndarray] = []
        chunk_seconds = self._config.chunk_duration_ms / 1000.0
        silence_chunks_needed = int(silence_duration / chunk_seconds)
        max_chunks = int(max_duration / chunk_seconds)
        consecutive_silence = 0
        total_chunks = 0

        while total_chunks < max_chunks:
            frame = await self._queue.get()
            chunks.append(frame)
            total_chunks += 1

            rms = _compute_rms(frame)
            if rms < _SILENCE_RMS_THRESHOLD:
                consecutive_silence += 1
            else:
                consecutive_silence = 0

            if consecutive_silence >= silence_chunks_needed and total_chunks > silence_chunks_needed:
                break

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks, axis=0).flatten()


class AudioPlayer:
    """Simple async audio playback wrapper.

    Resamples audio to the output device's native sample rate before
    playback to avoid garbled output on devices that don't handle
    non-native rates well (common with bluetooth speakers).
    """

    def __init__(self) -> None:
        self._playing = False
        self._device_rate: int | None = None

    def _get_device_rate(self) -> int:
        """Query and cache the default output device's native sample rate."""
        if self._device_rate is None:
            import sounddevice as sd

            dev = sd.query_devices(sd.default.device[1])
            self._device_rate = int(dev["default_samplerate"])
        return self._device_rate

    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play an audio array and wait for completion.

        If the audio sample rate differs from the output device's native
        rate, resamples using scipy (linear interpolation fallback if
        scipy is unavailable).
        """
        import sounddevice as sd

        device_rate = self._get_device_rate()
        if sample_rate != device_rate:
            audio = await asyncio.to_thread(_resample, audio, sample_rate, device_rate)
            sample_rate = device_rate

        self._playing = True
        try:
            await asyncio.to_thread(sd.play, audio, sample_rate)
            await asyncio.to_thread(sd.wait)
        finally:
            self._playing = False

    def stop(self) -> None:
        """Stop any current playback immediately."""
        import sounddevice as sd

        sd.stop()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing


def generate_ack_tone(
    duration_ms: int = 200,
    freq_hz: float = 880.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Generate a simple sine wave acknowledgment tone.

    Returns a float32 numpy array ready for playback.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False, dtype=np.float32)
    tone = 0.3 * np.sin(2 * math.pi * freq_hz * t)
    return tone.astype(np.float32)


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate or audio.size == 0:
        return audio
    try:
        from scipy.signal import resample

        num_samples = int(len(audio) * to_rate / from_rate)
        return resample(audio, num_samples).astype(np.float32)
    except ImportError:
        # Fallback: linear interpolation
        ratio = to_rate / from_rate
        num_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, num_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _compute_rms(frame: np.ndarray) -> float:
    """Compute root-mean-square energy of an audio frame."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
