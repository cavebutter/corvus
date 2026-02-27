"""Schemas for the voice I/O pipeline (Epic 13–14).

Covers: audio configuration, voice state machine, transcription results,
and wake word events.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class VoiceState(StrEnum):
    """States of the voice pipeline state machine."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class AudioConfig(BaseModel):
    """Audio device and format configuration."""

    input_device: int | None = Field(
        default=None,
        description="sounddevice input device index (None = system default)",
    )
    output_device: int | None = Field(
        default=None,
        description="sounddevice output device index (None = system default)",
    )
    sample_rate_capture: int = Field(
        default=16000,
        description="Capture sample rate in Hz (16 kHz for Whisper)",
    )
    sample_rate_playback: int = Field(
        default=24000,
        description="Playback sample rate in Hz (24 kHz for Kokoro)",
    )
    channels: int = Field(default=1, description="Number of audio channels (mono)")
    chunk_duration_ms: int = Field(
        default=80,
        description="Duration of each audio chunk in milliseconds",
    )

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk at the capture sample rate."""
        return int(self.sample_rate_capture * self.chunk_duration_ms / 1000)


class TranscriptionResult(BaseModel):
    """Structured output from the STT engine."""

    text: str = Field(description="Transcribed text")
    language: str = Field(default="en", description="Detected language code")
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Duration of the audio in seconds"
    )
    no_speech_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability that the audio contains no speech",
    )


class WakeWordEvent(BaseModel):
    """Event emitted when a wake word is detected."""

    keyword: str = Field(description="The detected wake word / model name")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the wake word was detected",
    )
