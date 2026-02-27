"""Tests for corvus.schemas.voice — validation, defaults, constraints."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from corvus.schemas.voice import (
    AudioConfig,
    TranscriptionResult,
    VoiceState,
    WakeWordEvent,
)


class TestVoiceState:
    def test_all_states_exist(self):
        assert VoiceState.IDLE == "idle"
        assert VoiceState.LISTENING == "listening"
        assert VoiceState.TRANSCRIBING == "transcribing"
        assert VoiceState.PROCESSING == "processing"
        assert VoiceState.SPEAKING == "speaking"

    def test_state_count(self):
        assert len(VoiceState) == 5


class TestAudioConfig:
    def test_defaults(self):
        config = AudioConfig()
        assert config.input_device is None
        assert config.output_device is None
        assert config.sample_rate_capture == 16000
        assert config.sample_rate_playback == 24000
        assert config.channels == 1
        assert config.chunk_duration_ms == 80

    def test_chunk_samples(self):
        config = AudioConfig(sample_rate_capture=16000, chunk_duration_ms=80)
        assert config.chunk_samples == 1280

    def test_chunk_samples_different_rate(self):
        config = AudioConfig(sample_rate_capture=48000, chunk_duration_ms=100)
        assert config.chunk_samples == 4800

    def test_custom_devices(self):
        config = AudioConfig(input_device=2, output_device=3)
        assert config.input_device == 2
        assert config.output_device == 3


class TestTranscriptionResult:
    def test_defaults(self):
        result = TranscriptionResult(text="hello world")
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.duration_seconds == 0.0
        assert result.no_speech_probability == 0.0

    def test_full_construction(self):
        result = TranscriptionResult(
            text="test phrase",
            language="fr",
            duration_seconds=2.5,
            no_speech_probability=0.1,
        )
        assert result.language == "fr"
        assert result.duration_seconds == 2.5
        assert result.no_speech_probability == 0.1

    def test_no_speech_probability_validation(self):
        with pytest.raises(ValidationError):
            TranscriptionResult(text="x", no_speech_probability=1.5)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptionResult(text="x", duration_seconds=-1.0)


class TestWakeWordEvent:
    def test_construction(self):
        event = WakeWordEvent(keyword="corvus", confidence=0.95)
        assert event.keyword == "corvus"
        assert event.confidence == 0.95
        assert isinstance(event.timestamp, datetime)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            WakeWordEvent(keyword="corvus", confidence=1.5)
        with pytest.raises(ValidationError):
            WakeWordEvent(keyword="corvus", confidence=-0.1)
