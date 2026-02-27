"""Tests for corvus.voice.audio — AudioCapture, AudioPlayer, generate_ack_tone."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from corvus.schemas.voice import AudioConfig
from corvus.voice.audio import (
    AudioCapture,
    AudioPlayer,
    _compute_rms,
    generate_ack_tone,
)


class TestGenerateAckTone:
    def test_returns_float32_array(self):
        tone = generate_ack_tone()
        assert tone.dtype == np.float32

    def test_correct_length(self):
        tone = generate_ack_tone(duration_ms=200, sample_rate=24000)
        expected_samples = int(24000 * 200 / 1000)
        assert len(tone) == expected_samples

    def test_amplitude_within_bounds(self):
        tone = generate_ack_tone()
        assert np.max(np.abs(tone)) <= 0.31  # 0.3 amplitude + floating point

    def test_custom_frequency(self):
        tone = generate_ack_tone(freq_hz=440.0, duration_ms=100, sample_rate=16000)
        assert len(tone) == 1600

    def test_zero_duration(self):
        tone = generate_ack_tone(duration_ms=0)
        assert len(tone) == 0


class TestComputeRms:
    def test_silence(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert _compute_rms(silence) == 0.0

    def test_loud_signal(self):
        loud = np.ones(1000, dtype=np.float32)
        assert _compute_rms(loud) == pytest.approx(1.0)

    def test_empty_array(self):
        assert _compute_rms(np.array([], dtype=np.float32)) == 0.0

    def test_known_value(self):
        # RMS of a constant 0.5 signal should be 0.5
        signal = np.full(100, 0.5, dtype=np.float32)
        assert _compute_rms(signal) == pytest.approx(0.5, abs=0.001)


class TestAudioCapture:
    @pytest.fixture
    def config(self):
        return AudioConfig(sample_rate_capture=16000, chunk_duration_ms=80)

    @patch("corvus.voice.audio.sd", create=True)
    async def test_context_manager_starts_and_stops_stream(self, mock_sd, config):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        # Patch the import inside the class
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            capture = AudioCapture(config)
            async with capture:
                mock_stream.start.assert_called_once()
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()

    async def test_record_until_silence(self, config):
        """Test that record_until_silence accumulates frames and stops on silence."""
        capture = AudioCapture(config)
        capture._queue = asyncio.Queue()

        # Push some speech frames then silence frames
        speech = np.full((config.chunk_samples, 1), 0.1, dtype=np.float32)
        silence = np.zeros((config.chunk_samples, 1), dtype=np.float32)

        for _ in range(5):
            capture._queue.put_nowait(speech)
        # Need enough silence frames to trigger stop
        silence_chunks = int(0.3 / (config.chunk_duration_ms / 1000)) + 1
        for _ in range(silence_chunks):
            capture._queue.put_nowait(silence)

        result = await capture.record_until_silence(silence_duration=0.3, max_duration=5.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0

    async def test_record_until_max_duration(self, config):
        """Test that recording stops at max_duration."""
        capture = AudioCapture(config)
        capture._queue = asyncio.Queue()

        # Only push speech frames — should stop at max_duration
        speech = np.full((config.chunk_samples, 1), 0.1, dtype=np.float32)
        max_dur = 0.5
        num_chunks = int(max_dur / (config.chunk_duration_ms / 1000))
        for _ in range(num_chunks):
            capture._queue.put_nowait(speech)

        result = await capture.record_until_silence(silence_duration=10.0, max_duration=max_dur)
        assert len(result) > 0

    async def test_record_empty_queue(self, config):
        """Test graceful handling when queue gets exactly 0 frames before max."""
        capture = AudioCapture(config)
        capture._queue = asyncio.Queue()
        # Push only silence immediately
        silence = np.zeros((config.chunk_samples, 1), dtype=np.float32)
        # Need enough silence to trigger, plus some extra
        silence_chunks = int(1.5 / (config.chunk_duration_ms / 1000)) + 2
        for _ in range(silence_chunks):
            capture._queue.put_nowait(silence)

        result = await capture.record_until_silence(silence_duration=1.5, max_duration=30.0)
        assert isinstance(result, np.ndarray)

    async def test_frames_raises_without_context_manager(self, config):
        capture = AudioCapture(config)
        with pytest.raises(RuntimeError, match="not entered"):
            async for _ in capture.frames():
                pass

    async def test_record_raises_without_context_manager(self, config):
        capture = AudioCapture(config)
        with pytest.raises(RuntimeError, match="not entered"):
            await capture.record_until_silence()


class TestAudioPlayer:
    async def test_play_calls_sounddevice(self):
        player = AudioPlayer()
        audio = np.zeros(1000, dtype=np.float32)

        with patch("corvus.voice.audio.sd", create=True) as mock_sd:
            with patch.dict("sys.modules", {"sounddevice": mock_sd}):
                mock_sd.play = MagicMock()
                mock_sd.wait = MagicMock()

                await player.play(audio, 24000)

                assert not player.is_playing

    def test_stop(self):
        player = AudioPlayer()
        player._playing = True

        with patch("corvus.voice.audio.sd", create=True) as mock_sd:
            with patch.dict("sys.modules", {"sounddevice": mock_sd}):
                mock_sd.stop = MagicMock()
                player.stop()
                assert not player.is_playing

    def test_is_playing_default(self):
        player = AudioPlayer()
        assert not player.is_playing
