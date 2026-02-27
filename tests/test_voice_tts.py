"""Tests for corvus.voice.tts — SpeechSynthesizer with mocked Kokoro."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corvus.voice.tts import SpeechSynthesizer


def _make_pipeline_callable(chunks):
    """Create a callable that mimics KPipeline.__call__."""

    def pipeline_call(text, voice=None, speed=None):
        for i, chunk in enumerate(chunks):
            yield (f"gs_{i}", f"ps_{i}", chunk)

    return pipeline_call


class TestSpeechSynthesizer:
    async def test_synthesize_yields_chunks(self):
        """Test that synthesize yields audio chunks per sentence."""
        chunk1 = np.random.randn(24000).astype(np.float32)
        chunk2 = np.random.randn(12000).astype(np.float32)

        synth = SpeechSynthesizer(voice="af_heart")
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = _make_pipeline_callable([chunk1, chunk2])
        synth._pipeline = mock_pipeline

        chunks = []
        async for audio, sr in synth.synthesize("Hello world. How are you?"):
            chunks.append((audio, sr))

        assert len(chunks) == 2
        assert chunks[0][1] == 24000
        assert chunks[1][1] == 24000
        np.testing.assert_array_equal(chunks[0][0], chunk1)
        np.testing.assert_array_equal(chunks[1][0], chunk2)

    async def test_synthesize_full_concatenates(self):
        """Test that synthesize_full returns a single concatenated array."""
        chunk1 = np.random.randn(1000).astype(np.float32)
        chunk2 = np.random.randn(2000).astype(np.float32)

        synth = SpeechSynthesizer()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = _make_pipeline_callable([chunk1, chunk2])
        synth._pipeline = mock_pipeline

        audio, sr = await synth.synthesize_full("Hello world. How are you?")

        assert sr == 24000
        assert len(audio) == 3000
        np.testing.assert_array_equal(audio[:1000], chunk1)
        np.testing.assert_array_equal(audio[1000:], chunk2)

    async def test_synthesize_full_empty_text(self):
        """Test synthesize_full with text that produces no output."""
        synth = SpeechSynthesizer()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = _make_pipeline_callable([])
        synth._pipeline = mock_pipeline

        audio, sr = await synth.synthesize_full("")

        assert sr == 24000
        assert len(audio) == 0

    async def test_synthesize_raises_without_context_manager(self):
        """Test that synthesize raises if pipeline not loaded."""
        synth = SpeechSynthesizer()
        with pytest.raises(RuntimeError, match="not entered"):
            async for _ in synth.synthesize("hello"):
                pass

    async def test_synthesize_uses_configured_voice(self):
        """Test that the configured voice is passed to the pipeline."""
        synth = SpeechSynthesizer(voice="bf_emma", speed=1.2)
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = _make_pipeline_callable(
            [np.zeros(100, dtype=np.float32)]
        )
        synth._pipeline = mock_pipeline

        async for _ in synth.synthesize("test"):
            pass

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("voice") == "bf_emma" or call_kwargs[1].get("voice") == "bf_emma"
        assert call_kwargs.kwargs.get("speed") == 1.2 or call_kwargs[1].get("speed") == 1.2

    async def test_synthesize_single_chunk(self):
        """Test with a single sentence producing one chunk."""
        chunk = np.random.randn(500).astype(np.float32)
        synth = SpeechSynthesizer()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = _make_pipeline_callable([chunk])
        synth._pipeline = mock_pipeline

        chunks = []
        async for audio, sr in synth.synthesize("Hello."):
            chunks.append(audio)

        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], chunk)

    async def test_lifecycle_cleanup(self):
        """Test that __aexit__ clears the pipeline."""
        synth = SpeechSynthesizer()
        synth._pipeline = MagicMock()

        await synth.__aexit__(None, None, None)

        assert synth._pipeline is None
