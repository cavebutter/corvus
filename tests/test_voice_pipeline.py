"""Tests for corvus.voice.pipeline — VoicePipeline with mocked components."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from corvus.schemas.orchestrator import (
    FetchPipelineResult,
    Intent,
    IntentClassification,
    OrchestratorAction,
    OrchestratorResponse,
    StatusResult,
    TagPipelineResult,
)
from corvus.schemas.voice import AudioConfig, TranscriptionResult
from corvus.voice.pipeline import VoicePipeline, _response_to_speech


# --- Fixtures ---


@pytest.fixture
def mock_ollama():
    return AsyncMock()


@pytest.fixture
def mock_paperless():
    return AsyncMock()


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.create.return_value = "test-conv-id"
    store.load_messages.return_value = []
    return store


@pytest.fixture
def pipeline_kwargs(mock_ollama, mock_paperless, mock_store, tmp_path):
    return dict(
        ollama=mock_ollama,
        paperless=mock_paperless,
        model="test-model",
        keep_alive="5m",
        queue_db_path=str(tmp_path / "queue.db"),
        audit_log_path=str(tmp_path / "audit.log"),
        wakeword_model_path="",  # disabled
        store=mock_store,
    )


# --- _response_to_speech tests ---


class TestResponseToSpeech:
    def test_chat_response(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.CHAT_RESPONSE,
            message="Hello there!",
        )
        assert _response_to_speech(response) == "Hello there!"

    def test_needs_clarification(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.NEEDS_CLARIFICATION,
            message="Could you rephrase?",
        )
        assert _response_to_speech(response) == "Could you rephrase?"

    def test_interactive_required(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.INTERACTIVE_REQUIRED,
            message="Use corvus review directly.",
        )
        assert _response_to_speech(response) == "That requires the terminal. Please use the Corvus CLI."

    def test_dispatched_tag(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.TAG_DOCUMENTS,
            result=TagPipelineResult(processed=3, queued=2, auto_applied=1, errors=0),
        )
        text = _response_to_speech(response)
        assert "3" in text
        assert "2" in text

    def test_dispatched_fetch(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.FETCH_DOCUMENT,
            result=FetchPipelineResult(
                documents_found=1,
                documents=[{"title": "Tax Return 2024", "id": 42}],
            ),
        )
        text = _response_to_speech(response)
        assert "1" in text
        assert "Tax Return 2024" in text

    def test_dispatched_status(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            intent=Intent.SHOW_STATUS,
            result=StatusResult(pending_count=5, processed_24h=10, reviewed_24h=3),
        )
        text = _response_to_speech(response)
        assert "5" in text

    def test_dispatched_no_result(self):
        response = OrchestratorResponse(
            action=OrchestratorAction.DISPATCHED,
            message="All done.",
        )
        assert _response_to_speech(response) == "All done."


class TestVoicePipeline:
    def test_construction(self, pipeline_kwargs):
        pipeline = VoicePipeline(**pipeline_kwargs)
        assert pipeline.state == "idle"
        assert pipeline.history is not None

    def test_construction_with_conversation_id(self, pipeline_kwargs, mock_store):
        mock_store.load_messages.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pipeline = VoicePipeline(**pipeline_kwargs, conversation_id="existing-id")
        assert pipeline.history.conversation_id == "existing-id"

    def test_stop(self, pipeline_kwargs):
        pipeline = VoicePipeline(**pipeline_kwargs)
        pipeline._running = True
        pipeline.stop()
        assert not pipeline._running

    @patch("corvus.voice.pipeline.classify_intent")
    @patch("corvus.voice.pipeline.dispatch")
    async def test_process_utterance_full_cycle(
        self, mock_dispatch, mock_classify, pipeline_kwargs
    ):
        """Test a single utterance processing cycle with mocked components."""
        # Setup classification
        classification = IntentClassification(
            intent=Intent.GENERAL_CHAT,
            confidence=0.95,
            reasoning="general chat",
        )
        mock_classify.return_value = (classification, MagicMock())

        # Setup dispatch response
        response = OrchestratorResponse(
            action=OrchestratorAction.CHAT_RESPONSE,
            message="Hello! How can I help?",
        )
        mock_dispatch.return_value = response

        # Build pipeline with mocked internals
        pipeline = VoicePipeline(**pipeline_kwargs)

        # Mock the audio capture
        mock_capture = AsyncMock()
        audio_buffer = np.random.randn(16000).astype(np.float32)
        mock_capture.record_until_silence.return_value = audio_buffer

        # Mock recognizer (already set on pipeline)
        pipeline._recognizer = AsyncMock()
        pipeline._recognizer.transcribe.return_value = TranscriptionResult(
            text="hello corvus",
            language="en",
            duration_seconds=1.5,
        )

        # Mock synthesizer — _speak now uses synthesize_full()
        pipeline._synthesizer = AsyncMock()
        pipeline._synthesizer.synthesize_full.return_value = (
            np.zeros(1000, dtype=np.float32), 24000
        )

        # Mock player
        pipeline._player = AsyncMock()

        # drain() is a sync method — set it as a plain MagicMock on the async capture mock
        mock_capture.drain = MagicMock()

        await pipeline._process_utterance(mock_capture)

        # Verify the cycle completed
        mock_classify.assert_called_once()
        mock_dispatch.assert_called_once()
        assert len(pipeline.history.get_messages()) == 2  # user + assistant

    @patch("corvus.voice.pipeline.classify_intent")
    @patch("corvus.voice.pipeline.dispatch")
    async def test_process_utterance_empty_transcription(
        self, mock_dispatch, mock_classify, pipeline_kwargs
    ):
        """Test that empty transcription returns early without dispatching."""
        pipeline = VoicePipeline(**pipeline_kwargs)

        mock_capture = AsyncMock()
        mock_capture.drain = MagicMock()
        mock_capture.record_until_silence.return_value = np.random.randn(16000).astype(np.float32)

        pipeline._recognizer = AsyncMock()
        pipeline._recognizer.transcribe.return_value = TranscriptionResult(
            text="", language="en", duration_seconds=0.5
        )
        pipeline._player = AsyncMock()

        await pipeline._process_utterance(mock_capture)

        mock_classify.assert_not_called()
        mock_dispatch.assert_not_called()

    @patch("corvus.voice.pipeline.classify_intent")
    @patch("corvus.voice.pipeline.dispatch")
    async def test_process_utterance_stop_command(
        self, mock_dispatch, mock_classify, pipeline_kwargs
    ):
        """Test that stop/quit commands trigger shutdown."""
        pipeline = VoicePipeline(**pipeline_kwargs)
        pipeline._running = True

        mock_capture = AsyncMock()
        mock_capture.drain = MagicMock()
        mock_capture.record_until_silence.return_value = np.random.randn(16000).astype(np.float32)

        pipeline._recognizer = AsyncMock()
        pipeline._recognizer.transcribe.return_value = TranscriptionResult(
            text="goodbye", language="en", duration_seconds=0.5
        )

        pipeline._synthesizer = AsyncMock()
        pipeline._synthesizer.synthesize_full.return_value = (
            np.zeros(100, dtype=np.float32), 24000
        )
        pipeline._player = AsyncMock()

        await pipeline._process_utterance(mock_capture)

        assert not pipeline._running
        mock_classify.assert_not_called()

    @patch("corvus.voice.pipeline.classify_intent")
    @patch("corvus.voice.pipeline.dispatch")
    async def test_process_utterance_error_recovery(
        self, mock_dispatch, mock_classify, pipeline_kwargs
    ):
        """Test that errors during processing are caught and spoken."""
        mock_classify.side_effect = RuntimeError("LLM offline")

        pipeline = VoicePipeline(**pipeline_kwargs)

        mock_capture = AsyncMock()
        mock_capture.drain = MagicMock()
        mock_capture.record_until_silence.return_value = np.random.randn(16000).astype(np.float32)

        pipeline._recognizer = AsyncMock()
        pipeline._recognizer.transcribe.return_value = TranscriptionResult(
            text="hello", language="en", duration_seconds=0.5
        )

        pipeline._synthesizer = AsyncMock()
        pipeline._synthesizer.synthesize_full.return_value = (
            np.zeros(100, dtype=np.float32), 24000
        )
        pipeline._player = AsyncMock()

        # Should not raise — error is caught and spoken
        await pipeline._process_utterance(mock_capture)

    @patch("corvus.voice.pipeline.classify_intent")
    @patch("corvus.voice.pipeline.dispatch")
    async def test_process_utterance_deferred_conversation_creation(
        self, mock_dispatch, mock_classify, pipeline_kwargs, mock_store
    ):
        """Test that conversation is created on first real utterance."""
        classification = IntentClassification(
            intent=Intent.GENERAL_CHAT, confidence=0.9, reasoning="chat"
        )
        mock_classify.return_value = (classification, MagicMock())
        mock_dispatch.return_value = OrchestratorResponse(
            action=OrchestratorAction.CHAT_RESPONSE, message="Hi"
        )

        pipeline = VoicePipeline(**pipeline_kwargs)
        assert pipeline.history.conversation_id is None

        mock_capture = AsyncMock()
        mock_capture.drain = MagicMock()
        mock_capture.record_until_silence.return_value = np.random.randn(16000).astype(np.float32)

        pipeline._recognizer = AsyncMock()
        pipeline._recognizer.transcribe.return_value = TranscriptionResult(
            text="hello", language="en", duration_seconds=1.0
        )
        pipeline._synthesizer = AsyncMock()
        pipeline._synthesizer.synthesize_full.return_value = (
            np.zeros(100, dtype=np.float32), 24000
        )
        pipeline._player = AsyncMock()

        await pipeline._process_utterance(mock_capture)

        mock_store.create.assert_called_once_with("hello")

    async def test_process_utterance_empty_audio(self, pipeline_kwargs):
        """Test that empty audio buffer returns early."""
        pipeline = VoicePipeline(**pipeline_kwargs)

        mock_capture = AsyncMock()
        mock_capture.drain = MagicMock()
        mock_capture.record_until_silence.return_value = np.array([], dtype=np.float32)

        pipeline._player = AsyncMock()

        await pipeline._process_utterance(mock_capture)

        # Should return early — no transcription attempted
