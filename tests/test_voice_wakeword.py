"""Tests for corvus.voice.wakeword — WakeWordDetector with mocked openWakeWord."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corvus.schemas.voice import WakeWordEvent
from corvus.voice.wakeword import WakeWordDetector


@pytest.fixture(autouse=True)
def _mock_openwakeword():
    """Inject a fake openwakeword module so the import inside __aenter__ works."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"openwakeword": mock_module}):
        yield mock_module


class TestWakeWordDetector:
    async def test_disabled_when_no_model_path(self):
        """Test that detector is disabled when model path is empty."""
        detector = WakeWordDetector(model_path="", threshold=0.5)
        async with detector:
            assert not detector.enabled
            result = detector.process_frame(np.zeros(1280, dtype=np.float32))
            assert result is None

    async def test_disabled_when_model_file_missing(self, tmp_path):
        """Test that detector is disabled when model file doesn't exist."""
        detector = WakeWordDetector(
            model_path=str(tmp_path / "nonexistent.onnx"), threshold=0.5
        )
        async with detector:
            assert not detector.enabled

    async def test_load_by_name(self, _mock_openwakeword):
        """Test that a model name (no path separator/extension) loads via openwakeword."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {}
        mock_model.models = {"hey_jarvis": MagicMock()}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path="hey_jarvis", threshold=0.5)
        async with detector:
            assert detector.enabled
            _mock_openwakeword.Model.assert_called_once_with(wakeword_models=["hey_jarvis"])

    async def test_detection_above_threshold(self, tmp_path, _mock_openwakeword):
        """Test that process_frame returns event when confidence exceeds threshold."""
        model_file = tmp_path / "corvus.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = {"corvus": 0.85}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path=str(model_file), threshold=0.5)
        async with detector:
            assert detector.enabled
            frame = np.zeros(1280, dtype=np.float32)
            event = detector.process_frame(frame)

            assert event is not None
            assert isinstance(event, WakeWordEvent)
            assert event.keyword == "corvus"
            assert event.confidence == 0.85

    async def test_no_detection_below_threshold(self, tmp_path, _mock_openwakeword):
        """Test that process_frame returns None when confidence is below threshold."""
        model_file = tmp_path / "corvus.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = {"corvus": 0.3}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path=str(model_file), threshold=0.5)
        async with detector:
            frame = np.zeros(1280, dtype=np.float32)
            result = detector.process_frame(frame)
            assert result is None

    async def test_reset_clears_buffers(self, tmp_path, _mock_openwakeword):
        """Test that reset() calls model.reset()."""
        model_file = tmp_path / "corvus.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = {}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path=str(model_file))
        async with detector:
            detector.reset()
            mock_model.reset.assert_called_once()

    async def test_float32_to_int16_conversion(self, tmp_path, _mock_openwakeword):
        """Test that float32 input is converted to int16 for openWakeWord."""
        model_file = tmp_path / "corvus.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = {}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path=str(model_file))
        async with detector:
            frame = np.full(1280, 0.5, dtype=np.float32)
            detector.process_frame(frame)

            call_args = mock_model.predict.call_args[0][0]
            assert call_args.dtype == np.int16

    async def test_cleanup_on_exit(self, tmp_path, _mock_openwakeword):
        """Test that __aexit__ clears model and disables detector."""
        model_file = tmp_path / "corvus.onnx"
        model_file.touch()

        mock_model = MagicMock()
        mock_model.predict.return_value = {}
        _mock_openwakeword.Model.return_value = mock_model

        detector = WakeWordDetector(model_path=str(model_file))
        async with detector:
            assert detector.enabled
        assert not detector.enabled
        assert detector._model is None
