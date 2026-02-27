"""Wake word detection via openWakeWord.

WakeWordDetector is an async context manager that loads an openWakeWord
model and provides frame-level detection via ``process_frame()``.
If the model path is empty or missing, the detector is disabled and
always returns None (useful for dev/testing without a trained model).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from corvus.schemas.voice import WakeWordEvent

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Async context manager wrapping openWakeWord for wake word detection.

    Usage::

        async with WakeWordDetector("models/corvus.onnx") as detector:
            event = detector.process_frame(audio_frame)
            if event:
                print(f"Wake word detected: {event.keyword}")
    """

    def __init__(
        self,
        model_path: str = "",
        threshold: float = 0.5,
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._model = None
        self._enabled = False

    async def __aenter__(self) -> WakeWordDetector:
        if not self._model_path or not Path(self._model_path).exists():
            logger.warning(
                "Wake word model not found at '%s' — detector disabled",
                self._model_path,
            )
            self._enabled = False
            return self

        from openwakeword import Model

        logger.info("Loading wake word model: %s", self._model_path)
        self._model = Model(wakeword_models=[self._model_path])
        self._enabled = True
        logger.info("Wake word detector ready (threshold=%.2f)", self._threshold)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._model = None
        self._enabled = False

    def process_frame(self, audio_frame: np.ndarray) -> WakeWordEvent | None:
        """Process a single audio frame and check for wake word detection.

        Args:
            audio_frame: Audio data as int16 or float32 numpy array.

        Returns:
            WakeWordEvent if the wake word was detected above threshold,
            otherwise None.
        """
        if not self._enabled or self._model is None:
            return None

        # openWakeWord expects int16 samples
        if audio_frame.dtype == np.float32:
            audio_int16 = (audio_frame * 32767).astype(np.int16)
        else:
            audio_int16 = audio_frame

        prediction = self._model.predict(audio_int16.flatten())

        for keyword, confidence in prediction.items():
            if confidence >= self._threshold:
                logger.info("Wake word '%s' detected (confidence=%.3f)", keyword, confidence)
                return WakeWordEvent(keyword=keyword, confidence=confidence)

        return None

    def reset(self) -> None:
        """Clear internal detection buffers after wake word is handled."""
        if self._model is not None:
            self._model.reset()

    @property
    def enabled(self) -> bool:
        return self._enabled
