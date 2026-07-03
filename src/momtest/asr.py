"""ASR worker: channel queues → segmenter → Parakeet → utterance callback."""
from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from momtest.segmenter import Segmenter

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


class ASRWorker(threading.Thread):
    def __init__(
        self,
        channels: dict[str, queue.Queue],  # {"you": q, "them": q}
        on_utterance: Callable[[str, str, float], None],  # (speaker, text, ts)
        model_id: str = MODEL_ID,
    ):
        super().__init__(daemon=True)
        self.channels = channels
        self.on_utterance = on_utterance
        self.model_id = model_id
        self.segmenters = {name: Segmenter(SAMPLE_RATE) for name in channels}
        self.ready = threading.Event()  # set once the model is loaded
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from parakeet_mlx import from_pretrained  # slow import, keep in thread

        model = from_pretrained(self.model_id)
        self.ready.set()

        while not self._stop.is_set():
            idle = True
            for speaker, q in self.channels.items():
                try:
                    block = q.get_nowait()
                except queue.Empty:
                    continue
                idle = False
                for segment in self.segmenters[speaker].push(block):
                    try:
                        text = self._transcribe(model, segment)
                        if text:
                            self.on_utterance(speaker, text, time.time())
                    except Exception:
                        logger.warning(
                            "transcription failed for %s segment; skipping",
                            speaker,
                            exc_info=True,
                        )
            if idle:
                time.sleep(0.05)

    @staticmethod
    def _transcribe(model, segment: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            sf.write(path, segment, SAMPLE_RATE)
            result = model.transcribe(str(path))
            return result.text.strip()
        finally:
            path.unlink(missing_ok=True)
