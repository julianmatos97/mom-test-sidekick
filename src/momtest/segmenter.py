"""Energy-based speech segmentation. Push audio blocks, receive finished segments."""
from __future__ import annotations

import numpy as np


class Segmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.01,
        min_speech_s: float = 0.8,
        trailing_silence_s: float = 0.6,
        max_segment_s: float = 10.0,
    ):
        self.sr = sample_rate
        self.rms_threshold = rms_threshold
        self.min_speech = int(min_speech_s * sample_rate)
        self.trailing_silence = int(trailing_silence_s * sample_rate)
        self.max_segment = int(max_segment_s * sample_rate)
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._silence_run = 0
        self._has_speech = False
        self._speech_len = 0

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Feed one audio block; returns zero or more completed segments."""
        out: list[np.ndarray] = []
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if len(block) else 0.0
        loud = rms >= self.rms_threshold

        if loud:
            self._silence_run = 0
            self._has_speech = True
            self._speech_len += len(block)
        else:
            self._silence_run += len(block)
            if not self._has_speech:
                self._buf, self._buf_len = [], 0  # don't accumulate leading silence
                return out

        self._buf.append(block)
        self._buf_len += len(block)

        end_of_turn = self._has_speech and self._silence_run >= self.trailing_silence
        too_long = self._buf_len >= self.max_segment
        if end_of_turn or too_long:
            if self._speech_len >= self.min_speech:
                full = np.concatenate(self._buf)
                # Trim trailing silence on end_of_turn (we only needed it to detect turn end)
                if end_of_turn and self._silence_run > 0:
                    full = full[:-self._silence_run]
                out.append(full)
            self._buf, self._buf_len = [], 0
            self._silence_run = 0
            self._has_speech = False
            self._speech_len = 0
        return out
