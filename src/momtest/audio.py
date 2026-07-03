"""Audio sources. Both push 16kHz mono float32 numpy blocks into a queue."""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK = 1600  # 100 ms

HELPER = Path(__file__).resolve().parents[2] / "helper" / "audiotap"


def _put_drop_oldest(q: queue.Queue, block) -> None:
    """Non-blocking put; on a full queue drop the oldest block to make room."""
    try:
        q.put_nowait(block)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(block)
        except queue.Full:
            pass  # still full; drop this block


class MicCapture:
    """Your voice. sounddevice input stream → queue."""

    def __init__(self, out: queue.Queue):
        self.out = out
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            _put_drop_oldest(self.out, indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, callback=callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()


class SystemCapture:
    """Prospect's voice: helper/audiotap subprocess stdout → queue.

    If Task 2's spike found a different raw format (48kHz / stereo),
    adjust _convert() accordingly.
    """

    def __init__(self, out: queue.Queue, helper: Path = HELPER):
        self.out = out
        self.helper = helper
        self.proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _convert(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.float32)

    def start(self) -> None:
        if not self.helper.exists():
            raise FileNotFoundError(
                f"{self.helper} missing — run helper/build.sh (needs Xcode CLT). "
                "First run also needs the Screen Recording permission.")
        self.proc = subprocess.Popen(
            [str(self.helper)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def drain_stderr():
            assert self.proc and self.proc.stderr
            for line in self.proc.stderr:
                logging.getLogger(__name__).warning(
                    "audiotap: %s", line.decode(errors="replace").rstrip())

        threading.Thread(target=drain_stderr, daemon=True).start()

        def reader():
            assert self.proc and self.proc.stdout
            while True:
                raw = self.proc.stdout.read(BLOCK * 4)  # float32 = 4 bytes
                if not raw:
                    break
                raw = raw[: len(raw) & ~3]  # partial read at EOF: trim to float32 boundary
                if not raw:
                    continue
                _put_drop_oldest(self.out, self._convert(raw))

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()


class FileCapture:
    """Demo mode: replay a 16kHz mono WAV as if it were live audio."""

    def __init__(self, out: queue.Queue, path: Path, realtime: bool = True):
        self.out = out
        self.path = path
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import time

        import soundfile as sf

        def reader():
            audio, sr = sf.read(self.path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            assert sr == SAMPLE_RATE, f"fixture must be 16kHz, got {sr}"
            for i in range(0, len(audio), BLOCK):
                if self._stop.is_set():
                    return
                _put_drop_oldest(self.out, audio[i:i + BLOCK])
                if self.realtime:
                    time.sleep(BLOCK / SAMPLE_RATE)

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
