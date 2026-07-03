# Mom Test Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Terminal HUD that listens to discovery calls locally (mic + system audio), transcribes with Parakeet, and coaches the user live with Mom Test questions, violation alerts, coverage tracking, and fact capture.

**Architecture:** Single Python process. Audio capture (sounddevice mic + Swift ScreenCaptureKit helper for system audio) feeds per-channel queues. An ASR worker segments audio on silence and transcribes chunks with parakeet-mlx into a transcript buffer. A coach loop sends the rolling transcript to Claude Haiku every ~15s and renders structured guidance in a Rich Live TUI. On quit, a markdown debrief is written.

**Tech Stack:** Python 3.11+ (uv), parakeet-mlx (Parakeet-TDT 0.6B v2), sounddevice, soundfile, numpy, rich, anthropic, pytest. Swift helper built with `swiftc` (ScreenCaptureKit, macOS 13+).

**Spec:** `docs/superpowers/specs/2026-07-03-mom-test-assistant-design.md`

---

## File Structure

```
pyproject.toml                  # uv project, deps, momtest entry point
helper/AudioTap.swift           # system-audio capture → Float32 PCM on stdout
helper/build.sh                 # swiftc build → helper/audiotap binary
src/momtest/__init__.py
src/momtest/transcript.py       # Utterance, Transcript buffer (rolling window)
src/momtest/segmenter.py        # silence-based audio segmentation (pure logic)
src/momtest/asr.py              # ASR worker thread: queues → parakeet → utterances
src/momtest/audio.py            # MicCapture (sounddevice), SystemCapture (helper subprocess)
src/momtest/coach.py            # prompt build, Claude call, JSON parse → CoachUpdate
src/momtest/tui.py              # CoachState → Rich renderable (pure)
src/momtest/debrief.py          # markdown debrief writer
src/momtest/session.py          # orchestrator: wires everything, hotkeys, coach tick
src/momtest/cli.py              # argparse: momtest start / momtest demo
tests/test_transcript.py
tests/test_segmenter.py
tests/test_coach.py
tests/test_debrief.py
tests/fixtures/                 # generated WAV fixture (Task 9)
calls/                          # debrief output (gitignored contents? no — keep, user data)
```

Conventions used throughout: audio is always **16 kHz mono float32 numpy arrays**. Speakers are the strings `"you"` and `"them"`.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/momtest/__init__.py`, `.gitignore`, `.python-version`

- [ ] **Step 1: Scaffold**

```bash
cd /Users/julianmatos/Github/mom-test
mkdir -p src/momtest tests helper calls
touch src/momtest/__init__.py
echo "3.12" > .python-version
```

Write `pyproject.toml`:

```toml
[project]
name = "momtest"
version = "0.1.0"
description = "Mom Test discovery-call HUD"
requires-python = ">=3.11"
dependencies = [
    "parakeet-mlx>=0.3",
    "sounddevice>=0.5",
    "soundfile>=0.12",
    "numpy>=1.26",
    "rich>=13.7",
    "anthropic>=0.40",
]

[project.scripts]
momtest = "momtest.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/momtest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs the real Parakeet model or live audio"]

[tool.uv]
# Without these, uv backtracks numba to 2021-era versions (llvmlite<0.37 build failure)
constraint-dependencies = ["numba>=0.60", "llvmlite>=0.43"]
```

Write `.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
helper/audiotap
calls/*.md
uv.lock
```

- [ ] **Step 2: Verify environment resolves**

Run: `uv sync && uv run python -c "import parakeet_mlx, sounddevice, rich, anthropic, soundfile; print('ok')"`
Expected: `ok` (first run downloads packages; parakeet-mlx needs Apple Silicon)

- [ ] **Step 3: Verify pytest runs**

Run: `uv run pytest`
Expected: `no tests ran` (exit code 5 is fine at this point)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore .python-version src
git commit -m "chore: scaffold momtest python project"
```

---

### Task 2: Swift system-audio tap (spike first — riskiest piece)

Captures all system audio output via ScreenCaptureKit and writes raw **Float32 PCM** to stdout. Requires Screen Recording permission (macOS prompts on first run).

**Files:**
- Create: `helper/AudioTap.swift`, `helper/build.sh`

- [ ] **Step 1: Write the helper**

`helper/AudioTap.swift`:

```swift
// Captures system audio via ScreenCaptureKit, writes 16kHz mono Float32 PCM to stdout.
// Build: helper/build.sh   Run: helper/audiotap > out.raw
import Foundation
import CoreMedia
import ScreenCaptureKit

final class AudioOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var length = 0
        var dataPointer: UnsafeMutablePointer<CChar>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
            totalLengthOut: &length, dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer else { return }
        FileHandle.standardOutput.write(Data(bytes: ptr, count: length))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("stream stopped: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let semaphore = DispatchSemaphore(value: 0)
Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            FileHandle.standardError.write("no display\n".data(using: .utf8)!)
            exit(1)
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        // SCStream requires a video config even for audio-only use; keep it tiny.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let output = AudioOutput()
        let stream = SCStream(filter: filter, configuration: config, delegate: output)
        try stream.addStreamOutput(output, type: .audio,
                                   sampleHandlerQueue: DispatchQueue(label: "audio"))
        try await stream.startCapture()
        FileHandle.standardError.write("capturing\n".data(using: .utf8)!)
    } catch {
        FileHandle.standardError.write("failed: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
semaphore.wait()
```

`helper/build.sh`:

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
swiftc -O -framework ScreenCaptureKit -framework CoreMedia AudioTap.swift -o audiotap
echo "built helper/audiotap"
```

Run: `chmod +x helper/build.sh && helper/build.sh`
Expected: `built helper/audiotap`

- [ ] **Step 2: Spike-verify the output format** (this is the whole point of doing this task first)

Play music/a YouTube video, then:

```bash
helper/audiotap > /tmp/tap.raw & TAP_PID=$!; sleep 5; kill $TAP_PID
uv run python - <<'EOF'
import numpy as np
a = np.frombuffer(open("/tmp/tap.raw","rb").read(), dtype=np.float32)
print("samples:", len(a), "≈", len(a)/16000, "s @16k mono; peak:", float(abs(a).max()))
EOF
```

Expected: sample count ≈ 16000 × seconds captured, peak > 0.01 (non-silence). First run triggers the Screen Recording permission prompt — grant it (to the terminal app) and re-run.

**If the numbers are off** (e.g. duration reads 2× or 3× real time): SCStream ignored the sample-rate/channel config and is emitting 48kHz and/or stereo/non-interleaved audio. Fix in Python instead of Swift: note the actual format here in the plan file, and in Task 6's `SystemCapture` deinterleave/downsample accordingly (`a[0::2]` for stereo interleaved; `a.reshape(2,-1)` chunks for non-interleaved per-buffer; `scipy`-free decimation via `a[::3]` from 48k→16k is acceptable for speech).

- [ ] **Step 3: Commit**

```bash
git add helper/AudioTap.swift helper/build.sh
git commit -m "feat: swift system-audio tap helper (ScreenCaptureKit)"
```

---

### Task 3: Transcript buffer

**Files:**
- Create: `src/momtest/transcript.py`
- Test: `tests/test_transcript.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_transcript.py`:

```python
from momtest.transcript import Transcript, Utterance


def test_add_and_order():
    t = Transcript()
    t.add("you", "hi there", ts=1.0)
    t.add("them", "hello", ts=2.0)
    assert [u.text for u in t.utterances] == ["hi there", "hello"]
    assert t.utterances[0].speaker == "you"


def test_window_returns_only_recent():
    t = Transcript()
    t.add("you", "old", ts=0.0)
    t.add("them", "recent", ts=100.0)
    recent = t.window(seconds=120, now=200.0)
    assert [u.text for u in recent] == ["recent"]


def test_render_formats_speakers():
    t = Transcript()
    t.add("you", "what do you use today?", ts=1.0)
    t.add("them", "spreadsheets", ts=2.0)
    text = t.render(t.utterances)
    assert text == "YOU: what do you use today?\nTHEM: spreadsheets"


def test_last_speaker():
    t = Transcript()
    assert t.last_speaker() is None
    t.add("them", "hello", ts=1.0)
    assert t.last_speaker() == "them"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcript.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'momtest.transcript'`

- [ ] **Step 3: Implement**

`src/momtest/transcript.py`:

```python
"""Append-only call transcript with a rolling recency window."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Utterance:
    speaker: str  # "you" | "them"
    text: str
    ts: float     # epoch seconds


@dataclass
class Transcript:
    utterances: list[Utterance] = field(default_factory=list)

    def add(self, speaker: str, text: str, ts: float | None = None) -> None:
        self.utterances.append(Utterance(speaker, text, ts if ts is not None else time.time()))

    def window(self, seconds: float = 120.0, now: float | None = None) -> list[Utterance]:
        cutoff = (now if now is not None else time.time()) - seconds
        return [u for u in self.utterances if u.ts >= cutoff]

    def last_speaker(self) -> str | None:
        return self.utterances[-1].speaker if self.utterances else None

    @staticmethod
    def render(utterances: list[Utterance]) -> str:
        return "\n".join(f"{u.speaker.upper()}: {u.text}" for u in utterances)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcript.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/momtest/transcript.py tests/test_transcript.py
git commit -m "feat: transcript buffer with rolling window"
```

---

### Task 4: Silence-based segmenter

Pure logic: feed audio blocks in, get complete speech segments out. Flushes when ≥0.6s of trailing silence follows ≥0.8s of speech, or when a segment hits 10s.

**Files:**
- Create: `src/momtest/segmenter.py`
- Test: `tests/test_segmenter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_segmenter.py`:

```python
import numpy as np
from momtest.segmenter import Segmenter

SR = 16000


def speech(seconds):  # loud noise stands in for speech
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SR * seconds)) * 0.3).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def feed(seg, audio, block=1600):
    out = []
    for i in range(0, len(audio), block):
        out.extend(seg.push(audio[i:i + block]))
    return out


def test_speech_then_silence_flushes_one_segment():
    seg = Segmenter(sample_rate=SR)
    segments = feed(seg, np.concatenate([speech(2.0), silence(1.0)]))
    assert len(segments) == 1
    assert 1.5 * SR < len(segments[0]) < 2.5 * SR


def test_pure_silence_yields_nothing():
    seg = Segmenter(sample_rate=SR)
    assert feed(seg, silence(3.0)) == []


def test_short_blip_is_discarded():
    seg = Segmenter(sample_rate=SR)  # 0.3s speech < min_speech 0.8s
    assert feed(seg, np.concatenate([speech(0.3), silence(1.0)])) == []


def test_long_speech_force_flushes_at_max():
    seg = Segmenter(sample_rate=SR, max_segment_s=10.0)
    segments = feed(seg, speech(12.0))
    assert len(segments) >= 1
    assert len(segments[0]) <= 10.5 * SR
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'momtest.segmenter'`

- [ ] **Step 3: Implement**

`src/momtest/segmenter.py`:

```python
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
                out.append(np.concatenate(self._buf))
            self._buf, self._buf_len = [], 0
            self._silence_run = 0
            self._has_speech = False
            self._speech_len = 0
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/momtest/segmenter.py tests/test_segmenter.py
git commit -m "feat: energy-based speech segmenter"
```

---

### Task 5: Coach engine (prompt + parse + API call)

**Files:**
- Create: `src/momtest/coach.py`
- Test: `tests/test_coach.py`

- [ ] **Step 1: Write the failing tests** (prompt build + parse only; the API call itself is a thin wrapper, exercised in demo mode)

`tests/test_coach.py`:

```python
from momtest.coach import CoachUpdate, build_user_prompt, parse_response


def test_parse_valid_json():
    raw = """{"questions": ["Ask: when did that last happen?"],
               "alerts": [{"kind": "pitching", "detail": "you described your product"}],
               "coverage": {"problem": "covered", "budget": "missing"},
               "facts": ["uses spreadsheets for tracking"]}"""
    u = parse_response(raw)
    assert u.questions == ["Ask: when did that last happen?"]
    assert u.alerts[0]["kind"] == "pitching"
    assert u.coverage["budget"] == "missing"
    assert u.facts == ["uses spreadsheets for tracking"]


def test_parse_strips_code_fences_and_clamps_questions():
    raw = '```json\n{"questions": ["a","b","c","d"], "alerts": [], "coverage": {}, "facts": []}\n```'
    u = parse_response(raw)
    assert u.questions == ["a", "b"]


def test_parse_garbage_returns_none():
    assert parse_response("sorry, I can't do that") is None


def test_build_user_prompt_includes_context():
    p = build_user_prompt(
        hypothesis="landlords struggle to track maintenance requests",
        recent="THEM: we mostly use email",
        summary="Intro done.",
        coverage={"problem": "partial"},
    )
    assert "landlords struggle" in p
    assert "THEM: we mostly use email" in p
    assert "Intro done." in p
    assert '"problem": "partial"' in p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coach.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'momtest.coach'`

- [ ] **Step 3: Implement**

`src/momtest/coach.py`:

```python
"""Mom Test coach: builds prompts, calls Claude, parses structured guidance."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a live coach for customer discovery calls, enforcing The Mom Test (Rob Fitzpatrick).
You see a rolling transcript. YOU = the founder (your user). THEM = the prospect.

Rules you enforce:
- Ask about past behavior and specifics, never hypotheticals ("would you...") .
- Compliments and generic praise are worthless data — flag them, redirect to facts.
- If YOU is pitching or explaining their product instead of listening, flag it.
- Chase concrete facts: money, time, tools, named events ("when did that last happen?",
  "walk me through the last time", "what else did you try?", "who else should I talk to?").

Respond with ONLY a JSON object, no prose:
{
  "questions": [up to 2 short imperative suggestions, e.g. "Ask: when did that last happen?"],
  "alerts": [{"kind": "pitching"|"hypothetical"|"compliment"|"fluff", "detail": "<1 short sentence>"}],
  "coverage": {"<goal>": "missing"|"partial"|"covered", ...for each goal given},
  "facts": [new concrete facts from the recent transcript only]
}
Alerts only for things happening in the RECENT transcript. Empty lists are fine.
"""


@dataclass
class CoachUpdate:
    questions: list[str] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)


def build_user_prompt(hypothesis: str, recent: str, summary: str, coverage: dict[str, str]) -> str:
    return (
        f"Hypothesis being tested: {hypothesis}\n\n"
        f"Discovery goals and current coverage:\n{json.dumps(coverage)}\n\n"
        f"Summary of earlier conversation:\n{summary or '(call just started)'}\n\n"
        f"Recent transcript (last ~2 min):\n{recent or '(no speech yet)'}"
    )


def parse_response(raw: str) -> CoachUpdate | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return CoachUpdate(
        questions=[str(q) for q in data.get("questions", [])][:2],
        alerts=[a for a in data.get("alerts", []) if isinstance(a, dict)],
        coverage={str(k): str(v) for k, v in data.get("coverage", {}).items()},
        facts=[str(f) for f in data.get("facts", [])],
    )


class CoachEngine:
    def __init__(self, hypothesis: str, goals: list[str]):
        self.client = anthropic.Anthropic()
        self.hypothesis = hypothesis
        self.coverage: dict[str, str] = {g: "missing" for g in goals}
        self.summary = ""

    def tick(self, recent: str) -> CoachUpdate | None:
        """One coaching pass. Returns None on parse failure (caller keeps last state)."""
        msg = self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(
                self.hypothesis, recent, self.summary, self.coverage)}],
        )
        update = parse_response(msg.content[0].text)
        if update and update.coverage:
            self.coverage.update(update.coverage)
        return update
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coach.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/momtest/coach.py tests/test_coach.py
git commit -m "feat: mom test coach engine (prompt, parse, claude call)"
```

---

### Task 6: Audio capture (mic + system tap subprocess)

Thin I/O wrappers — no unit tests; verified live in Task 10 and via `momtest demo`. Both push float32 blocks into a `queue.Queue`.

**Files:**
- Create: `src/momtest/audio.py`

- [ ] **Step 1: Implement**

`src/momtest/audio.py`:

```python
"""Audio sources. Both push 16kHz mono float32 numpy blocks into a queue."""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK = 1600  # 100 ms

HELPER = Path(__file__).resolve().parents[2] / "helper" / "audiotap"


class MicCapture:
    """Your voice. sounddevice input stream → queue."""

    def __init__(self, out: queue.Queue):
        self.out = out
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            self.out.put(indata[:, 0].copy())

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
            [str(self.helper)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def reader():
            assert self.proc and self.proc.stdout
            while True:
                raw = self.proc.stdout.read(BLOCK * 4)  # float32 = 4 bytes
                if not raw:
                    break
                self.out.put(self._convert(raw))

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
                self.out.put(audio[i:i + BLOCK])
                if self.realtime:
                    time.sleep(BLOCK / SAMPLE_RATE)

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 2: Sanity check imports**

Run: `uv run python -c "from momtest.audio import MicCapture, SystemCapture, FileCapture; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/momtest/audio.py
git commit -m "feat: mic, system-tap, and file audio capture sources"
```

---

### Task 7: ASR worker

One thread owns the Parakeet model. Pulls blocks from both channel queues, segments per channel, transcribes finished segments **sequentially** (no concurrent model use), emits utterances via callback. Chunked transcription (write segment to temp WAV, `model.transcribe(path)`) — simple and robust; streaming API is a future optimization.

**Files:**
- Create: `src/momtest/asr.py`
- Test: `tests/test_asr.py` (integration, marked slow)

- [ ] **Step 1: Implement**

`src/momtest/asr.py`:

```python
"""ASR worker: channel queues → segmenter → Parakeet → utterance callback."""
from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from momtest.segmenter import Segmenter

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
                    text = self._transcribe(model, segment)
                    if text:
                        self.on_utterance(speaker, text, time.time())
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
```

- [ ] **Step 2: Write the integration test**

`tests/test_asr.py`:

```python
import queue
import time

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.slow  # downloads/loads the real model


def test_transcribes_spoken_wav(tmp_path):
    from momtest.asr import ASRWorker

    # Generate a spoken fixture with macOS `say` (16kHz mono wav)
    import subprocess
    aiff = tmp_path / "s.aiff"
    wav = tmp_path / "s.wav"
    subprocess.run(["say", "-o", str(aiff), "we use spreadsheets to track maintenance"], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(wav)], check=True)

    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000

    q: queue.Queue = queue.Queue()
    results = []
    worker = ASRWorker({"them": q}, lambda s, t, ts: results.append((s, t)))
    worker.start()
    assert worker.ready.wait(timeout=300), "model failed to load"

    for i in range(0, len(audio), 1600):
        q.put(audio[i:i + 1600])
    q.put(np.zeros(16000, dtype=np.float32))  # trailing silence to flush

    deadline = time.time() + 60
    while not results and time.time() < deadline:
        time.sleep(0.5)
    worker.stop()

    assert results, "no transcription produced"
    speaker, text = results[0]
    assert speaker == "them"
    assert "spreadsheet" in text.lower()
```

- [ ] **Step 3: Run the integration test** (first run downloads the ~1.2GB model)

Run: `uv run pytest tests/test_asr.py -v -m slow`
Expected: 1 PASS. If `model.transcribe(path)` has a different signature in the installed parakeet-mlx version, check `uv run python -c "from parakeet_mlx import from_pretrained; help(from_pretrained('mlx-community/parakeet-tdt-0.6b-v2').transcribe)"` and adapt `_transcribe` — the contract stays (segment array in, text out).

- [ ] **Step 4: Verify fast tests still pass without the model**

Run: `uv run pytest -m "not slow"`
Expected: all prior tests PASS, asr test deselected

- [ ] **Step 5: Commit**

```bash
git add src/momtest/asr.py tests/test_asr.py
git commit -m "feat: parakeet asr worker with per-channel segmentation"
```

---

### Task 8: TUI + debrief writer

**Files:**
- Create: `src/momtest/tui.py`, `src/momtest/debrief.py`
- Test: `tests/test_debrief.py`

- [ ] **Step 1: Write the failing debrief test**

`tests/test_debrief.py`:

```python
from momtest.coach import CoachUpdate
from momtest.debrief import write_debrief
from momtest.transcript import Transcript


def test_write_debrief(tmp_path):
    t = Transcript()
    t.add("you", "what do you use today?", ts=1.0)
    t.add("them", "spreadsheets", ts=2.0)
    state = CoachUpdate(
        questions=[],
        alerts=[{"kind": "pitching", "detail": "you pitched"}],
        coverage={"problem": "covered", "budget": "missing"},
        facts=["uses spreadsheets"],
    )
    path = write_debrief(tmp_path, prospect="Acme Jane", hypothesis="tracking is painful",
                         transcript=t, state=state,
                         alert_history=[{"kind": "pitching", "detail": "you pitched"},
                                        {"kind": "hypothetical", "detail": "would you use"}])
    content = path.read_text()
    assert "Acme Jane" in content
    assert "tracking is painful" in content
    assert "uses spreadsheets" in content
    assert "budget" in content            # coverage gap listed
    assert "pitching: 1" in content       # scorecard counts
    assert "YOU: what do you use today?" in content
    assert path.name.endswith("acme-jane.md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_debrief.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'momtest.debrief'`

- [ ] **Step 3: Implement debrief**

`src/momtest/debrief.py`:

```python
"""Post-call markdown debrief."""
from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path

from momtest.coach import CoachUpdate
from momtest.transcript import Transcript


def write_debrief(
    out_dir: Path,
    prospect: str,
    hypothesis: str,
    transcript: Transcript,
    state: CoachUpdate,
    alert_history: list[dict],
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", prospect.lower()).strip("-") or "call"
    path = Path(out_dir) / f"{date.today().isoformat()}-{slug}.md"
    counts = Counter(a.get("kind", "?") for a in alert_history)
    gaps = [g for g, s in state.coverage.items() if s != "covered"]

    lines = [
        f"# Discovery call — {prospect}",
        f"\n**Date:** {date.today().isoformat()}",
        f"**Hypothesis:** {hypothesis}",
        "\n## Facts captured",
        *([f"- {f}" for f in state.facts] or ["- (none)"]),
        "\n## Coverage",
        *[f"- {g}: {s}" for g, s in state.coverage.items()],
        "\n## Gaps to chase next time",
        *([f"- {g}" for g in gaps] or ["- none — full coverage"]),
        "\n## Mom Test scorecard (violations flagged)",
        *([f"- {kind}: {n}" for kind, n in counts.items()] or ["- clean call"]),
        "\n## Transcript",
        "```",
        Transcript.render(transcript.utterances),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_debrief.py -v`
Expected: PASS

- [ ] **Step 5: Implement TUI** (pure state → renderable; verified visually in demo)

`src/momtest/tui.py`:

```python
"""CoachState → Rich layout. Pure rendering, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from momtest.coach import CoachUpdate


@dataclass
class HudState:
    update: CoachUpdate = field(default_factory=CoachUpdate)
    facts: list[str] = field(default_factory=list)      # accumulated across ticks
    status: str = "listening"                            # listening | paused | stale | error
    last_heard: str = ""                                 # most recent utterance snippet


STATUS_STYLE = {"listening": "green", "paused": "yellow", "stale": "yellow", "error": "red"}
COVERAGE_MARK = {"covered": ("✓", "green"), "partial": ("~", "yellow"), "missing": ("✗", "red")}


def build_hud(state: HudState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="questions", ratio=3),
        Layout(name="alerts", ratio=2),
        Layout(name="bottom", ratio=3),
    )
    layout["bottom"].split_row(Layout(name="coverage"), Layout(name="facts"))

    questions = state.update.questions or ["(listening…)"]
    layout["questions"].update(Panel(
        Group(*[Text(q, style="bold cyan") for q in questions]),
        title=f"ASK NEXT — {state.status}",
        border_style=STATUS_STYLE.get(state.status, "white")))

    if state.update.alerts:
        alerts = Group(*[Text(f"⚠ {a.get('kind', '?').upper()}: {a.get('detail', '')}",
                              style="bold red") for a in state.update.alerts])
        border = "red"
    else:
        alerts, border = Text("no violations", style="dim"), "dim"
    layout["alerts"].update(Panel(alerts, title="MOM TEST ALERTS", border_style=border))

    cov_lines = []
    for goal, s in state.update.coverage.items():
        mark, color = COVERAGE_MARK.get(s, ("?", "white"))
        cov_lines.append(Text(f"{mark} {goal}", style=color))
    layout["coverage"].update(Panel(Group(*cov_lines) if cov_lines else Text("—"),
                                    title="COVERAGE"))

    fact_lines = [Text(f"• {f}") for f in state.facts[-8:]]
    layout["facts"].update(Panel(Group(*fact_lines) if fact_lines else Text("—"),
                                 title="FACTS"))
    return layout
```

- [ ] **Step 6: Smoke the TUI once**

Run:

```bash
uv run python - <<'EOF'
from momtest.coach import CoachUpdate
from momtest.tui import HudState, build_hud
from rich.console import Console
s = HudState(update=CoachUpdate(
    questions=["Ask: when did that last happen?"],
    alerts=[{"kind": "compliment", "detail": "prospect said 'cool idea'"}],
    coverage={"problem": "covered", "budget": "missing"}),
    facts=["uses spreadsheets"])
Console().print(build_hud(s))
EOF
```

Expected: 4-panel layout renders, no traceback.

- [ ] **Step 7: Commit**

```bash
git add src/momtest/tui.py src/momtest/debrief.py tests/test_debrief.py
git commit -m "feat: rich hud renderer and markdown debrief writer"
```

---

### Task 9: Session orchestrator + CLI

**Files:**
- Create: `src/momtest/session.py`, `src/momtest/cli.py`
- Create: `tests/fixtures/demo.wav` (generated)

- [ ] **Step 1: Implement session**

`src/momtest/session.py`:

```python
"""Wires audio → asr → transcript → coach → hud. Blocking run loop."""
from __future__ import annotations

import queue
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from rich.console import Console
from rich.live import Live

from momtest import audio
from momtest.asr import ASRWorker
from momtest.coach import CoachEngine, CoachUpdate
from momtest.debrief import write_debrief
from momtest.transcript import Transcript
from momtest.tui import HudState, build_hud

COACH_INTERVAL_S = 15.0
DEFAULT_GOALS = ["problem", "current solution", "cost of problem", "budget & authority"]


class Session:
    def __init__(self, prospect: str, hypothesis: str, sources: list, channels: dict):
        self.prospect = prospect
        self.hypothesis = hypothesis
        self.sources = sources          # objects with .start()/.stop()
        self.channels = channels        # {"you": Queue, "them": Queue}
        self.transcript = Transcript()
        self.coach = CoachEngine(hypothesis, DEFAULT_GOALS)
        self.state = HudState(update=CoachUpdate(coverage=self.coach.coverage.copy()))
        self.alert_history: list[dict] = []
        self.paused = False
        self.quit = threading.Event()

    # -- construction helpers -------------------------------------------------

    @classmethod
    def live(cls, prospect: str, hypothesis: str) -> "Session":
        you_q, them_q = queue.Queue(), queue.Queue()
        return cls(prospect, hypothesis,
                   sources=[audio.MicCapture(you_q), audio.SystemCapture(them_q)],
                   channels={"you": you_q, "them": them_q})

    @classmethod
    def demo(cls, fixture: Path) -> "Session":
        them_q: queue.Queue = queue.Queue()
        return cls("Demo", "demo hypothesis",
                   sources=[audio.FileCapture(them_q, fixture)],
                   channels={"them": them_q})

    # -- run loop --------------------------------------------------------------

    def on_utterance(self, speaker: str, text: str, ts: float) -> None:
        self.transcript.add(speaker, text, ts)
        self.state.last_heard = f"{speaker}: {text}"

    def coach_loop(self) -> None:
        while not self.quit.wait(COACH_INTERVAL_S):
            if self.paused or not self.transcript.utterances:
                continue
            window = self.transcript.window(120)
            recent = Transcript.render(window)
            older = self.transcript.utterances[: len(self.transcript.utterances) - len(window)]
            self.coach.summary = Transcript.render(older)[-1200:]  # cheap "summary": older tail
            for attempt in range(3):
                try:
                    update = self.coach.tick(recent)
                    break
                except Exception:
                    time.sleep(2 ** attempt)
            else:
                self.state.status = "error"
                continue
            if update is None:
                self.state.status = "stale"   # parse failed; keep last guidance
                continue
            self.state.status = "listening"
            self.state.update = update
            for f in update.facts:
                if f not in self.state.facts:
                    self.state.facts.append(f)
            self.alert_history.extend(update.alerts)

    def keys_loop(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit.is_set():
                ch = sys.stdin.read(1)
                if ch == "q":
                    self.quit.set()
                elif ch == "p":
                    self.paused = not self.paused
                    self.state.status = "paused" if self.paused else "listening"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self) -> Path | None:
        console = Console()
        console.print("[bold]loading parakeet…[/bold] (first run downloads ~1.2GB)")
        worker = ASRWorker(self.channels, self.on_utterance)
        worker.start()
        worker.ready.wait()
        for s in self.sources:
            s.start()
        threading.Thread(target=self.coach_loop, daemon=True).start()
        threading.Thread(target=self.keys_loop, daemon=True).start()

        last_reconnect = 0.0
        with Live(build_hud(self.state), console=console, refresh_per_second=4) as live:
            while not self.quit.is_set():
                # auto-reconnect a dead system tap (spec: banner + reconnect loop)
                for s in self.sources:
                    if isinstance(s, audio.SystemCapture) and not s.alive():
                        self.state.status = "error"
                        if time.time() - last_reconnect > 5.0:
                            last_reconnect = time.time()
                            try:
                                s.start()
                                self.state.status = "listening"
                            except Exception:
                                pass
                live.update(build_hud(self.state))
                time.sleep(0.25)

        for s in self.sources:
            s.stop()
        worker.stop()
        if not self.transcript.utterances:
            console.print("no speech captured — no debrief written")
            return None
        final = CoachUpdate(coverage=self.coach.coverage,
                            facts=self.state.facts, alerts=[], questions=[])
        path = write_debrief(Path("calls"), self.prospect, self.hypothesis,
                             self.transcript, final, self.alert_history)
        console.print(f"debrief: [bold]{path}[/bold]")
        return path
```

- [ ] **Step 2: Implement CLI**

`src/momtest/cli.py`:

```python
"""momtest CLI: start (live call) and demo (wav replay)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from momtest.session import Session

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.wav"


def main() -> None:
    parser = argparse.ArgumentParser(prog="momtest", description="Mom Test call HUD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="live call")
    start.add_argument("--prospect", default=None)
    start.add_argument("--hypothesis", default=None)

    demo = sub.add_parser("demo", help="replay a wav fixture through the pipeline")
    demo.add_argument("--fixture", type=Path, default=FIXTURE)

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — the coach needs it.")

    if args.cmd == "start":
        prospect = args.prospect or input("Prospect name: ").strip() or "unknown"
        hypothesis = args.hypothesis or input("Hypothesis you're testing: ").strip()
        session = Session.live(prospect, hypothesis)
    else:
        if not args.fixture.exists():
            sys.exit(f"fixture not found: {args.fixture} — run scripts in Task 9 Step 3")
        session = Session.demo(args.fixture)

    session.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the demo fixture** (spoken prospect audio via macOS `say`)

```bash
mkdir -p tests/fixtures
say -o /tmp/demo.aiff "[[slnc 500]] Well, honestly we track all our maintenance requests in a shared spreadsheet. [[slnc 1500]] Last month we lost two requests completely and a tenant escalated to the owner. [[slnc 1500]] I would definitely pay for something better, your product sounds really cool. [[slnc 1000]]"
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/demo.aiff tests/fixtures/demo.wav
uv run python -c "import soundfile as sf; a, sr = sf.read('tests/fixtures/demo.wav'); print(sr, len(a)/sr, 's')"
```

Expected: `16000 <≈20> s`

- [ ] **Step 4: Run the demo end-to-end** (needs `ANTHROPIC_API_KEY`)

Run: `uv run momtest demo`
Expected: HUD renders; within ~30s utterances appear (spreadsheet/maintenance content), coach panel shows suggestions; the "your product sounds really cool" line should trigger a `compliment` alert on a subsequent tick. Press `q` → debrief written to `calls/`.

- [ ] **Step 5: Run full fast test suite**

Run: `uv run pytest -m "not slow"`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/momtest/session.py src/momtest/cli.py tests/fixtures/demo.wav
git commit -m "feat: session orchestrator, cli, demo mode"
```

---

### Task 10: Live smoke test + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Live smoke test**

Play a video with speech (stands in for the prospect), run:

```bash
uv run momtest start --prospect "smoke" --hypothesis "testing the pipeline"
```

Talk into the mic too. Expected: both `you` and `them` utterances flow (visible via facts/coach reacting), no crash over 2+ minutes, `q` writes debrief. Known first-run friction: Screen Recording permission (system tap) and Microphone permission prompts — grant to the terminal app and rerun.

- [ ] **Step 2: Write README**

`README.md`:

```markdown
# momtest — Mom Test discovery-call HUD

Terminal heads-up display for customer discovery calls. Listens locally
(mic = you, system audio = prospect), transcribes with Parakeet on-device,
and coaches you live: next Mom Test questions, violation alerts
(pitching / hypotheticals / compliments), coverage tracker, fact capture.
Writes a markdown debrief per call to `calls/`.

## Requirements

- Apple Silicon Mac (Parakeet runs via MLX), macOS 13+
- Xcode command line tools (`xcode-select --install`)
- `uv`
- `ANTHROPIC_API_KEY` in the environment

## Setup

    uv sync
    helper/build.sh          # builds the system-audio tap

First run prompts for Microphone and Screen Recording permissions
(grant to your terminal app).

## Use

    uv run momtest start     # asks for prospect + hypothesis, then HUD
    uv run momtest demo      # replay a canned call through the pipeline

Hotkeys: `p` pause coaching, `q` end call + write debrief.

## Tests

    uv run pytest -m "not slow"   # fast unit tests
    uv run pytest -m slow         # real-model ASR integration test (~1.2GB download)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: readme with setup and usage"
```
