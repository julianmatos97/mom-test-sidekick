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
