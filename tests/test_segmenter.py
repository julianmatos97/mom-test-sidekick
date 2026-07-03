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
