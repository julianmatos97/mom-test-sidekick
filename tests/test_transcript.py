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
