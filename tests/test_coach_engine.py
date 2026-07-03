from unittest.mock import MagicMock

from momtest.coach import CoachEngine


def _fake_response(text):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def test_tick_merges_partial_coverage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    eng = CoachEngine("hypo", ["problem", "budget"])
    eng.client = MagicMock()
    eng.client.messages.create.return_value = _fake_response(
        '{"questions": [], "alerts": [], "coverage": {"problem": "covered"}, "facts": []}')
    update = eng.tick("THEM: stuff")
    assert eng.coverage == {"problem": "covered", "budget": "missing"}
    assert update.coverage == {"problem": "covered"}


def test_tick_parse_failure_returns_none_and_keeps_coverage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    eng = CoachEngine("hypo", ["problem"])
    eng.client = MagicMock()
    eng.client.messages.create.return_value = _fake_response("not json")
    assert eng.tick("THEM: stuff") is None
    assert eng.coverage == {"problem": "missing"}
