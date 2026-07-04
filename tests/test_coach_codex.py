"""CodexCoachEngine tests using fake codex executables — no network, no real codex."""
import json
import os
import stat

import pytest

from momtest.coach import CodexCoachEngine, CoachEngine, create_coach

CANNED = json.dumps({
    "questions": ["Ask: when did that last happen?"],
    "alerts": [],
    "coverage": {"problem": "covered"},
    "facts": ["they lost two requests last month"],
})


def make_fake(tmp_path, body: str):
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/bash\n" + body)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return str(fake)


def out_path_script(payload: str) -> str:
    # find the argument after "-o" and write the payload there
    return (
        'out=""\n'
        'prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "-o" ]; then out="$a"; fi\n'
        '  prev="$a"\n'
        'done\n'
        f"printf %s '{payload}' > \"$out\"\n"
        "exit 0\n"
    )


def test_tick_parses_and_merges_coverage(tmp_path):
    fake = make_fake(tmp_path, out_path_script(CANNED))
    engine = CodexCoachEngine("hypo", ["problem", "budget"], binary=fake)
    update = engine.tick("THEM: x")
    assert update is not None
    assert update.questions == ["Ask: when did that last happen?"]
    assert update.coverage == {"problem": "covered"}
    assert engine.coverage == {"problem": "covered", "budget": "missing"}


def test_tick_raises_on_codex_failure(tmp_path):
    fake = make_fake(tmp_path, 'echo "boom" >&2\nexit 1\n')
    engine = CodexCoachEngine("hypo", ["problem"], binary=fake)
    with pytest.raises(RuntimeError):
        engine.tick("THEM: x")


def test_tick_returns_none_on_garbage(tmp_path):
    fake = make_fake(tmp_path, out_path_script("not json at all"))
    engine = CodexCoachEngine("hypo", ["problem"], binary=fake)
    assert engine.tick("THEM: x") is None
    assert engine.coverage == {"problem": "missing"}


def test_factory_codex_with_model(monkeypatch):
    monkeypatch.setenv("MOMTEST_MODEL", "codex:gpt-5.2")
    coach = create_coach("hypo", ["problem"])
    assert isinstance(coach, CodexCoachEngine)
    assert coach.model == "gpt-5.2"


def test_factory_codex_default_model(monkeypatch):
    monkeypatch.setenv("MOMTEST_MODEL", "codex")
    coach = create_coach("hypo", ["problem"])
    assert isinstance(coach, CodexCoachEngine)
    assert coach.model is None


def test_factory_defaults_to_codex_mini(monkeypatch):
    monkeypatch.delenv("MOMTEST_MODEL", raising=False)
    coach = create_coach("hypo", ["problem"])
    assert isinstance(coach, CodexCoachEngine)
    assert coach.model == "gpt-5.4-mini"


def test_factory_pydantic_ai_for_provider_models(monkeypatch):
    monkeypatch.setenv("MOMTEST_MODEL", "anthropic:claude-haiku-4-5-20251001")
    assert isinstance(create_coach("hypo", ["problem"]), CoachEngine)
