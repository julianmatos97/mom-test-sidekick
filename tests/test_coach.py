import pytest
from pydantic import ValidationError

from momtest.coach import CoachUpdate, build_user_prompt


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


def test_coach_update_clamps_questions_to_two():
    u = CoachUpdate(questions=["a", "b", "c", "d"])
    assert u.questions == ["a", "b"]


def test_coach_update_defaults_empty():
    u = CoachUpdate()
    assert u.questions == [] and u.alerts == [] and u.coverage == {} and u.facts == []


def test_alert_kind_literal_enforced():
    u = CoachUpdate(alerts=[{"kind": "pitching", "detail": "you pitched"}])
    assert u.alerts[0].kind == "pitching"
    with pytest.raises(ValidationError):
        CoachUpdate(alerts=[{"kind": "sarcasm", "detail": "nope"}])
