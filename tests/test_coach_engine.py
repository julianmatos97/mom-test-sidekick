from pydantic_ai.models.test import TestModel

from momtest.coach import CoachEngine


def test_tick_merges_partial_coverage():
    eng = CoachEngine("hypo", ["problem", "budget"])
    model = TestModel(custom_output_args={
        "questions": [], "alerts": [], "coverage": {"problem": "covered"}, "facts": []})
    with eng.agent.override(model=model):
        update = eng.tick("THEM: stuff")
    assert eng.coverage == {"problem": "covered", "budget": "missing"}
    assert update.coverage == {"problem": "covered"}


def test_tick_empty_coverage_keeps_existing():
    eng = CoachEngine("hypo", ["problem"])
    model = TestModel(custom_output_args={
        "questions": ["Ask: when did that last happen?"],
        "alerts": [], "coverage": {}, "facts": []})
    with eng.agent.override(model=model):
        update = eng.tick("THEM: stuff")
    assert eng.coverage == {"problem": "missing"}
    assert update.questions == ["Ask: when did that last happen?"]
