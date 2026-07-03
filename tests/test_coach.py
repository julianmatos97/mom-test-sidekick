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
