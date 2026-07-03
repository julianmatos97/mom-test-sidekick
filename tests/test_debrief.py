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
