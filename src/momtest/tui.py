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
        alerts = Group(*[Text(f"⚠ {a.kind.upper()}: {a.detail}",
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
