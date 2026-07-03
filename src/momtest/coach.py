"""Mom Test coach: builds prompts, calls Claude, parses structured guidance."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a live coach for customer discovery calls, enforcing The Mom Test (Rob Fitzpatrick).
You see a rolling transcript. YOU = the founder (your user). THEM = the prospect.

Rules you enforce:
- Ask about past behavior and specifics, never hypotheticals ("would you...") .
- Compliments and generic praise are worthless data — flag them, redirect to facts.
- If YOU is pitching or explaining their product instead of listening, flag it.
- Chase concrete facts: money, time, tools, named events ("when did that last happen?",
  "walk me through the last time", "what else did you try?", "who else should I talk to?").

Respond with ONLY a JSON object, no prose:
{
  "questions": [up to 2 short imperative suggestions, e.g. "Ask: when did that last happen?"],
  "alerts": [{"kind": "pitching"|"hypothetical"|"compliment"|"fluff", "detail": "<1 short sentence>"}],
  "coverage": {"<goal>": "missing"|"partial"|"covered", ...for each goal given},
  "facts": [new concrete facts from the recent transcript only]
}
Alerts only for things happening in the RECENT transcript. Empty lists are fine.
"""


@dataclass
class CoachUpdate:
    questions: list[str] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)


def build_user_prompt(hypothesis: str, recent: str, summary: str, coverage: dict[str, str]) -> str:
    return (
        f"Hypothesis being tested: {hypothesis}\n\n"
        f"Discovery goals and current coverage:\n{json.dumps(coverage)}\n\n"
        f"Summary of earlier conversation:\n{summary or '(call just started)'}\n\n"
        f"Recent transcript (last ~2 min):\n{recent or '(no speech yet)'}"
    )


def parse_response(raw: str) -> CoachUpdate | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return CoachUpdate(
        questions=[str(q) for q in data.get("questions", [])][:2],
        alerts=[a for a in data.get("alerts", []) if isinstance(a, dict)],
        coverage={str(k): str(v) for k, v in data.get("coverage", {}).items()},
        facts=[str(f) for f in data.get("facts", [])],
    )


class CoachEngine:
    def __init__(self, hypothesis: str, goals: list[str]):
        self.client = anthropic.Anthropic()
        self.hypothesis = hypothesis
        self.coverage: dict[str, str] = {g: "missing" for g in goals}
        self.summary = ""

    def tick(self, recent: str) -> CoachUpdate | None:
        """One coaching pass. Returns None on parse failure (caller keeps last state)."""
        msg = self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(
                self.hypothesis, recent, self.summary, self.coverage)}],
        )
        update = parse_response(msg.content[0].text)
        if update and update.coverage:
            self.coverage.update(update.coverage)
        return update
