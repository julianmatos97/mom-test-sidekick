"""Mom Test coach: builds prompts, calls an LLM via pydantic-ai, returns structured guidance."""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a live coach for customer discovery calls, enforcing The Mom Test (Rob Fitzpatrick).
You see a rolling transcript. YOU = the founder (your user). THEM = the prospect.

Rules you enforce:
- Ask about past behavior and specifics, never hypotheticals ("would you...") .
- Compliments and generic praise are worthless data — flag them, redirect to facts.
- If YOU is pitching or explaining their product instead of listening, flag it.
- Chase concrete facts: money, time, tools, named events ("when did that last happen?",
  "walk me through the last time", "what else did you try?", "who else should I talk to?").

Fill each field of the structured output as follows:
- questions: up to 2 short imperative suggestions, e.g. "Ask: when did that last happen?"
- alerts: Mom Test violations, each with a kind (pitching, hypothetical, compliment, or fluff)
  and a 1-sentence detail.
- coverage: a status for each discovery goal given (missing, partial, or covered).
- facts: new concrete facts from the recent transcript only.
Alerts only for things happening in the RECENT transcript. Empty lists are fine.
"""


class Alert(BaseModel):
    kind: Literal["pitching", "hypothetical", "compliment", "fluff"]
    detail: str


class CoachUpdate(BaseModel):
    questions: list[str] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    coverage: dict[str, str] = Field(default_factory=dict)
    facts: list[str] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def _clamp_questions(cls, v: list[str]) -> list[str]:
        return v[:2]


def resolve_model() -> str:
    return os.environ.get("MOMTEST_MODEL", DEFAULT_MODEL)


def build_user_prompt(hypothesis: str, recent: str, summary: str, coverage: dict[str, str]) -> str:
    return (
        f"Hypothesis being tested: {hypothesis}\n\n"
        f"Discovery goals and current coverage:\n{json.dumps(coverage)}\n\n"
        f"Summary of earlier conversation:\n{summary or '(call just started)'}\n\n"
        f"Recent transcript (last ~2 min):\n{recent or '(no speech yet)'}"
    )


class CoachEngine:
    def __init__(self, hypothesis: str, goals: list[str], model: str | None = None):
        self.agent: Agent[None, CoachUpdate] = Agent(
            model or resolve_model(),
            output_type=CoachUpdate,
            system_prompt=SYSTEM_PROMPT,
            defer_model_check=True,
        )
        self.hypothesis = hypothesis
        self.coverage: dict[str, str] = {g: "missing" for g in goals}
        self.summary = ""

    def tick(self, recent: str) -> CoachUpdate | None:
        """One coaching pass. Raises on model failure (caller retries); merges coverage."""
        result = self.agent.run_sync(build_user_prompt(
            self.hypothesis, recent, self.summary, self.coverage))
        update = result.output
        if update.coverage:
            self.coverage.update(update.coverage)
        return update
