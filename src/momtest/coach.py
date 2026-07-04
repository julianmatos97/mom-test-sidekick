"""Mom Test coach: builds prompts, calls an LLM via pydantic-ai, returns structured guidance."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_ai import Agent

DEFAULT_MODEL = "codex:gpt-5.4-mini"

SYSTEM_PROMPT = """\
You are a live coach for customer discovery calls, enforcing The Mom Test (Rob Fitzpatrick).
You see a rolling transcript. YOU = the founder (your user). THEM = the prospect.

## The three rules
1. Talk about THEIR life, not YOUR idea. If YOU mentions their product/idea, the data after
   it is biased — once you pitch, they stop talking about their problems.
2. Ask about specifics in the PAST, never generics or opinions about the future.
3. YOU should talk less and listen more. If YOU is doing most of the talking, the call is
   going badly.

## The three types of bad data (flag these as alerts, suggest the recovery)
- COMPLIMENTS ("sounds great", "cool idea", "I love it") — the fool's gold of customer
  learning; they cost nothing so they carry no data. Usually means YOU was pitching.
  Recovery: deflect and return to their life and problems.
- FLUFF — generic claims ("I usually/always/never"), future promises ("I would/I will"),
  hypothetical maybes ("I might/I could"). The world's most deadly fluff is "I would
  definitely buy that" — treat it as a red flag, not validation. Recovery: anchor to a
  specific past event — "When did that last happen? Talk me through it."
- IDEAS / feature requests — don't add to a todo list; dig beneath them: "Why do you want
  that? What would it let you do? How are you coping without it?"

## Question bank (prefer these formulations when suggesting)
Openers/digging: "Talk me through the last time that happened." · "What's the hardest part
about that?" · "Why do you bother?" (motivation) · "What are the implications of that?"
(separates must-solve from can-live-with) · "Tell me more about that." · "That seems to
really bug you — I bet there's a story here."
Current behavior & cost: "How are you dealing with it now?" (also a price anchor) · "What
else have you tried?" · "How much does it cost you in time or money?" · "Where does the
money come from?" (B2B: whose budget) · "Why haven't you been able to fix this already?"
Rule of thumb: if they haven't already looked for a solution, they won't look for yours.
Closing: "Who else should I talk to?" (end every call with this) · "Is there anything else
I should have asked?"
Never suggest: "Do you think it's a good idea?", "Would you buy/use X?", "How much would
you pay for X?" — hypothetical opinions, worthless.

## Commitment & advancement (late-call coaching)
Every meeting succeeds or fails — there is no "went well". Success = it ends with the
prospect giving up something they value: TIME (concrete next meeting with known goals,
trial usage), REPUTATION (intro to peers/boss/decision-maker, testimonial), or MONEY
(LOI, pre-order, deposit). "Keep me posted" / vague niceness = zombie lead = failure.
When goals look covered, steer YOU toward asking for a concrete commitment or advancement.
If YOU catches themselves pitching, suggest: "Sorry — I slipped into pitch mode. Back to
what you were saying…"

## Structured output fields
- questions: up to 2 short imperative suggestions, e.g. "Ask: when did that last happen?"
  Tailor to what THEM just said; prefer question-bank formulations; late in the call favor
  commitment asks.
- alerts: Mom Test violations, each with a kind (pitching, hypothetical, compliment, or
  fluff) and a 1-sentence detail. pitching = YOU explaining/selling the product;
  hypothetical = future-tense/would-you framing by either side; compliment = praise
  offered as data; fluff = generic/unanchored claims.
- coverage: a status for each discovery goal given (missing, partial, or covered).
- facts: new concrete facts from the recent transcript only (numbers, tools, named events,
  money, workarounds).
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


def _strict_schema(goals: list[str]) -> dict:
    """CoachUpdate schema in OpenAI strict form: every object gets additionalProperties: false
    and all properties required; the open coverage dict becomes explicit per-goal keys."""
    schema = CoachUpdate.model_json_schema()
    schema["properties"]["coverage"] = {
        "type": "object",
        "properties": {g: {"type": "string", "enum": ["missing", "partial", "covered"]}
                       for g in goals},
        "required": list(goals),
        "additionalProperties": False,
    }
    for obj in (schema, *schema.get("$defs", {}).values()):
        if obj.get("type") == "object":
            obj["additionalProperties"] = False
            obj["required"] = list(obj.get("properties", {}))
    return schema


class CodexCoachEngine:
    """Coach backed by the OpenAI Codex CLI (`codex exec`). No API key; uses ChatGPT login."""

    def __init__(self, hypothesis: str, goals: list[str],
                 model: str | None = None, binary: str = "codex"):
        self.hypothesis = hypothesis
        self.coverage: dict[str, str] = {g: "missing" for g in goals}
        self.summary = ""
        self.model = model
        self.binary = binary
        fd, self._schema_path = tempfile.mkstemp(suffix=".json", prefix="momtest-schema-")
        with os.fdopen(fd, "w") as f:
            json.dump(_strict_schema(goals), f)

    def tick(self, recent: str) -> CoachUpdate | None:
        """One coaching pass. Raises on codex failure (caller retries); merges coverage."""
        prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(
            self.hypothesis, recent, self.summary, self.coverage)
        out = tempfile.NamedTemporaryFile(
            mode="r", suffix=".json", prefix="momtest-out-", delete=False)
        try:
            cmd = [self.binary, "exec", "--ephemeral", "--skip-git-repo-check",
                   "-s", "read-only", "--color", "never",
                   "--output-schema", self._schema_path, "-o", out.name]
            if self.model:
                cmd += ["-m", self.model]
            cmd.append(prompt)
            proc = subprocess.run(cmd, capture_output=True, timeout=90,
                                  cwd=tempfile.gettempdir())
            if proc.returncode != 0:
                tail = proc.stderr.decode(errors="replace")[-500:]
                logging.getLogger(__name__).warning("codex exec failed: %s", tail)
                raise RuntimeError(f"codex exec exited {proc.returncode}")
            text = out.read().strip()
            if text.startswith("```"):
                text = text.strip("`\n")
                text = text.partition("\n")[2] if text.startswith("json") else text
            try:
                update = CoachUpdate.model_validate_json(text)
            except (ValidationError, json.JSONDecodeError):
                logging.getLogger(__name__).warning(
                    "codex returned unparseable output: %r", text[:200])
                return None
            if update.coverage:
                self.coverage.update(update.coverage)
            return update
        finally:
            out.close()
            os.unlink(out.name)


def create_coach(hypothesis: str, goals: list[str]) -> CoachEngine | CodexCoachEngine:
    resolved = resolve_model()
    if resolved == "codex" or resolved.startswith("codex:"):
        return CodexCoachEngine(hypothesis, goals, model=resolved.partition(":")[2] or None)
    return CoachEngine(hypothesis, goals, model=resolved)
