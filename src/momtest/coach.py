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
