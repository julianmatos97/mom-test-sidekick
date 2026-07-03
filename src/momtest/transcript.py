"""Append-only call transcript with a rolling recency window."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Utterance:
    speaker: str  # "you" | "them"
    text: str
    ts: float     # epoch seconds


@dataclass
class Transcript:
    utterances: list[Utterance] = field(default_factory=list)

    def add(self, speaker: str, text: str, ts: float | None = None) -> None:
        self.utterances.append(Utterance(speaker, text, ts if ts is not None else time.time()))

    def window(self, seconds: float = 120.0, now: float | None = None) -> list[Utterance]:
        cutoff = (now if now is not None else time.time()) - seconds
        return [u for u in self.utterances if u.ts >= cutoff]

    def last_speaker(self) -> str | None:
        return self.utterances[-1].speaker if self.utterances else None

    @staticmethod
    def render(utterances: list[Utterance]) -> str:
        return "\n".join(f"{u.speaker.upper()}: {u.text}" for u in utterances)
