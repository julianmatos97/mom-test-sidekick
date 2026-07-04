# momtest

**Discovery call assistant for shy nerds.**

You finally got a real human on a call. They're telling you your idea is *amazing* and
they would *definitely buy it*. You're glowing. You are also — per Rob Fitzpatrick's
[The Mom Test](https://www.momtestbook.com/) — being lied to, politely, and it's your
own fault, because you asked lie-inducing questions.

`momtest` is a terminal HUD that sits next to your Zoom window and coaches you through
the call in real time. It listens locally, transcribes on-device, and every few seconds
tells you:

- **what to ask next** — actual Mom Test formulations ("Ask: when did that last happen?"),
  tailored to what the prospect just said
- **when you're blowing it** — red alerts for pitching, hypotheticals, compliments, and
  fluff ("I would definitely buy that" = the world's most deadly fluff, not validation)
- **what you still haven't learned** — a coverage checklist (problem, current solution,
  cost, budget)
- **the facts worth keeping** — numbers, tools, named events, extracted as they're said

When you hit `q`, it writes a markdown debrief: full transcript, captured facts, coverage
gaps, and a scorecard of how many times you pitched. It's like a fitness tracker, but for
shutting up and listening.

```
┌─ ASK NEXT — listening · next tick 6s ─────────────────────────┐
│ Ask: walk me through the last time you lost a request.        │
│ Ask: how are you dealing with it now?                         │
├─ MOM TEST ALERTS ─────────────────────────────────────────────┤
│ ⚠ PITCHING: you just explained your dashboard for 40 seconds  │
├─ COVERAGE ─────────────────┬─ FACTS ──────────────────────────┤
│ ✓ problem                  │ • tracks requests in spreadsheet │
│ ~ current solution         │ • lost two requests last month   │
│ ✗ cost of problem          │ • tenant escalated to owner      │
│ ✗ budget & authority       │                                  │
└────────────────────────────┴──────────────────────────────────┘
```

## How it works

Everything audio stays on your Mac. Your mic is one channel (that's YOU), system audio is
the other (that's THEM — Zoom, Meet, whatever makes sound). Both streams go through
[Parakeet](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) running locally via
MLX. Only the resulting *text* goes to an LLM for coaching.

```
mic ──────┐
          ├─→ Parakeet (on-device ASR) ─→ transcript ─→ LLM coach ─→ HUD
system ───┘                                              every 8s
audio tap
```

The coach's system prompt is built from a proper research pass over the book — the three
rules, the bad-data taxonomy with recovery moves, the book's verbatim question bank, and
the commitment/advancement framework. Receipts in
[`docs/research/mom-test-digest.md`](docs/research/mom-test-digest.md).

## Requirements

- Apple Silicon Mac, macOS 13+ (Parakeet runs via MLX; sorry, everyone else)
- Xcode command line tools: `xcode-select --install`
- [uv](https://docs.astral.sh/uv/)
- A coach model (pick one):
  - **Default, zero config**: the [Codex CLI](https://github.com/openai/codex) with a
    ChatGPT login — `brew install codex && codex login`. Uses `gpt-5.4-mini`.
  - **Any API provider**: set `MOMTEST_MODEL` (e.g. `anthropic:claude-haiku-4-5-20251001`,
    `openai:gpt-5`) and export that provider's API key. Anything
    [pydantic-ai](https://ai.pydantic.dev) speaks, we speak.

## Setup

```bash
uv sync
helper/build.sh      # builds the tiny Swift system-audio tap
```

Two macOS permission prompts on first run — both go to your **terminal app**:

1. **Microphone** — so it can hear you.
2. **Screen & System Audio Recording** — so it can hear *them*. (That's how macOS gates
   system-audio capture. No, we are not recording your screen. The tap is ~80 lines of
   Swift in [`helper/AudioTap.swift`](helper/AudioTap.swift); read it.)

Grant both, then **fully restart the terminal app**. First launch also downloads the
~1.2GB Parakeet model, once.

## Use

```bash
uv run momtest start    # asks who you're talking to + what hypothesis you're testing
uv run momtest demo     # no call? replay a canned prospect through the whole pipeline
```

Put the terminal next to your call window. Glance, don't read. Hotkeys: `p` pauses
coaching (for when they ask if you're typing), `q` ends the call and writes the debrief
to `calls/`.

Switching models mid-life-crisis:

```bash
MOMTEST_MODEL=codex:gpt-5.4 uv run momtest start
MOMTEST_MODEL=anthropic:claude-haiku-4-5-20251001 uv run momtest start
```

## Honest limitations

- English only (Parakeet's, not our, worldview).
- Two channels means one "THEM" — a 3-person call gets merged into one very confusing
  prospect.
- The coach ticks every 8 seconds; it will not save you from a sentence you are currently
  halfway through regretting.
- It cannot make you ask for the sale. It will, however, nag you about it.

## Privacy

Audio never leaves your machine. Transcript text goes to whichever LLM you configured —
that's the only network traffic. Debriefs are local markdown files in `calls/`.

## Tests

```bash
uv run pytest -m "not slow"   # fast unit tests
uv run pytest -m slow         # real-model ASR integration test (downloads Parakeet)
```

## Contributing

Issues and PRs welcome. Especially wanted: Windows/Linux audio capture, diarization for
multi-prospect calls, non-English ASR. Keep PRs small and testable; `uv run pytest -m
"not slow"` must pass.

## Credit

The method is Rob Fitzpatrick's [The Mom Test](https://www.momtestbook.com/) — buy it,
it's short and it will pay for itself on your next call. This project is not affiliated
with or endorsed by the author; it's just fans with a terminal.
