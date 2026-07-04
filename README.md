# momtest — Mom Test discovery-call HUD

Terminal heads-up display for customer discovery calls. Listens locally
(mic = you, system audio = prospect), transcribes with Parakeet on-device,
and coaches you live: next Mom Test questions, violation alerts
(pitching / hypotheticals / compliments), coverage tracker, fact capture.
Writes a markdown debrief per call to `calls/`.

## Requirements

- Apple Silicon Mac (Parakeet runs via MLX), macOS 13+
- Xcode command line tools (`xcode-select --install`)
- `uv`
- Default coach model is `codex:gpt-5.4-mini` via the OpenAI Codex CLI (ChatGPT login, no API key). Set `MOMTEST_MODEL` (e.g. `anthropic:claude-haiku-4-5-20251001`, `openai:gpt-5`) to use an API provider instead — then its API key must be in the environment

## Setup

    uv sync
    helper/build.sh          # builds the system-audio tap

First run prompts for Microphone and Screen Recording permissions
(grant to your terminal app, then restart it).

## Use

    uv run momtest start     # asks for prospect + hypothesis, then HUD
    uv run momtest demo      # replay a canned call through the pipeline

Hotkeys: `p` pause coaching, `q` end call + write debrief.

## Tests

    uv run pytest -m "not slow"   # fast unit tests
    uv run pytest -m slow         # real-model ASR integration test (~1.2GB download)
