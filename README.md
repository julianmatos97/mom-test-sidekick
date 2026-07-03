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
- API key for your model provider in the environment (default `ANTHROPIC_API_KEY`; set `MOMTEST_MODEL` like `openai:gpt-5` to switch providers)

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
