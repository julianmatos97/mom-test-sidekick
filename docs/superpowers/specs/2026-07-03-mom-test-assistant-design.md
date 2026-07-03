# Mom Test Assistant — Design

Date: 2026-07-03
Status: Approved

## Purpose

Real-time heads-up display for customer discovery calls. Listens to the call locally, transcribes it, and coaches the user live with Mom Test-style prompts: suggested next questions, violation alerts (pitching, hypotheticals, compliments), a coverage tracker, and captured facts. Produces a structured debrief after the call.

## Decisions made

- **Audio source**: local capture, not Granola. Mic + macOS system-audio process tap. Granola remains the user's note-taker but is not a dependency.
- **ASR**: Parakeet-TDT 0.6B v2 via `parakeet-mlx`, running locally on Apple Silicon. One streaming instance per audio channel.
- **Speaker attribution**: two physical channels (mic = you, system tap = them). No diarization model.
- **HUD**: terminal TUI (Rich `Live`), not a browser or native overlay.
- **Coach model**: Claude Haiku via Anthropic API for ~1s latency.

## Architecture

Python CLI `momtest`, single process, three async loops:

```
mic audio ──┐
            ├─→ Parakeet-MLX (2 streams) ─→ transcript buffer ─→ coach loop ─→ TUI
system tap ─┘         (you / them)          (speaker, text, ts)   (Claude API)
```

### Components

1. **Audio capture**
   - Mic: `sounddevice`, 16 kHz mono PCM.
   - Prospect: macOS Core Audio process tap via a small Swift helper binary that streams PCM to stdout. Captures output of Zoom/Meet/any app. No BlackHole or virtual device install.
2. **ASR**: `parakeet-mlx` streaming transcription, one instance per channel. Emits finalized utterances with timestamps.
3. **Transcript buffer**: append-only list of `{speaker: "you"|"them", text, ts}`. Maintains a rolling window (last ~2 min verbatim) plus a running summary for older content.
4. **Coach engine**: triggers every ~15 s or when the prospect finishes a turn. Sends rolling window + running summary + pre-call hypothesis + coverage state to Claude Haiku. Structured JSON response:
   - `questions[]` — max 2, short imperative ("Ask: when did that last happen?")
   - `alerts[]` — Mom Test violations: user pitching, prospect giving compliments/hypotheticals/generalities, fluff answers accepted
   - `coverage{}` — status per discovery goal
   - `facts[]` — concrete extracted facts (numbers, tools, past events)
5. **TUI**: Rich `Live`, four panels — suggested questions (largest, top), violation alerts (red, prominent), coverage checklist, captured facts. Hotkeys: `p` pause, `q` end call.

## Session flow

1. `momtest start` — prompts: what hypothesis are you testing? who is the prospect? Selects discovery goals for the coverage tracker (problem, current solution, cost of problem, budget/authority — editable).
2. Live call — HUD updates continuously.
3. `q` — writes markdown debrief to `./calls/YYYY-MM-DD-<prospect>.md`: full transcript, captured facts, coverage gaps, Mom Test scorecard (pitch count, hypotheticals accepted, past-behavior questions asked).
4. `momtest demo` — replays a WAV fixture through the full pipeline for smoke testing without a live call.

## Coach prompt (core logic)

System prompt encodes Mom Test rules:

- Ask about past behavior, not future hypotheticals.
- Dig into specifics ("when did that last happen?", "walk me through it").
- Compliments are worthless data — flag and redirect.
- Detect when the user is pitching instead of listening.
- Chase concrete facts: money, time, tools, named events.

Input per request: last ~2 min verbatim transcript, running summary, hypothesis, coverage state. Output: the structured JSON above.

## Error handling

- Audio device lost → red banner, HUD stays alive, auto-reconnect loop.
- Claude API failure → retry with backoff; keep last suggestions visible with a staleness indicator.
- ASR falls behind → drop oldest unprocessed audio; capture never blocks.
- Missing Screen Recording permission (needed for system tap) → first-run check with clear setup instructions.

## Testing

- **Coach engine**: unit tests with canned transcripts; assert expected alerts fire (pitching detected, compliment flagged) and question suggestions are past-behavior shaped.
- **ASR pipeline**: WAV fixture playback in place of live audio devices.
- **End-to-end**: `momtest demo` manual smoke test.

## Risks

- Swift audio-tap helper is the hairiest piece — spike it standalone first, before anything else.
- Parakeet degrades on crosstalk/background music; acceptable for typical calls.
- Parakeet is English-only.

## Out of scope (v1)

- Windows/Linux support (macOS only).
- Diarization within a channel (multiple prospects on one call share the "them" channel).
- Granola integration.
- Non-terminal UI.
