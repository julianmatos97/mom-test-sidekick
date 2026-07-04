"""Wires audio → asr → transcript → coach → hud. Blocking run loop."""
from __future__ import annotations

import logging
import queue
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from rich.console import Console
from rich.live import Live

from momtest import audio
from momtest.asr import ASRWorker
from momtest.coach import CoachUpdate, create_coach
from momtest.debrief import write_debrief
from momtest.transcript import Transcript
from momtest.tui import HudState, build_hud

COACH_INTERVAL_S = 8.0
DEFAULT_GOALS = ["problem", "current solution", "cost of problem", "budget & authority"]


class Session:
    def __init__(self, prospect: str, hypothesis: str, sources: list, channels: dict):
        self.prospect = prospect
        self.hypothesis = hypothesis
        self.sources = sources          # objects with .start()/.stop()
        self.channels = channels        # {"you": Queue, "them": Queue}
        self.transcript = Transcript()
        self.coach = create_coach(hypothesis, DEFAULT_GOALS)
        self.state = HudState(update=CoachUpdate(coverage=self.coach.coverage.copy()))
        self.alert_history: list[dict] = []
        self.paused = False
        self.quit = threading.Event()

    # -- construction helpers -------------------------------------------------

    @classmethod
    def live(cls, prospect: str, hypothesis: str) -> "Session":
        you_q, them_q = queue.Queue(maxsize=100), queue.Queue(maxsize=100)
        return cls(prospect, hypothesis,
                   sources=[audio.MicCapture(you_q), audio.SystemCapture(them_q)],
                   channels={"you": you_q, "them": them_q})

    @classmethod
    def demo(cls, fixture: Path) -> "Session":
        them_q: queue.Queue = queue.Queue(maxsize=100)
        return cls("Demo", "demo hypothesis",
                   sources=[audio.FileCapture(them_q, fixture)],
                   channels={"them": them_q})

    # -- run loop --------------------------------------------------------------

    def on_utterance(self, speaker: str, text: str, ts: float) -> None:
        self.transcript.add(speaker, text, ts)
        self.state.last_heard = f"{speaker}: {text}"

    def coach_loop(self) -> None:
        while not self.quit.wait(COACH_INTERVAL_S):
            if self.paused or not self.transcript.utterances:
                continue
            window = self.transcript.window(120)
            recent = Transcript.render(window)
            older = self.transcript.utterances[: len(self.transcript.utterances) - len(window)]
            self.coach.summary = Transcript.render(older)[-1200:]  # cheap "summary": older tail
            for attempt in range(3):
                try:
                    update = self.coach.tick(recent)
                    break
                except Exception:
                    logging.getLogger(__name__).warning("coach tick failed", exc_info=True)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                self.state.status = "error"
                continue
            if update is None:
                self.state.status = "stale"   # parse failed; keep last guidance
                continue
            self.state.status = "listening"
            update.coverage = self.coach.coverage.copy()
            self.state.update = update
            for f in update.facts:
                if f not in self.state.facts:
                    self.state.facts.append(f)
            self.alert_history.extend(a.model_dump() for a in update.alerts)

    def keys_loop(self) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit.is_set():
                ch = sys.stdin.read(1)
                if ch == "q":
                    self.quit.set()
                elif ch == "p":
                    self.paused = not self.paused
                    self.state.status = "paused" if self.paused else "listening"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self) -> Path | None:
        console = Console()
        console.print("[bold]loading parakeet…[/bold] (first run downloads ~1.2GB)")
        worker = ASRWorker(self.channels, self.on_utterance)
        worker.start()
        waited = 0.0
        while not worker.ready.wait(timeout=1.0):
            waited += 1.0
            if not worker.is_alive():
                sys.exit("parakeet model failed to load (see log)")
            if waited >= 600:
                sys.exit("timed out loading parakeet model")
        for s in self.sources:
            s.start()
        threading.Thread(target=self.coach_loop, daemon=True).start()
        old_term = termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
        threading.Thread(target=self.keys_loop, daemon=True).start()

        last_reconnect = 0.0
        try:
            with Live(build_hud(self.state), console=console, refresh_per_second=4) as live:
                while not self.quit.is_set():
                    # auto-reconnect a dead system tap (spec: banner + reconnect loop)
                    for s in self.sources:
                        if isinstance(s, audio.SystemCapture) and not s.alive():
                            self.state.status = "error"
                            if time.time() - last_reconnect > 5.0:
                                last_reconnect = time.time()
                                try:
                                    s.start()
                                    self.state.status = "listening"
                                except Exception:
                                    logging.getLogger(__name__).warning(
                                        "system tap reconnect failed", exc_info=True)
                    live.update(build_hud(self.state))
                    time.sleep(0.25)
        except KeyboardInterrupt:
            self.quit.set()
        finally:
            if old_term is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)

        for s in self.sources:
            s.stop()
        worker.stop()
        if not self.transcript.utterances:
            console.print("no speech captured — no debrief written")
            return None
        final = CoachUpdate(coverage=self.coach.coverage,
                            facts=self.state.facts, alerts=[], questions=[])
        path = write_debrief(Path("calls"), self.prospect, self.hypothesis,
                             self.transcript, final, self.alert_history)
        console.print(f"debrief: [bold]{path}[/bold]")
        return path
