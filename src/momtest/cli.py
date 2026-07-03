"""momtest CLI: start (live call) and demo (wav replay)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from momtest.session import Session

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.wav"


def main() -> None:
    parser = argparse.ArgumentParser(prog="momtest", description="Mom Test call HUD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="live call")
    start.add_argument("--prospect", default=None)
    start.add_argument("--hypothesis", default=None)

    demo = sub.add_parser("demo", help="replay a wav fixture through the pipeline")
    demo.add_argument("--fixture", type=Path, default=FIXTURE)

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — the coach needs it.")

    if args.cmd == "start":
        prospect = args.prospect or input("Prospect name: ").strip() or "unknown"
        hypothesis = args.hypothesis or input("Hypothesis you're testing: ").strip()
        session = Session.live(prospect, hypothesis)
    else:
        if not args.fixture.exists():
            sys.exit(f"fixture not found: {args.fixture}")
        session = Session.demo(args.fixture)

    session.run()


if __name__ == "__main__":
    main()
