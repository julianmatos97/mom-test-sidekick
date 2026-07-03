"""momtest CLI: start (live call) and demo (wav replay)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from momtest.coach import resolve_model
from momtest.session import Session

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo.wav"

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


def check_api_key() -> None:
    model = resolve_model()
    provider = model.split(":", 1)[0]
    var = PROVIDER_ENV_VARS.get(provider)
    if var and not os.environ.get(var):
        sys.exit(f"{var} not set — the coach needs it for model {model!r} "
                 "(set MOMTEST_MODEL to use a different provider).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="momtest", description="Mom Test call HUD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="live call")
    start.add_argument("--prospect", default=None)
    start.add_argument("--hypothesis", default=None)

    demo = sub.add_parser("demo", help="replay a wav fixture through the pipeline")
    demo.add_argument("--fixture", type=Path, default=FIXTURE)

    args = parser.parse_args()

    check_api_key()

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
