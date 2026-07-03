"""Post-call markdown debrief."""
from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path

from momtest.coach import CoachUpdate
from momtest.transcript import Transcript


def write_debrief(
    out_dir: Path,
    prospect: str,
    hypothesis: str,
    transcript: Transcript,
    state: CoachUpdate,
    alert_history: list[dict],
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", prospect.lower()).strip("-") or "call"
    path = Path(out_dir) / f"{date.today().isoformat()}-{slug}.md"
    counts = Counter(a.get("kind", "?") for a in alert_history)
    gaps = [g for g, s in state.coverage.items() if s != "covered"]

    lines = [
        f"# Discovery call — {prospect}",
        f"\n**Date:** {date.today().isoformat()}",
        f"**Hypothesis:** {hypothesis}",
        "\n## Facts captured",
        *([f"- {f}" for f in state.facts] or ["- (none)"]),
        "\n## Coverage",
        *[f"- {g}: {s}" for g, s in state.coverage.items()],
        "\n## Gaps to chase next time",
        *([f"- {g}" for g in gaps] or ["- none — full coverage"]),
        "\n## Mom Test scorecard (violations flagged)",
        *([f"- {kind}: {n}" for kind, n in counts.items()] or ["- clean call"]),
        "\n## Transcript",
        "```",
        Transcript.render(transcript.utterances),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
