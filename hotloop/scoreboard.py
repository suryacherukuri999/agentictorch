"""Compare the setups over everything saved in results/eval.

python -m hotloop.scoreboard

For planted cases, "recovered" is the share of the slowdown the setup won back:
with t the step time, (t_planted - t_after) / (t_planted - t_clean), from the
measured cost of each planted mistake in results/bench.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
SETUPS = ["rules", "llm", "llm-kb", "agent"]
# A blind patch that runs this much slower than the original counts as harmful.
HARM_MARGIN = 0.97


def planted_cost(target: str) -> float | None:
    """How much faster the clean script is than the planted one, if it was measured."""
    path = RESULTS / "bench" / f"{target}-reference.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record["speedup"] if record["keep"] else 1.0


def recovered(speedup: float, cost: float) -> float | None:
    if cost <= 1.0:
        return None
    return (1 - 1 / speedup) / (1 - 1 / cost)


def latest(setup: str) -> dict[str, dict]:
    """The most recent result per target for one setup."""
    out = {}
    for path in sorted((RESULTS / "eval" / setup).glob("*/*.json")):
        record = json.loads(path.read_text())
        out[record["target"]] = record
    return out


def outcome(record: dict, setup: str) -> tuple[float, bool]:
    """(speedup that ships, harmful). The agent ships only what passed the keep rule;
    the blind setups ship their patch whatever it does."""
    if setup == "agent":
        return record["speedup"], False
    if not record["attempts"]:
        return 1.0, False
    reason = record["attempts"][-1]["reason"]
    if reason.startswith(("run failed", "loss")):
        return 1.0, True
    speedup = record["measured_speedup"]
    return speedup, speedup < HARM_MARGIN


def table() -> str:
    results = {setup: latest(setup) for setup in SETUPS}
    targets = sorted({t for r in results.values() for t in r})
    lines = [
        "| Target | Planted cost | " + " | ".join(SETUPS) + " |",
        "|---" * (len(SETUPS) + 2) + "|",
    ]
    for target in targets:
        cost = planted_cost(target)
        cells = []
        for setup in SETUPS:
            record = results[setup].get(target)
            if record is None:
                cells.append("")
                continue
            speedup, harmful = outcome(record, setup)
            share = recovered(speedup, cost) if cost else None
            cell = f"{speedup:.2f}x" + (f" ({share:.0%})" if share is not None else "")
            cells.append(cell + (" harmful" if harmful else ""))
        planted = f"{cost:.2f}x" if cost else ""
        lines.append(f"| {target} | {planted} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
