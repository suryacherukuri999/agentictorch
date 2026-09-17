"""Compare the setups over everything saved in results/eval, and build the README's
results table from the headline runs in results/headline.

python -m hotloop.scoreboard

For planted cases, "recovered" is the share of the slowdown the setup won back:
with t the step time, (t_planted - t_after) / (t_planted - t_clean), from the
measured cost of each planted mistake in results/bench.
"""

import csv
import json
from pathlib import Path
from statistics import mean

from hotloop.agent.models import MODELS

RESULTS = Path(__file__).parent.parent / "results"
SETUPS = ["rules", "llm", "llm-kb", "agent"]
# The setups are compared with the local model; the agent also runs with the others.
MAIN_MODEL = MODELS["ornith-9b"].name
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


def latest(setup: str, model: str | None = None) -> dict[str, dict]:
    """The most recent result per target for one setup, and for one model if given."""
    out = {}
    # Labels end in a UTC timestamp, which orders runs across models.
    for path in sorted((RESULTS / "eval" / setup).glob("*/*.json"), key=lambda p: p.stem[-15:]):
        record = json.loads(path.read_text())
        if model in (None, record["model"]):
            out[record["target"]] = record
    return out


def setup_runs(setup: str) -> dict[str, dict]:
    return latest(setup, None if setup == "rules" else MAIN_MODEL)


def harmful_attempt(attempt: dict) -> bool:
    """Would this change have hurt if it had been applied without measuring?"""
    reason = attempt["reason"]
    if reason.startswith(("run failed", "loss")):
        return True
    speedup = attempt.get("speedup")
    return speedup is not None and speedup < HARM_MARGIN


def outcome(record: dict, setup: str) -> tuple[float, bool]:
    """(speedup that ships, harmful). The agent ships only what passed the keep rule;
    the blind setups ship their patch whatever it does."""
    if setup == "agent":
        return record["speedup"], False
    if not record["attempts"]:
        return 1.0, False
    patch = record["attempts"][-1]
    if patch["reason"].startswith(("run failed", "loss")):
        return 1.0, True
    speedup = record["measured_speedup"]
    return speedup, speedup < HARM_MARGIN


def table() -> str:
    results = {setup: setup_runs(setup) for setup in SETUPS}
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


def totals() -> str:
    """Per setup: targets run, harmful changes shipped and, for the agent, the harmful
    attempts its keep rule caught and reverted."""
    lines = [
        "| Setup | Targets | Harmful changes shipped | Harmful attempts caught |",
        "|---|---|---|---|",
    ]
    for setup in SETUPS:
        records = setup_runs(setup)
        shipped = sum(outcome(r, setup)[1] for r in records.values())
        caught = (
            sum(harmful_attempt(a) for r in records.values() for a in r["attempts"])
            if setup == "agent"
            else ""
        )
        lines.append(f"| {setup} | {len(records)} | {shipped} | {caught} |")
    return "\n".join(lines)


def maintainers() -> dict[str, dict]:
    """The maintainer check per owner/repo: stars and recent outside merges."""
    record = json.loads((RESULTS / "survey" / "maintainers.json").read_text())
    return {r["repo"]: r for r in record["repos"]}


def stars() -> dict[str, int]:
    """GitHub stars per owner/repo, as of the survey or the maintainer check."""
    with (RESULTS / "survey" / "repos.csv").open() as f:
        found = {row["repo"]: int(row["stars"]) for row in csv.DictReader(f)}
    found.update({repo: r["stars"] for repo, r in maintainers().items()})
    return found


def headline() -> str:
    """One row per real repo: the original code against the agent's kept changes,
    from the latest paired run saved for it in results/headline."""
    records = {}
    for path in (RESULTS / "headline").glob("*.json"):
        record = json.loads(path.read_text())
        latest = records.get(record["target"])
        if latest is None or record["recorded"] > latest["recorded"]:
            records[record["target"]] = record
    prs_path = RESULTS / "prs.json"
    prs = json.loads(prs_path.read_text()) if prs_path.exists() else {}
    known = stars()
    checked = maintainers()
    lines = [
        "| Repo | Stars | GPU | Before (ms/step) | After (ms/step) | Speedup | Loss | Fixes "
        "| Pull request |",
        "|---" * 9 + "|",
    ]
    for target, r in sorted(records.items(), key=lambda item: -item[1]["speedup"]):
        repo = r["repo"].removeprefix("https://github.com/")
        before = mean(run["step_ms"] for run in r["baseline"])
        after = mean(run["step_ms"] for run in r["candidate"])
        speedup = f"{r['speedup']:.2f}x" + ("" if r["keep"] else f" ({r['reason']})")
        # The keep rule checks the loss before the speed, so only a loss or failure
        # reason means the loss check did not pass.
        loss = "changed" if r["reason"].startswith(("loss", "run failed")) else "within noise"
        cells = [
            f"[{repo}]({r['repo']})",
            f"{known[repo]:,}" if repo in known else "",
            r["gpu"].removeprefix("NVIDIA "),
            f"{before:.1f}",
            f"{after:.1f}",
            speedup,
            loss,
            r["change"],
            prs.get(target) or _not_sent(checked.get(repo)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def models() -> str:
    """The agent with each model: the speedup it kept per target, the harmful attempts
    its keep rule caught, and what the model cost."""
    runs = {key: latest("agent", model.name) for key, model in MODELS.items()}
    targets = sorted({t for r in runs.values() for t in r})
    lines = ["| Target | " + " | ".join(runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for target in targets:
        cells = [f"{r[target]['speedup']:.2f}x" if target in r else "" for r in runs.values()]
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    caught = [
        sum(harmful_attempt(a) for x in r.values() for a in x["attempts"]) for r in runs.values()
    ]
    cost = [sum(x["llm_cost"] for x in r.values()) for r in runs.values()]
    lines.append("| Harmful attempts caught | " + " | ".join(map(str, caught)) + " |")
    lines.append("| LLM cost | " + " | ".join(f"${c:.2f}" for c in cost) + " |")
    return "\n".join(lines)


def _not_sent(check: dict | None) -> str:
    if check and check["outside_merges_12_months"] == 0:
        return "not sent: no outside pull request merged in 12 months"
    return "not sent"


if __name__ == "__main__":
    print(table())
    print()
    print(totals())
    print()
    print(models())
    print()
    print(headline())
