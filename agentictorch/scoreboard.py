"""Compare the setups and models over everything saved in results/eval, and build the
README's results tables.

python -m agentictorch.scoreboard

For planted cases, "recovered" is the share of the slowdown the setup won back:
with t the step time, (t_planted - t_after) / (t_planted - t_clean), from the
measured slowdown of each planted mistake in results/bench.
"""

import csv
import json
from pathlib import Path
from statistics import geometric_mean, mean

from agentictorch import targets
from agentictorch.agent.models import MODELS

RESULTS = Path(__file__).parent.parent / "results"
SETUPS = ["rules", "llm", "llm-kb", "agent"]
# The setups were compared with gpt-6-luna. The planted suite ran with the local model,
# which turned out too slow for the real repos.
MAIN_MODEL = MODELS["gpt-6-luna"].name
SOL_MODEL = MODELS["gpt-6.1-sol"].name
LOCAL_MODEL = MODELS["ornith-9b"].name
# A blind patch that runs this much slower than the original counts as harmful.
HARM_MARGIN = 0.97


def planted_slowdown(target: str) -> float | None:
    """How much faster the clean script is than the planted one, if it was measured."""
    path = RESULTS / "bench" / f"{target}-reference.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record["speedup"] if record["keep"] else 1.0


def recovered(speedup: float, slowdown: float) -> float | None:
    if slowdown <= 1.0:
        return None
    return (1 - 1 / speedup) / (1 - 1 / slowdown)


def latest(setup: str, model: str | None = None) -> dict[str, dict]:
    """The most recent result per target for one setup, and for one model if given."""
    out = {}
    # Labels end in a UTC timestamp, which orders runs across models.
    for path in sorted((RESULTS / "eval" / setup).glob("*/*.json"), key=lambda p: p.stem[-15:]):
        record = json.loads(path.read_text())
        if model in (None, record["model"]):
            out[record["target"]] = record
    return out


def setup_runs(setup: str, model: str = MAIN_MODEL) -> dict[str, dict]:
    return latest(setup, None if setup == "rules" else model)


def compared(model: str) -> dict[str, dict[str, dict]]:
    """Each setup's runs, limited to the targets every setup ran with this model."""
    runs = {setup: setup_runs(setup, model) for setup in SETUPS}
    common = set.intersection(*(set(r) for r in runs.values()))
    return {setup: {t: r[t] for t in r if t in common} for setup, r in runs.items()}


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


def table(model: str = MAIN_MODEL) -> str:
    results = compared(model)
    targets = sorted(results["agent"])
    # Only planted cases have a known slowdown; real repos leave the column out.
    planted = any(planted_slowdown(t) for t in targets)
    columns = ["Target", *(["Planted slowdown"] if planted else []), *SETUPS]
    lines = ["| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
    for target in targets:
        slowdown = planted_slowdown(target)
        cells = [f"{slowdown:.2f}x" if slowdown else ""] if planted else []
        for setup in SETUPS:
            record = results[setup].get(target)
            if record is None:
                cells.append("")
                continue
            speedup, harmful = outcome(record, setup)
            share = recovered(speedup, slowdown) if slowdown else None
            cell = f"{speedup:.2f}x" + (f" ({share:.0%})" if share is not None else "")
            cells.append(cell + (" harmful" if harmful else ""))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def totals(model: str = MAIN_MODEL) -> str:
    """Per setup, over the targets every setup ran: the typical speedup that ships (a crash
    counts as 1.00x), harmful changes shipped, and harmful attempts the agent caught."""
    lines = [
        "| Setup | Targets | Speedup (geometric mean) | Harmful changes shipped "
        "| Harmful attempts caught |",
        "|---|---|---|---|---|",
    ]
    for setup, records in compared(model).items():
        outcomes = [outcome(r, setup) for r in records.values()]
        typical = f"{geometric_mean(s for s, _ in outcomes):.2f}x" if outcomes else ""
        shipped = sum(harmful for _, harmful in outcomes)
        caught = (
            sum(harmful_attempt(a) for r in records.values() for a in r["attempts"])
            if setup == "agent"
            else ""
        )
        lines.append(f"| {setup} | {len(records)} | {typical} | {shipped} | {caught} |")
    return "\n".join(lines)


def maintainers() -> dict[str, dict]:
    record = json.loads((RESULTS / "survey" / "maintainers.json").read_text())
    return {r["repo"]: r for r in record["repos"]}


def stars() -> dict[str, int]:
    """GitHub stars per owner/repo, as of the survey or the maintainer check."""
    with (RESULTS / "survey" / "repos.csv").open() as f:
        found = {row["repo"]: int(row["stars"]) for row in csv.DictReader(f)}
    found.update({repo: r["stars"] for repo, r in maintainers().items()})
    return found


def shipped(record: dict) -> float:
    """What a run ships: its speedup if the keep rule kept it, else the original's 1.00x."""
    kept = record["keep"] if "keep" in record else bool(record["kept"])
    return record["speedup"] if kept else 1.0


def sol_agent_runs() -> dict[str, dict]:
    """The latest gpt-6.1-sol agent run per target. A lost report falls back to its final
    pair, measured again afterwards."""
    runs = latest("agent", SOL_MODEL)
    for path in (RESULTS / "final-pairs").glob("*.json"):
        record = json.loads(path.read_text())
        runs.setdefault(record["target"], record)
    return runs


def _comparison() -> tuple[list[tuple[str, dict, bool]], list[str]]:
    """Each approach as (header, runs, blind), and the real repos all of them ran."""
    # A blind patch ships whatever it does; the agent keeps measured wins only.
    columns = [
        ("Agent, gpt-6.1-sol", sol_agent_runs(), False),
        ("One patch, gpt-6.1-sol", latest("llm", SOL_MODEL), True),
        ("Agent, gpt-6-luna", latest("agent", MAIN_MODEL), False),
        ("One patch, gpt-6-luna", latest("llm", MAIN_MODEL), True),
    ]
    common = set.intersection(*(set(runs) for _, runs, _ in columns))
    return columns, sorted(t for t in common if not t.startswith("bench-"))


def _ships(runs: dict, blind: bool, target: str) -> tuple[float, bool]:
    return outcome(runs[target], "llm") if blind else (shipped(runs[target]), False)


def compare_models() -> str:
    """Each repo against the original on an L4: both models with and without the loop."""
    columns, targets = _comparison()
    lines = [
        "| Repo | " + " | ".join(header for header, _, _ in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    speedups, harmful = [[] for _ in columns], [0] * len(columns)
    for target in targets:
        cells = []
        for i, (_, runs, blind) in enumerate(columns):
            speedup, harm = _ships(runs, blind, target)
            speedups[i].append(speedup)
            harmful[i] += harm
            cells.append(f"{speedup:.2f}x" + (" harmful" if harm else ""))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    if targets:
        means = " | ".join(f"**{geometric_mean(s):.2f}x**" for s in speedups)
        lines.append(f"| **Geometric mean** | {means} |")
        lines.append("| Harmful changes shipped | " + " | ".join(map(str, harmful)) + " |")
    return "\n".join(lines)


def loop_gain() -> str:
    """Per model, the agent against its own one blind patch over the same repos. The
    ratio of the geometric means is also the geometric mean of the per-repo ratios."""
    columns, targets = _comparison()
    lines = [
        "| Model | Agent | One patch | Agent vs one patch | Harmful changes (agent, one patch) |",
        "|---|---|---|---|---|",
    ]
    for model, agent, patch in (("gpt-6.1-sol", *columns[0:2]), ("gpt-6-luna", *columns[2:4])):
        a = [_ships(agent[1], agent[2], t) for t in targets]
        p = [_ships(patch[1], patch[2], t) for t in targets]
        ga, gp = geometric_mean(x for x, _ in a), geometric_mean(x for x, _ in p)
        harm = f"{sum(h for _, h in a)}, {sum(h for _, h in p)}"
        lines.append(f"| {model} | {ga:.2f}x | {gp:.2f}x | {ga / gp:.2f}x | {harm} |")
    return "\n".join(lines)


def headline() -> str:
    """One row per real repo: the original code against everything the gpt-6.1-sol agent
    kept, from the final pair of its run. results/fixes.json labels each repo's changes."""
    runs = sol_agent_runs()
    fixes = json.loads((RESULTS / "fixes.json").read_text())
    known = stars()
    lines = ["| Repo | Stars | Step time (ms) | Speedup | Changes kept |", "|---|---|---|---|---|"]
    names = sorted((t for t in runs if not t.startswith("bench-")), key=lambda t: -shipped(runs[t]))
    for target in names:
        run, url = runs[target], targets.load(target).repo
        repo = url.removeprefix("https://github.com/")
        before, after = _final_pair_ms(run)
        kept = shipped(run) > 1.0
        cells = [
            f"[{repo}]({url})",
            f"{known[repo]:,}" if repo in known else "",
            f"{before:.1f} → {after:.1f}" if kept else f"{before:.1f}",
            f"{shipped(run):.2f}x" if kept else "nothing kept",
            fixes[target] if kept else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _final_pair_ms(run: dict) -> tuple[float, float | None]:
    """Mean step time of the original and the final code in a run's last pair. A run that
    kept nothing has only its baseline."""
    if "final_pair" not in run:
        return mean(r["step_ms"] for r in run["baseline"]), mean(
            r["step_ms"] for r in run["candidate"]
        )
    pair = run["final_pair"]
    if not pair or not pair.get("candidate_ms"):
        return run["baseline_ms"], None
    return mean(pair["baseline_ms"]), mean(pair["candidate_ms"])


def models() -> str:
    """The agent with each model: the speedup it kept per target, the harmful attempts
    its keep rule caught."""
    runs = {key: latest("agent", model.name) for key, model in MODELS.items()}
    runs = {key: r for key, r in runs.items() if r}
    targets = sorted({t for r in runs.values() for t in r})
    lines = ["| Target | " + " | ".join(runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for target in targets:
        cells = [f"{r[target]['speedup']:.2f}x" if target in r else "" for r in runs.values()]
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    caught = [
        sum(harmful_attempt(a) for x in r.values() for a in x["attempts"]) for r in runs.values()
    ]
    lines.append("| Harmful attempts caught | " + " | ".join(map(str, caught)) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    for title, model in (("gpt-6-luna", MAIN_MODEL), ("the local model", LOCAL_MODEL)):
        print(f"Setups with {title}\n")
        print(table(model))
        print()
        print(totals(model))
        print()
    print(models())
    print()
    print(headline())
    print()
    print(compare_models())
    print()
    print(loop_gain())
