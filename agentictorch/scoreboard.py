"""Compare the setups over everything saved in results/eval, and build the README's
results table from the headline runs in results/headline.

python -m agentictorch.scoreboard

For planted cases, "recovered" is the share of the slowdown the setup won back:
with t the step time, (t_planted - t_after) / (t_planted - t_clean), from the
measured cost of each planted mistake in results/bench.
"""

import csv
import json
from pathlib import Path
from statistics import geometric_mean, mean

from agentictorch import targets
from agentictorch.agent.models import MODELS

RESULTS = Path(__file__).parent.parent / "results"
SETUPS = ["rules", "llm", "llm-kb", "agent"]
# The setups are compared, and the headline runs made, with gpt-6-luna. The planted suite
# ran with the local model, which turned out too slow for the real repos.
MAIN_MODEL = MODELS["gpt-6-luna"].name
LOCAL_MODEL = MODELS["ornith-9b"].name
# A blind patch that runs this much slower than the original counts as harmful.
HARM_MARGIN = 0.97
# Who made the reference fixes in results/hand: the model editing each repo itself, with
# the profiler and paired benchmark but outside the agent loop.
DIRECT = "Reference model, direct"


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
    # Only planted cases have a known cost; real repos leave the column out.
    planted = any(planted_cost(t) for t in targets)
    columns = ["Target", *(["Planted cost"] if planted else []), *SETUPS]
    lines = ["| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
    for target in targets:
        cost = planted_cost(target)
        cells = [f"{cost:.2f}x" if cost else ""] if planted else []
        for setup in SETUPS:
            record = results[setup].get(target)
            if record is None:
                cells.append("")
                continue
            speedup, harmful = outcome(record, setup)
            share = recovered(speedup, cost) if cost else None
            cell = f"{speedup:.2f}x" + (f" ({share:.0%})" if share is not None else "")
            cells.append(cell + (" harmful" if harmful else ""))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def totals(model: str = MAIN_MODEL) -> str:
    """Per setup, over the targets every setup ran: the typical speedup that ships (a patch
    that crashes counts as 1.00x), harmful changes shipped and, for the agent, the harmful
    attempts its keep rule caught and reverted."""
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
    """The maintainer check per owner/repo: stars and recent outside merges."""
    record = json.loads((RESULTS / "survey" / "maintainers.json").read_text())
    return {r["repo"]: r for r in record["repos"]}


def stars() -> dict[str, int]:
    """GitHub stars per owner/repo, as of the survey or the maintainer check."""
    with (RESULTS / "survey" / "repos.csv").open() as f:
        found = {row["repo"]: int(row["stars"]) for row in csv.DictReader(f)}
    found.update({repo: r["stars"] for repo, r in maintainers().items()})
    return found


def headline_records() -> dict[str, dict]:
    """The latest paired run per repo in results/headline."""
    records = {}
    for path in (RESULTS / "headline").glob("*.json"):
        record = json.loads(path.read_text())
        previous = records.get(record["target"])
        if previous is None or record["recorded"] > previous["recorded"]:
            records[record["target"]] = record
    return records


def hand_fixes() -> dict[str, dict]:
    """The fastest measured hand fix per repo, from results/hand and the go/no-go check."""
    best = {}
    for path in [*(RESULTS / "hand").glob("*.json"), *(RESULTS / "go-no-go").glob("*.json")]:
        record = json.loads(path.read_text())
        previous = best.get(record["target"])
        if previous is None or (record["speedup"] or 0) > (previous["speedup"] or 0):
            best[record["target"]] = record
    return best


def versus() -> str:
    """Per real repo: the agent's kept changes and the fastest hand fix, each measured
    in pairs against the original code on the same kind of GPU."""
    hands = hand_fixes()
    agents = latest("agent", MAIN_MODEL)
    labels = {t: r["change"] for t, r in headline_records().items()}
    lines = [
        f"| Repo | GPU | Original (ms/step) | Agent | Agent's changes | {DIRECT} | Its changes |",
        "|---" * 7 + "|",
    ]
    for target in sorted(set(agents) & set(hands), key=lambda t: -hands[t]["speedup"]):
        agent, hand = agents[target], hands[target]
        if agent["gpu"] != hand["gpu"]:
            continue
        changes = labels.get(target, f"{len(agent['kept'])} changes") if agent["kept"] else ""
        cells = [
            target,
            hand["gpu"].removeprefix("NVIDIA "),
            f"{mean(run['step_ms'] for run in hand['baseline']):.1f}",
            f"{agent['speedup']:.2f}x" if agent["kept"] else "nothing kept",
            changes,
            f"{hand['speedup']:.2f}x",
            hand["change"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shipped(record: dict) -> float:
    """What a run ships: its speedup if the keep rule kept it, else the original's 1.00x."""
    kept = record["keep"] if "keep" in record else bool(record["kept"])
    return record["speedup"] if kept else 1.0


def compare_models() -> str:
    """Per repo, each against the original on an L4: each model as the agent and as one
    blind patch (the same model with and without the loop), and the direct fix as a
    reference; then the geometric means and harmful changes shipped over the repos all
    of them ran. A stronger-model agent run whose report was lost keeps its later final
    pair."""
    sol_model = MODELS["gpt-6.1-sol"].name
    sol = latest("agent", sol_model)
    for path in (RESULTS / "final-pairs").glob("*.json"):
        record = json.loads(path.read_text())
        sol.setdefault(record["target"], record)
    # A blind patch ships whatever it does; the others keep measured wins only.
    columns = [
        ("Agent, gpt-6.1-sol", sol, False),
        ("One patch, gpt-6.1-sol", latest("llm", sol_model), True),
        ("Agent, gpt-6-luna", latest("agent", MAIN_MODEL), False),
        ("One patch, gpt-6-luna", latest("llm", MAIN_MODEL), True),
        (DIRECT, hand_fixes(), False),
    ]
    common = set.intersection(*(set(runs) for _, runs, _ in columns))
    targets = sorted(t for t in common if not t.startswith("bench-"))
    lines = [
        "| Repo | " + " | ".join(header for header, _, _ in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    speedups, harmful = [[] for _ in columns], [0] * len(columns)
    for target in targets:
        cells = []
        for i, (_, runs, blind) in enumerate(columns):
            speedup, harm = (
                outcome(runs[target], "llm") if blind else (shipped(runs[target]), False)
            )
            speedups[i].append(speedup)
            harmful[i] += harm
            cells.append(f"{speedup:.2f}x" + (" harmful" if harm else ""))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    if targets:
        means = " | ".join(f"**{geometric_mean(s):.2f}x**" for s in speedups)
        lines.append(f"| **Geometric mean** | {means} |")
        lines.append("| Harmful changes shipped | " + " | ".join(map(str, harmful)) + " |")
    return "\n".join(lines)


def headline() -> str:
    """One row per real repo: the original code against the agent's kept changes,
    from the latest paired run saved for it in results/headline."""
    records = headline_records()
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
    # Repos where the agent kept nothing have no headline run, but they belong in the table.
    for target, run in sorted(latest("agent", MAIN_MODEL).items()):
        if target in records or target.startswith("bench-") or run["kept"]:
            continue
        url = targets.load(target).repo
        repo = url.removeprefix("https://github.com/")
        cells = [
            f"[{repo}]({url})",
            f"{known[repo]:,}" if repo in known else "",
            run["gpu"].removeprefix("NVIDIA "),
            f"{run['baseline_ms']:.1f}",
            "",
            "nothing kept",
            "",
            "",
            prs.get(target) or _not_sent(checked.get(repo)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def models() -> str:
    """The agent with each model: the speedup it kept per target, the harmful attempts
    its keep rule caught, and what the model cost."""
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
    cost = [sum(x["llm_cost"] for x in r.values()) for r in runs.values()]
    lines.append("| Harmful attempts caught | " + " | ".join(map(str, caught)) + " |")
    lines.append("| LLM cost | " + " | ".join(f"${c:.2f}" for c in cost) + " |")
    return "\n".join(lines)


def _not_sent(check: dict | None) -> str:
    if check and check["outside_merges_12_months"] == 0:
        return "not sent: no outside pull request merged in 12 months"
    return "not sent"


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
    print(versus())
    print()
    print(compare_models())
