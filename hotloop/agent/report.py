"""The report of one agent run: what was wrong, each change and its measured effect."""

import json
from pathlib import Path

from hotloop.agent.loop import Run
from hotloop.measurement import Measurement


def _speed(m: Measurement) -> str:
    return f"{m.mean_step_ms:.1f} ms/step" if m.completed else f"failed ({m.failure()})"


def _baseline(run: Run) -> Measurement | None:
    """The first measurement of the original code: the session's own, or, for setups
    that never measured before editing, the baseline side of their one pair."""
    if run.session.baseline is not None:
        return run.session.baseline
    return run.final.baseline if run.final else None


def markdown(run: Run) -> str:
    s = run.session
    base = _baseline(run)
    lines = [f"# Hotloop report: {s.target.name}", ""]
    if run.final and run.final.verdict.keep:
        lines.append(f"**Result: {run.final.verdict.reason}** on an {run.final.baseline.gpu}.")
    else:
        lines.append("**Result: no change was kept.**")
    lines += [
        "",
        f"- Command: `{s.target.command}`",
        f"- Baseline: {_speed(base)} over {len(base.runs)} runs" if base else "- Baseline: none",
        f"- Model: {run.model}, {run.turns} turns, "
        f"{run.tokens['input']} input and {run.tokens['output']} output tokens, "
        f"${run.cost:.4f}",
        "",
    ]
    if s.profiles:
        lines += ["## Diagnosis", "", "```", s.profiles[0], "```", ""]
    if run.summary:
        lines += [run.summary, ""]
    lines += [
        "## Changes",
        "",
        "| # | Change | Before | After | Outcome |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(s.attempts, 1):
        outcome = ("kept: " if a.verdict.keep else "reverted: ") + a.verdict.reason
        lines.append(
            f"| {i} | {a.description} | {_speed(a.baseline)} | {_speed(a.candidate)} | {outcome} |"
        )
    if run.final:
        lines += ["", "## Diff of the kept changes", "", "```diff", run.final.diff.strip(), "```"]
    if run.sources:
        lines += ["", "## Sources the agent read", ""]
        lines += [f"- [{title}]({url})" for url, title in run.sources.items()]
    return "\n".join(lines) + "\n"


def summary(run: Run) -> dict:
    s = run.session
    final = run.final
    base = _baseline(run)
    return {
        "target": s.target.name,
        "model": run.model,
        "gpu": base.gpu if base else None,
        "baseline_ms": base.mean_step_ms if base and base.completed else None,
        # The measured effect of the final change, whether or not the keep rule kept it:
        # setups that patch blind would apply it either way.
        "measured_speedup": final.verdict.speedup if final else None,
        "speedup": final.verdict.speedup if final and final.verdict.keep else 1.0,
        "kept": [a.description for a in s.kept],
        "attempts": [
            {
                "change": a.description,
                "keep": a.verdict.keep,
                "reason": a.verdict.reason,
                "speedup": a.verdict.speedup,
            }
            for a in s.attempts
        ],
        # Per-run numbers of the final pair (original against everything kept), so
        # the total can be checked without the raw runs.
        "final_pair": {
            "baseline_ms": final.baseline.step_ms if final.baseline.completed else None,
            "candidate_ms": final.candidate.step_ms if final.candidate.completed else None,
            "baseline_loss": final.baseline.loss,
            "candidate_loss": final.candidate.loss,
        }
        if final
        else None,
        "llm_cost": run.cost,
        "turns": run.turns,
        "tokens": run.tokens,
    }


def write(run: Run, folder: Path, label: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.json").write_text(json.dumps(summary(run), indent=1) + "\n")
    path = folder / f"{label}.md"
    path.write_text(markdown(run))
    return path
