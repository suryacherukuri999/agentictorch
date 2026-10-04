"""Per-repo survey rows and the summary table."""

from datetime import date

from agentictorch.survey.detect import Findings

FIELDS = [
    "repo",
    "url",
    "stars",
    "pushed_at",
    "commit",
    "training_code",
    "custom_loop",
    "per_step_syncs",
    "no_mixed_precision",
    "main_thread_loaders",
    "uses_torch_compile",
    "trainer_framework",
    "description",
]


def row(repo: dict, commit: str, found: Findings) -> dict:
    return {
        "repo": repo["repo"],
        "url": repo["url"],
        "stars": repo["stars"],
        "pushed_at": repo["pushed_at"],
        "commit": commit,
        "training_code": found.backward_calls > 0 or found.trainer_framework,
        "custom_loop": found.training_loops > 0,
        "per_step_syncs": len(found.loop_syncs),
        "no_mixed_precision": not found.amp,
        "main_thread_loaders": "; ".join(found.main_thread_loaders),
        "uses_torch_compile": found.uses_compile,
        "trainer_framework": found.trainer_framework,
        "description": repo["description"],
    }


def big_two(r: dict) -> bool:
    """No mixed precision anywhere, or a training DataLoader on the main thread."""
    return r["no_mixed_precision"] or bool(r["main_thread_loaders"])


def summary(rows: list[dict], scanned_on: date) -> str:
    training = [r for r in rows if r["training_code"]]
    custom = [r for r in training if r["custom_loop"]]
    top30 = sorted(training, key=lambda r: -r["stars"])[:30]
    active = [r for r in training if r["pushed_at"] >= "2025-01-01"]

    def share(subset, test):
        hits = sum(1 for r in subset if test(r))
        return f"{hits / len(subset):.0%} ({hits}/{len(subset)})" if subset else "n/a"

    stars = [r["stars"] for r in training]
    lines = [
        f"Static scan of {len(rows)} PyTorch repos on {scanned_on.isoformat()} "
        f"({min(stars)}-{max(stars)} stars); {len(training)} had training code.",
        "",
        "| Problem | Share of repos with training code |",
        "|---|---|",
        f"| No mixed precision anywhere | {share(training, lambda r: r['no_mixed_precision'])} |",
        "| Training DataLoader on the main thread | "
        f"{share(training, lambda r: bool(r['main_thread_loaders']))} |",
        f"| At least one of those two | {share(training, big_two)} |",
        "| Custom training loop that syncs CPU and GPU every step "
        f"(share of custom loops) | {share(custom, lambda r: r['per_step_syncs'] > 0)} |",
        f"| Never uses torch.compile | {share(training, lambda r: not r['uses_torch_compile'])} |",
        "",
        f"At least one of the big two: {share(top30, big_two)} of the 30 most-starred, "
        f"{share(active, big_two)} of repos pushed since 2025.",
    ]
    return "\n".join(lines) + "\n"
