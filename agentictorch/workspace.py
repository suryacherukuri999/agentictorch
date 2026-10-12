"""Local copies of target repos and their run results, kept in ~/agentictorch-work.

The pristine clone is never edited. Each experiment gets its own working copy,
with the target's compatibility patch committed on top so that `git diff`
shows only performance changes.
"""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from agentictorch.targets import Target

WORK = Path.home() / "agentictorch-work"


def clone(target: Target) -> Path:
    dest = WORK / "repos" / target.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(None, "clone", "--quiet", "--filter=blob:none", "--no-checkout", target.repo, dest)
        _git(dest, "checkout", "--quiet", target.commit)
    return dest


def working_copy(target: Target, label: str) -> Path:
    """Return the working copy for `label`, creating it from the pinned commit if needed."""
    dest = WORK / "work" / target.name / label
    if dest.exists():
        return dest
    shutil.copytree(clone(target), dest, symlinks=True)
    if target.compat_patch:
        _git(dest, "apply", target.compat_patch)
        _git(dest, "add", "--all")
        _git(dest, "-c", "user.name=agentictorch", "-c", "user.email=agentictorch@localhost",
             "commit", "--quiet", "-m", "compatibility tweaks")  # fmt: skip
    return dest


def diff(repo: Path, target: Target) -> str:
    """Performance changes in a working copy, committed or not, since the commit it
    started from. The agent commits what it keeps, so a plain `git diff` misses it."""
    return _git(repo, "diff", start(repo, target), capture=True)


def start(repo: Path, target: Target) -> str:
    """The commit a working copy started from: the pinned commit, or the compatibility
    tweaks committed on top of it. A planted test script starts from its root commit."""
    if not target.commit:
        return _git(repo, "rev-list", "--max-parents=0", "HEAD", capture=True).split()[0]
    if not target.compat_patch:
        return target.commit
    return _git(repo, "rev-list", "--reverse", f"{target.commit}..HEAD", capture=True).split()[0]


def save_runs(target: Target, label: str, runs: list[dict]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = WORK / "runs" / target.name / f"{label}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target": target.name, "label": label, "runs": runs}, indent=1))
    return path


def save_ab(
    target: Target,
    base: str,
    cand: str,
    base_runs: list[dict],
    cand_runs: list[dict],
    base_args: str,
    cand_args: str,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = WORK / "runs" / target.name / f"ab-{cand}-vs-{base}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "target": target.name,
        "baseline": {"label": base, "extra_args": base_args, "runs": base_runs},
        "candidate": {"label": cand, "extra_args": cand_args, "runs": cand_runs},
    }
    path.write_text(json.dumps(record, indent=1))
    return path


def load_ab(target: Target, base: str, cand: str) -> dict:
    """The most recent paired runs of `cand` against `base`."""
    paths = sorted((WORK / "runs" / target.name).glob(f"ab-{cand}-vs-{base}-*.json"))
    if not paths:
        raise FileNotFoundError(f"no paired runs saved for {target.name} {cand} vs {base}")
    return json.loads(paths[-1].read_text())


def _git(repo: Path | None, *args, capture: bool = False) -> str:
    command = ["git", *(["-C", str(repo)] if repo else []), *map(str, args)]
    out = subprocess.run(command, check=True, capture_output=capture, text=True)
    return out.stdout if capture else ""
