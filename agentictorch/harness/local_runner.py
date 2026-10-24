"""Measurement runs on this machine's own GPU, with the same harness and pairing as Modal.

Each target gets its own uv environment (the Modal image's torch plus its packages),
datasets live in a local folder, and every run starts from a fresh copy of the code.
System packages aren't installed. Repo code runs directly on this machine, so keep it
to code you trust."""

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from agentictorch.harness import runner
from agentictorch.harness.runner import Job
from agentictorch.targets import Target

WORK = Path.home() / "agentictorch-work"
ENVS = WORK / "envs"
DATA = WORK / "local-data"
# The Modal image's versions, so both backends run the same libraries.
BASE_PACKAGES = ("torch==2.14.1", "torchvision==0.29.1", "numpy")


def measure(
    target: Target,
    repo_dir: Path,
    *,
    gpu: str | None = None,
    extra_args: str = "",
    seeds: tuple[int, ...] = (0, 1),
    steps: int | None = None,
    warmup: int | None = None,
    profile: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Run one working copy once per seed, on whatever GPU this machine has."""
    job = Job(repo_dir, extra_args)
    return [_run(target, job, seed, steps=steps, warmup=warmup, profile=profile) for seed in seeds]


def ab(
    target: Target,
    base: Job,
    cand: Job,
    *,
    gpu: str | None = None,
    seeds: tuple[int, ...] = (0, 1),
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run baseline and candidate back to back, once per seed, alternating which goes
    first, so neither always runs on a GPU the other has just warmed up."""
    base_runs, cand_runs = [], []
    for i, seed in enumerate(seeds):
        first, second = (base, cand) if i % 2 == 0 else (cand, base)
        runs = [_run(target, first, seed), _run(target, second, seed)]
        if i % 2:
            runs.reverse()
        # The keep rule trusts a comparison only between runs on the same hardware.
        for run in runs:
            run["container"] = f"{platform.node()}-pair{i}"
        base_runs.append(runs[0])
        cand_runs.append(runs[1])
    return base_runs, cand_runs


def download(target: Target, *, verbose: bool = True) -> str:
    """Fill the target's dataset folder under ~/agentictorch-work/local-data, once."""
    if not (target.dataset and target.download):
        raise ValueError(f"{target.name} has no dataset to download")
    dest = DATA / target.dataset
    if not dest.exists():
        # Download beside the final folder and rename on success, so a failed download
        # never leaves a half-filled folder that later runs would trust.
        partial = DATA / f".{target.dataset}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        env = {**os.environ, **_python_first(target)}
        subprocess.run(target.download, shell=True, cwd=partial, check=True, env=env)
        partial.rename(dest)
    files = [p for p in dest.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.0f} MiB"


def environment(target: Target) -> Path:
    """The bin folder of the target's own Python environment, made on first use."""
    python = ENVS / target.name / "bin" / "python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--quiet", "--python", "3.12", str(python.parent.parent)],
                       check=True)  # fmt: skip
        subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(python),
                        *BASE_PACKAGES, *target.pip], check=True)  # fmt: skip
    return python.parent


def _python_first(target: Target) -> dict[str, str]:
    return {"PATH": f"{environment(target)}{os.pathsep}{os.environ['PATH']}"}


def _run(
    target: Target,
    job: Job,
    seed: int,
    *,
    steps: int | None = None,
    warmup: int | None = None,
    profile: bool = False,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="agentictorch-local-") as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(job.repo_dir, repo, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        if target.dataset:
            shutil.copytree(DATA / target.dataset, repo / target.data_path, dirs_exist_ok=True)
        result = runner.run(
            f"{target.command} {job.extra_args}".strip(),
            repo,
            steps=steps or target.steps,
            warmup=warmup or target.warmup,
            seed=seed,
            timeout=target.timeout,
            env={**target.env, **_python_first(target)},
            profile=profile,
        )
    return {**result, "container": platform.node()}
