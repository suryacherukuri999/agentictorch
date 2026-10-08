"""Measurement runs on Modal.

Only the measurement runs happen on Modal; the LLM calls, code edits and the
keep/revert logic stay on the Mac. Each target gets its own image (a shared
torch base plus the repo's dependencies), datasets live in a Volume, and the
current working copy of the repo is uploaded with every run. Every call gets a
fresh container; `ab` runs a baseline and a candidate in the same container so
they can be compared on the same hardware.
"""

import contextlib
import io
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

import modal

from agentictorch import costs
from agentictorch.harness import runner
from agentictorch.harness.runner import Job
from agentictorch.targets import Target

# These runs need a few GB of GPU memory; bigger GPUs only spend the credits faster.
GPUS = ("T4", "L4", "A100-40GB")
# An A100-40GB request has run on 40GB, 80GB and PCIe cards, billed as A100-40GB.
# A pair split over two containers could land on two GPU models, so it runs in one.
MIXED_POOLS = {"A100-40GB"}
CPU_CORES = 8.0
MEMORY_GIB = 16
# Image pull, container start and data staging, on top of the run's own timeout.
STARTUP_ALLOWANCE = 600
# Billed time outside our own timer in each container (image load, Python start), for
# costing. Measured on 2026-10-25: Modal billed about 10 s per container beyond it,
# over 503 L4 containers.
BILLED_STARTUP = 10
MAX_UPLOAD = 200 * 2**20

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "numpy")
)
data_volume = modal.Volume.from_name("agentictorch-data", create_if_missing=True)


def image_for(target: Target) -> modal.Image:
    image = base_image
    if target.apt:
        image = image.apt_install(*target.apt)
    if target.pip:
        image = image.uv_pip_install(*target.pip)
    return image.add_local_python_source("agentictorch")


def measure(
    target: Target,
    repo_dir: Path,
    *,
    gpu: str | None = "L4",
    extra_args: str = "",
    seeds: tuple[int, ...] = (0, 1),
    steps: int | None = None,
    warmup: int | None = None,
    profile: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Run one working copy once per seed, each in its own container."""
    job = Job(repo_dir, extra_args)
    results = _run(
        target,
        [[job] for _ in seeds],
        seeds,
        gpu=gpu,
        steps=steps,
        warmup=warmup,
        profile=profile,
        verbose=verbose,
    )
    return [runs[0] for runs in results]


def ab(
    target: Target,
    base: Job,
    cand: Job,
    *,
    gpu: str | None = "L4",
    seeds: tuple[int, ...] = (0, 1),
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run baseline and candidate in the same container, one container per seed.

    The order alternates between seeds, so neither configuration always runs
    second on a GPU the other has just warmed up. On a GPU type whose requests
    Modal serves with more than one model, every run goes in one container.
    """
    orders = [[base, cand] if i % 2 == 0 else [cand, base] for i in range(len(seeds))]
    if gpu in MIXED_POOLS:
        jobs = [
            replace(job, seed=seed)
            for order, seed in zip(orders, seeds, strict=True)
            for job in order
        ]
        [runs] = _run(target, [jobs], seeds[:1], gpu=gpu, verbose=verbose)
        results = [runs[i : i + 2] for i in range(0, len(runs), 2)]
    else:
        results = _run(target, orders, seeds, gpu=gpu, verbose=verbose)
    base_runs, cand_runs = [], []
    for i, (first, second) in enumerate(results):
        base_runs.append(first if i % 2 == 0 else second)
        cand_runs.append(second if i % 2 == 0 else first)
    return base_runs, cand_runs


def _run(
    target: Target,
    job_lists: list[list[Job]],
    seeds: tuple[int, ...],
    *,
    gpu: str | None,
    steps: int | None = None,
    warmup: int | None = None,
    profile: bool = False,
    verbose: bool = False,
) -> list[list[dict]]:
    """Run each list of jobs in its own container, in order, and return their results."""
    if gpu is not None and gpu not in GPUS:
        raise ValueError(f"GPU must be one of {GPUS}, got {gpu}")
    timeout = max(map(len, job_lists)) * target.timeout + STARTUP_ALLOWANCE
    rate = costs.hourly_rate(gpu, CPU_CORES, MEMORY_GIB)
    costs.check(planned=len(job_lists) * rate * timeout / 3600)

    payloads = {}
    specs = []
    for jobs in job_lists:
        for job in jobs:
            if job.repo_dir not in payloads:
                payloads[job.repo_dir] = pack(job.repo_dir)
        specs.append(
            [
                {
                    "payload": payloads[job.repo_dir],
                    "command": f"{target.command} {job.extra_args}".strip(),
                    "seed": job.seed,
                }
                for job in jobs
            ]
        )

    app = modal.App(f"agentictorch-{target.name}")
    function = app.function(
        image=image_for(target),
        gpu=gpu,
        cpu=CPU_CORES,
        memory=MEMORY_GIB * 1024,
        volumes={"/data": data_volume},
        timeout=timeout,
        single_use_containers=True,
        serialized=True,
    )(_measure_in_container)

    start = time.monotonic()
    results = None
    try:
        with _output(verbose), _running(app):
            calls = [
                function.spawn(
                    spec,
                    steps=steps or target.steps,
                    warmup=warmup or target.warmup,
                    seed=seed,
                    timeout=target.timeout,
                    env=target.env,
                    dataset=target.dataset,
                    data_path=target.data_path,
                    profile=profile,
                )
                for spec, seed in zip(specs, seeds, strict=True)
            ]
            results = [call.get() for call in calls]
            return results
    finally:
        if results:
            # Each container timed itself; add what Modal bills before our code runs.
            seconds = sum(runs[-1]["container_s"] + BILLED_STARTUP for runs in results)
        else:
            # Failed or interrupted: count every container for the whole wall time.
            seconds = len(job_lists) * (time.monotonic() - start)
        costs.record(
            target=target.name,
            gpu=gpu,
            containers=len(job_lists),
            seconds=round(seconds, 1),
            cost=round(rate * seconds / 3600, 4),
        )


def download(target: Target, *, verbose: bool = True) -> str:
    """Fill the target's dataset folder in the Volume, once. Runs without a GPU."""
    if not (target.dataset and target.download):
        raise ValueError(f"{target.name} has no dataset to download")
    cpu, memory_gib, timeout = 2.0, 8, 3600
    rate = costs.hourly_rate(None, cpu, memory_gib)
    costs.check(planned=rate * timeout / 3600)

    app = modal.App(f"agentictorch-{target.name}-data")
    function = app.function(
        image=image_for(target),
        cpu=cpu,
        memory=memory_gib * 1024,
        volumes={"/data": data_volume},
        timeout=timeout,
        serialized=True,
    )(_download_in_container)
    start = time.monotonic()
    try:
        with _output(verbose), _running(app):
            return function.remote(target.download, target.dataset)
    finally:
        seconds = time.monotonic() - start
        costs.record(
            target=target.name,
            gpu=None,
            containers=1,
            seconds=round(seconds, 1),
            cost=round(rate * seconds / 3600, 4),
        )


def pack(repo_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(repo_dir, arcname=".", filter=_skip_git)
    data = buffer.getvalue()
    if len(data) > MAX_UPLOAD:
        raise ValueError(f"{repo_dir} packs to {len(data) / 2**20:.0f} MiB, too big to upload")
    return data


def _skip_git(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if ".git" in Path(info.name).parts else info


@contextlib.contextmanager
def _running(app: modal.App):
    """app.run(), retried when Modal rate-limits app creation (several runs at once)."""
    for attempt in range(4):
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(app.run())
        except modal.exception.ResourceExhaustedError:
            if attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))
            continue
        with stack:
            yield
        return


def _output(verbose: bool):
    return modal.enable_output() if verbose else contextlib.nullcontext()


def _measure_in_container(
    jobs: list[dict],
    *,
    steps: int,
    warmup: int,
    seed: int,
    timeout: int,
    env: dict[str, str],
    dataset: str | None,
    data_path: str | None,
    profile: bool,
) -> list[dict]:
    started = time.monotonic()
    container = os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
    results = []
    for i, job in enumerate(jobs):
        repo = Path(f"/work/repo{i}")
        repo.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(job["payload"])) as tar:
            tar.extractall(repo, filter="data")
        if dataset:
            # Copy to local disk: the Volume is network storage, and reading it
            # during training would make data loading slower and noisier.
            shutil.copytree(Path("/data") / dataset, repo / data_path, dirs_exist_ok=True)
        result = runner.run(
            job["command"],
            repo,
            steps=steps,
            warmup=warmup,
            # Containers load this file from the working tree, so a caller started
            # before jobs carried seeds may send none.
            seed=seed if job.get("seed") is None else job["seed"],
            timeout=timeout,
            env=env,
            profile=profile,
        )
        # container_s is the time so far, so the last job's value covers the whole container.
        results.append(
            {**result, "container": container, "container_s": time.monotonic() - started}
        )
    return results


def _download_in_container(command: str, dataset: str) -> str:
    dest = Path("/data") / dataset
    if not dest.exists():
        # Download beside the final folder and rename on success, so a failed
        # download never leaves a half-filled folder that later runs would trust.
        partial = Path("/data") / f".{dataset}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        subprocess.run(command, shell=True, cwd=partial, check=True)
        partial.rename(dest)
        data_volume.commit()
    files = [p for p in dest.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.0f} MiB"
