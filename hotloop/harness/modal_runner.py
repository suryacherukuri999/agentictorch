"""Measurement runs on Modal.

Only the measurement runs happen on Modal; the LLM calls, code edits and the
keep/revert logic stay on the Mac. Each target gets its own image (a shared
torch base plus the repo's dependencies), datasets live in a Volume, and the
current working copy of the repo is uploaded with every run. Every run gets a
fresh container, so the two seeds of a configuration also measure the noise
between containers.
"""

import contextlib
import io
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import modal

from hotloop import costs
from hotloop.harness import runner
from hotloop.targets import Target

# These runs need a few GB of GPU memory; bigger GPUs only spend the credits faster.
GPUS = ("T4", "L4", "A100-40GB")
CPU_CORES = 8.0
MEMORY_GIB = 16
# Image pull, container start and data staging, on top of the run's own timeout.
STARTUP_ALLOWANCE = 600
MAX_UPLOAD = 200 * 2**20

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "numpy")
)
data_volume = modal.Volume.from_name("hotloop-data", create_if_missing=True)


def image_for(target: Target) -> modal.Image:
    image = base_image
    if target.apt:
        image = image.apt_install(*target.apt)
    if target.pip:
        image = image.uv_pip_install(*target.pip)
    return image.add_local_python_source("hotloop")


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
    """Run the working copy at `repo_dir` once per seed, each in its own container.

    `extra_args` is appended to the target's command, to switch on opt-in changes.
    """
    if gpu is not None and gpu not in GPUS:
        raise ValueError(f"GPU must be one of {GPUS}, got {gpu}")
    rate = costs.hourly_rate(gpu, CPU_CORES, MEMORY_GIB)
    costs.check(planned=len(seeds) * rate * (target.timeout + STARTUP_ALLOWANCE) / 3600)
    payload = pack(repo_dir)

    app = modal.App(f"hotloop-{target.name}")
    function = app.function(
        image=image_for(target),
        gpu=gpu,
        cpu=CPU_CORES,
        memory=MEMORY_GIB * 1024,
        volumes={"/data": data_volume},
        timeout=target.timeout + STARTUP_ALLOWANCE,
        single_use_containers=True,
        serialized=True,
    )(_measure_in_container)

    start = time.monotonic()
    try:
        with _output(verbose), app.run():
            calls = [
                function.spawn(
                    payload,
                    f"{target.command} {extra_args}".strip(),
                    steps=steps or target.steps,
                    warmup=warmup or target.warmup,
                    seed=seed,
                    timeout=target.timeout,
                    env=target.env,
                    dataset=target.dataset,
                    data_path=target.data_path,
                    profile=profile,
                )
                for seed in seeds
            ]
            return [call.get() for call in calls]
    finally:
        # Wall time includes queueing, which Modal doesn't bill, so this errs high.
        seconds = time.monotonic() - start
        costs.record(
            target=target.name,
            gpu=gpu,
            containers=len(seeds),
            seconds=round(seconds, 1),
            cost=round(len(seeds) * rate * seconds / 3600, 4),
        )


def download(target: Target, *, verbose: bool = True) -> str:
    """Fill the target's dataset folder in the Volume, once. Runs without a GPU."""
    if not (target.dataset and target.download):
        raise ValueError(f"{target.name} has no dataset to download")
    cpu, memory_gib, timeout = 2.0, 8, 3600
    rate = costs.hourly_rate(None, cpu, memory_gib)
    costs.check(planned=rate * timeout / 3600)

    app = modal.App(f"hotloop-{target.name}-data")
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
        with _output(verbose), app.run():
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


def _output(verbose: bool):
    return modal.enable_output() if verbose else contextlib.nullcontext()


def _measure_in_container(
    payload: bytes,
    command: str,
    *,
    steps: int,
    warmup: int,
    seed: int,
    timeout: int,
    env: dict[str, str],
    dataset: str | None,
    data_path: str | None,
    profile: bool,
) -> dict:
    repo = Path("/work/repo")
    repo.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        tar.extractall(repo, filter="data")
    if dataset:
        # Copy to local disk: the Volume is network storage, and reading it
        # during training would make data loading slower and noisier.
        shutil.copytree(Path("/data") / dataset, repo / data_path, dirs_exist_ok=True)
    return runner.run(
        command,
        repo,
        steps=steps,
        warmup=warmup,
        seed=seed,
        timeout=timeout,
        env=env,
        profile=profile,
    )


def _download_in_container(command: str, dataset: str) -> str:
    dest = Path("/data") / dataset
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.iterdir()):
        subprocess.run(command, shell=True, cwd=dest, check=True)
        data_volume.commit()
    files = [p for p in dest.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.0f} MiB"
