"""Run a training command with the harness injected and collect its results.

This runs on the Mac (CPU checks of our own scripts) and inside Modal
containers, so it uses only the standard library.
"""

import contextlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from hotloop.profile import summary

INJECT_DIR = Path(__file__).parent / "inject"


def run(
    command: str,
    cwd: str | Path,
    *,
    steps: int,
    warmup: int,
    seed: int = 0,
    timeout: float = 1800,
    env: dict[str, str] | None = None,
    profile: bool = False,
) -> dict:
    """Run a shell command until `warmup` + `steps` optimizer steps and return its results.

    With `profile`, the measured steps run under the PyTorch profiler and the
    result carries a short summary of it under "profile".
    """
    if steps < 1 or warmup < 1:
        raise ValueError("steps and warmup must both be at least 1")

    with tempfile.TemporaryDirectory(prefix="hotloop-") as tmp:
        out = Path(tmp) / "result.json"
        log = Path(tmp) / "log.txt"
        profile_dir = Path(tmp) / "profile"
        run_env = {**os.environ, **(env or {})}
        run_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(INJECT_DIR), run_env.get("PYTHONPATH")) if p
        )
        run_env |= {
            "HOTLOOP_OUT": str(out),
            "HOTLOOP_STEPS": str(steps),
            "HOTLOOP_WARMUP": str(warmup),
            "HOTLOOP_SEED": str(seed),
        }
        if profile:
            run_env["HOTLOOP_PROFILE"] = str(profile_dir)

        start = time.monotonic()
        with log.open("wb") as log_file:
            # A new session lets us kill the whole process tree, including
            # DataLoader workers that outlive the main process.
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=run_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
            _kill_group(proc.pid)
            proc.wait()

        result = json.loads(out.read_text()) if out.exists() else {"completed": False}
        result |= {
            "exit_code": proc.returncode,
            "timed_out": timed_out,
            "wall_s": round(time.monotonic() - start, 2),
            "gpu": gpu_name(),
            "cpu": cpu_name(),
            "log_tail": tail(log.read_bytes().decode(errors="replace")),
        }
        if (profile_dir / "trace.json").exists():
            summarized = summary.load(profile_dir, steps)
            result["profile"] = asdict(summarized)
            result["profile_text"] = summary.render(summarized)
    return result


def gpu_name() -> str:
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    names = out.stdout.strip().splitlines()
    return names[0] if out.returncode == 0 and names else "unknown"


def cpu_name() -> str:
    # Containers can land on different CPUs, which matters for data loading.
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def tail(text: str, lines: int = 80) -> str:
    # Progress bars redraw with carriage returns; treat each redraw as a line.
    parts = [p for p in re.split(r"[\r\n]+", text) if p.strip()]
    return "\n".join(parts[-lines:])


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
