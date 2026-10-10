"""Turn a PyTorch profiler run into a short summary the agent can read.

The raw trace never goes into a prompt: a few steps of a real model produce
megabytes of events. Each top op and each host-device sync is named with the
repo line behind it, so the agent knows where to edit. This runs inside the
measurement container, so it uses only the standard library.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

GPU_EVENTS = {"kernel", "gpu_memcpy", "gpu_memset"}
# .item(), .cpu() and similar calls end in one of these, so counting them counts syncs.
SYNC_CALLS = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"}
TOP = 8
TOP_SYNC_SITES = 6
NO_LINE = "no repo line: library code"


@dataclass
class ProfileSummary:
    steps: int
    window_ms: float
    gpu_busy: float | None  # share of the window with a kernel or copy running
    data_wait: float  # share of the window spent waiting on the DataLoader
    syncs_per_step: float  # host waits for the GPU: .item(), .cpu(), synchronize()
    copies_to_host_per_step: float
    sync_sites: list[dict]  # {"where", "per_step", "code"}, most frequent first
    # {"name", "share", "where", "backward", "code"}: share of device time, or CPU time on CPU.
    # A backward op's "where" is the forward line that created it.
    top_ops: list[dict]


def summarize(
    trace: dict,
    ops: list[dict],
    steps: int,
    syncs: list[dict] | None = None,
    repo: Path | None = None,
) -> ProfileSummary:
    """`syncs` are the syncs caught by CUDA's sync debug mode, by repo line; without
    them (on CPU) the count comes from sync calls in the trace."""
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    timed = [e for e in events if e.get("cat") in {"cpu_op", "kernel"} | GPU_EVENTS]
    start = min(e["ts"] for e in timed)
    window = max(e["ts"] + e["dur"] for e in timed) - start

    gpu = [(e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") in GPU_EVENTS]
    gpu_busy = _covered(gpu) / window if gpu else None

    data = sum(
        e["dur"]
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("enumerate(DataLoader)")
    )
    names = Counter(e["name"] for e in events if e.get("cat") == "cuda_runtime")
    total_syncs = sum(s["count"] for s in syncs) if syncs else sum(names[n] for n in SYNC_CALLS)
    to_host = sum(1 for e in events if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"])
    code = _SourceLines(repo)
    sites = sorted(syncs or [], key=lambda s: -s["count"])[:TOP_SYNC_SITES]

    return ProfileSummary(
        steps=steps,
        window_ms=window / 1000,
        gpu_busy=gpu_busy,
        data_wait=data / window,
        syncs_per_step=total_syncs / steps,
        copies_to_host_per_step=to_host / steps,
        sync_sites=[
            {"where": s["where"], "per_step": s["count"] / steps, "code": code(s["where"])}
            for s in sites
        ],
        top_ops=_top_ops(ops, gpu_busy is not None, code),
    )


def load(directory: Path, steps: int, repo: Path | None = None) -> ProfileSummary:
    trace = json.loads((directory / "trace.json").read_text())
    ops = json.loads((directory / "ops.json").read_text())
    syncs_file = directory / "syncs.json"
    syncs = json.loads(syncs_file.read_text()) if syncs_file.exists() else None
    return summarize(trace, ops, steps, syncs, repo)


def render(s: ProfileSummary) -> str:
    lines = [f"profiled {s.steps} steps, {s.window_ms / s.steps:.1f} ms/step under the profiler"]
    if s.gpu_busy is not None:
        lines.append(f"GPU busy {s.gpu_busy:.0%} of the time")
    lines.append(f"waiting on the DataLoader {s.data_wait:.0%} of the time")
    lines.append(
        f"host-device syncs per step: {s.syncs_per_step:.1f}"
        f" (copies to the CPU: {s.copies_to_host_per_step:.1f})"
    )
    lines += [f"  {x['per_step']:.1f} per step {_at(x)}" for x in s.sync_sites]
    unit = "GPU" if s.gpu_busy is not None else "CPU"
    lines.append(f"top ops by {unit} time:")
    lines += [f"  {op['share']:5.1%}  {op['name']}  {_at(op)}" for op in s.top_ops]
    return "\n".join(lines)


def _at(item: dict) -> str:
    if not item["where"]:
        return f"({NO_LINE})"
    at = "backward of" if item.get("backward") else "at"
    return f"{at} {item['where']}" + (f": {item['code']}" if item["code"] else "")


class _SourceLines:
    """Looks up the code at `path:line` in the repo, to show next to each location."""

    def __init__(self, repo: Path | None):
        self.repo = repo
        self.files: dict[str, list[str]] = {}

    def __call__(self, where: str | None) -> str:
        if not (self.repo and where):
            return ""
        path, _, line = where.rpartition(":")
        if path not in self.files:
            file = self.repo / path
            self.files[path] = (
                file.read_text(errors="replace").splitlines() if file.is_file() else []
            )
        lines = self.files[path]
        number = int(line)
        return lines[number - 1].strip()[:100] if 0 < number <= len(lines) else ""


def _covered(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def _top_ops(ops: list[dict], device: bool, code: _SourceLines) -> list[dict]:
    key = "self_device_us" if device else "self_cpu_us"
    total = sum(op[key] for op in ops)
    if not total:
        return []
    ranked = sorted(ops, key=lambda op: -op[key])[:TOP]
    return [
        {
            "name": op["name"],
            "share": op[key] / total,
            "where": op.get("where"),
            "backward": op.get("backward", False),
            "code": code(op.get("where")),
        }
        for op in ranked
        if op[key] > 0
    ]
