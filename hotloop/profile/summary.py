"""Turn a PyTorch profiler run into a short summary the agent can read.

The raw trace never goes into a prompt: a few steps of a real model produce
megabytes of events. This runs inside the measurement container, so it uses
only the standard library.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

GPU_EVENTS = {"kernel", "gpu_memcpy", "gpu_memset"}
# .item(), .cpu() and friends end in one of these, so counting them counts syncs.
SYNC_CALLS = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"}
TOP = 8


@dataclass
class ProfileSummary:
    steps: int
    window_ms: float
    gpu_busy: float | None  # share of the window with a kernel or copy running
    data_wait: float  # share of the window spent waiting on the DataLoader
    syncs_per_step: float  # host waits for the GPU: .item(), .cpu(), synchronize()
    copies_to_host_per_step: float
    top_ops: list[tuple[str, float]]  # (op, share of device time, or CPU time on CPU)


def summarize(trace: dict, ops: list[dict], steps: int) -> ProfileSummary:
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
    syncs = sum(names[n] for n in SYNC_CALLS)
    to_host = sum(1 for e in events if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"])

    return ProfileSummary(
        steps=steps,
        window_ms=window / 1000,
        gpu_busy=gpu_busy,
        data_wait=data / window,
        syncs_per_step=syncs / steps,
        copies_to_host_per_step=to_host / steps,
        top_ops=_top_ops(ops, device=gpu_busy is not None),
    )


def load(directory: Path, steps: int) -> ProfileSummary:
    trace = json.loads((directory / "trace.json").read_text())
    ops = json.loads((directory / "ops.json").read_text())
    return summarize(trace, ops, steps)


def render(s: ProfileSummary) -> str:
    lines = [f"profiled {s.steps} steps, {s.window_ms / s.steps:.1f} ms/step under the profiler"]
    if s.gpu_busy is not None:
        lines.append(f"GPU busy {s.gpu_busy:.0%} of the time")
    lines.append(f"waiting on the DataLoader {s.data_wait:.0%} of the time")
    lines.append(
        f"host-device syncs per step: {s.syncs_per_step:.1f}"
        f" (copies to the CPU: {s.copies_to_host_per_step:.1f})"
    )
    unit = "GPU" if s.gpu_busy is not None else "CPU"
    lines.append(f"top ops by {unit} time:")
    lines += [f"  {share:5.1%}  {name}" for name, share in s.top_ops]
    return "\n".join(lines)


def _covered(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def _top_ops(ops: list[dict], device: bool) -> list[tuple[str, float]]:
    key = "self_device_us" if device else "self_cpu_us"
    total = sum(op[key] for op in ops)
    if not total:
        return []
    ranked = sorted(ops, key=lambda op: -op[key])[:TOP]
    return [(op["name"], op[key] / total) for op in ranked if op[key] > 0]
