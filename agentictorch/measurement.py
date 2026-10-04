"""Summaries of harness runs: one Measurement is one configuration run once per seed."""

import math
import re
from dataclasses import dataclass
from statistics import mean


def tail_loss(losses: list[float | None] | None, fraction: float = 0.25, min_steps: int = 5):
    """Mean loss over the last part of the run, or None if no loss was recorded."""
    values = [x for x in losses or () if x is not None]
    if not values:
        return None
    n = max(min_steps, round(len(values) * fraction))
    return mean(values[-n:])


@dataclass(frozen=True)
class Measurement:
    runs: list[dict]

    @property
    def completed(self) -> bool:
        return bool(self.runs) and all(r.get("completed") for r in self.runs)

    @property
    def oom(self) -> bool:
        return any(r.get("oom") for r in self.runs)

    @property
    def seeds(self) -> list[int]:
        return [r.get("seed") for r in self.runs]

    @property
    def gpu(self) -> str:
        names = {r.get("gpu") for r in self.runs}
        if len(names) != 1:
            raise ValueError(f"runs of one configuration used different GPUs: {sorted(names)}")
        return names.pop()

    @property
    def step_ms(self) -> list[float]:
        return [r["step_ms"] for r in self.runs]

    @property
    def mean_step_ms(self) -> float:
        return mean(self.step_ms)

    @property
    def noise(self) -> float:
        """Spread of the step time across runs, relative to the mean."""
        return (max(self.step_ms) - min(self.step_ms)) / self.mean_step_ms

    @property
    def loss(self) -> list[float | None]:
        return [tail_loss(r.get("losses")) for r in self.runs]

    @property
    def loss_finite(self) -> bool:
        return all(x is not None and math.isfinite(x) for x in self.loss)

    @property
    def peak_mem(self) -> int | None:
        peaks = [r["peak_mem_allocated"] for r in self.runs if r.get("peak_mem_allocated")]
        return max(peaks) if peaks else None

    @property
    def data_wait_share(self) -> float | None:
        shares = [
            r["data_wait_ms"] / r["step_ms"]
            for r in self.runs
            if r.get("data_wait_ms") is not None and r.get("step_ms")
        ]
        return mean(shares) if shares else None

    def failure(self) -> str:
        """Why the configuration did not complete, for reports and the agent."""
        for r in self.runs:
            if r.get("completed"):
                continue
            if r.get("oom"):
                return "out of memory"
            if r.get("timed_out"):
                return "timed out"
            if r.get("error"):
                return r["error"].splitlines()[-1]
            exit_code = r.get("exit_code")
            if r.get("steps") is not None:
                return f"stopped after {r['steps']} measured steps (exit code {exit_code})"
            return f"no results (exit code {exit_code})"
        return ""

    def failure_log(self, lines: int = 15) -> str:
        """The end of the first failed run's log, with container paths made relative
        to the repo, so a traceback reads as repo file and line."""
        for r in self.runs:
            if not r.get("completed"):
                tail = (r.get("log_tail") or "").splitlines()[-lines:]
                return re.sub(r"/work/repo\d+/", "", "\n".join(tail))
        return ""
