"""The keep/revert rule.

A change is kept only if it is faster beyond run-to-run noise (or lets a run
that ran out of memory finish) and its loss stays within the baseline's own
seed-to-seed variation.
"""

from dataclasses import dataclass
from statistics import mean

from hotloop.measurement import Measurement

# Smaller gains aren't worth a change to someone's repo, even when they beat the noise.
MIN_GAIN = 0.03
# Relative loss tolerance for repos where changing the seed barely moves the loss.
LOSS_FLOOR = 0.02


@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str
    speedup: float | None = None


def speedup(base: Measurement, cand: Measurement) -> float:
    return base.mean_step_ms / cand.mean_step_ms


def is_faster(base: Measurement, cand: Measurement) -> bool:
    """True if every candidate run beats every baseline run by at least MIN_GAIN."""
    return max(cand.step_ms) < min(base.step_ms) * (1 - MIN_GAIN)


def loss_tolerance(base: Measurement) -> float:
    spread = max(base.loss) - min(base.loss)
    return max(spread, LOSS_FLOOR * abs(mean(base.loss)))


def loss_ok(base: Measurement, cand: Measurement) -> bool:
    """True if each candidate run's loss is within tolerance of the baseline run
    with the same seed (same initial weights and data order)."""
    if base.seeds != cand.seeds:
        raise ValueError(f"seeds differ: {base.seeds} vs {cand.seeds}")
    if not cand.loss_finite:
        return False
    tolerance = loss_tolerance(base)
    return all(abs(c - b) <= tolerance for b, c in zip(base.loss, cand.loss, strict=True))


def decide(base: Measurement, cand: Measurement) -> Verdict:
    if base.gpu != cand.gpu:
        raise ValueError(f"never compare across GPUs: {base.gpu} vs {cand.gpu}")
    if not cand.completed:
        return Verdict(False, f"run failed: {cand.failure()}")
    if not base.completed:
        if base.oom and cand.loss_finite:
            return Verdict(True, "now fits in memory")
        return Verdict(False, f"baseline did not complete: {base.failure()}")
    if not base.loss_finite:
        return Verdict(False, "baseline has no finite loss to compare against")

    gain = speedup(base, cand)
    if not loss_ok(base, cand):
        moved = max(abs(c - b) for b, c in zip(base.loss, cand.loss, strict=True))
        return Verdict(
            False, f"loss moved by {moved:.4g}, more than {loss_tolerance(base):.4g}", gain
        )
    if not is_faster(base, cand):
        return Verdict(False, f"{gain:.2f}x is within noise", gain)
    return Verdict(True, f"{gain:.2f}x faster, loss within noise", gain)
