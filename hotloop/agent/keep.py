"""The keep/revert rule.

A change is kept only if it is faster beyond run-to-run noise (or lets a run
that ran out of memory finish) and its loss stays within the run-to-run
variation of the loss.

Runs are best measured in pairs: baseline and candidate in the same container,
one pair per seed. Containers can land on different CPUs and GPUs, and for
CPU-bound training that alone moved step time by 40% between runs, so
comparing within a pair cancels it out.
"""

import math
from dataclasses import dataclass
from statistics import geometric_mean, mean

from hotloop.measurement import Measurement

# Smaller gains aren't worth a change to someone's repo, even when they beat the noise.
MIN_GAIN = 0.03
# Relative loss tolerance for repos whose loss barely varies between runs.
LOSS_FLOOR = 0.02
# Two-sided 95% critical values of Student's t, by degrees of freedom.
T_95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}


@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str
    speedup: float | None = None


def paired(base: Measurement, cand: Measurement) -> bool:
    """True if each baseline run shared a container with the candidate run of the same seed."""
    containers = [r.get("container") for r in base.runs]
    return None not in containers and containers == [r.get("container") for r in cand.runs]


def speedup(base: Measurement, cand: Measurement) -> float:
    if paired(base, cand):
        return geometric_mean(b / c for b, c in zip(base.step_ms, cand.step_ms, strict=True))
    return base.mean_step_ms / cand.mean_step_ms


def is_faster(base: Measurement, cand: Measurement) -> bool:
    """True if the candidate wins by at least MIN_GAIN in every pair or, for unpaired
    runs, if every candidate run beats every baseline run by that much."""
    if paired(base, cand):
        return all(c < b * (1 - MIN_GAIN) for b, c in zip(base.step_ms, cand.step_ms, strict=True))
    return max(cand.step_ms) < min(base.step_ms) * (1 - MIN_GAIN)


def loss_tolerance(base: Measurement, cand: Measurement) -> float:
    """How far the mean loss may move: a two-sample t-test at 95%, using the
    run-to-run spread of the loss pooled over both configurations.

    Comparing each run with the baseline run of the same seed looked tighter but
    failed on chaotic training: with cuDNN autotuning and a high learning rate,
    two ResNet runs with the same seed and code ended 0.1 apart in loss.
    """
    deviations = [x - mean(m.loss) for m in (base, cand) for x in m.loss]
    dof = len(deviations) - 2
    pooled_sd = math.sqrt(sum(d * d for d in deviations) / dof)
    standard_error = pooled_sd * math.sqrt(1 / len(base.loss) + 1 / len(cand.loss))
    return max(T_95.get(dof, 2.26) * standard_error, LOSS_FLOOR * abs(mean(base.loss)))


def loss_ok(base: Measurement, cand: Measurement) -> bool:
    """True if the candidate's mean loss is within tolerance of the baseline's."""
    if base.seeds != cand.seeds:
        raise ValueError(f"seeds differ: {base.seeds} vs {cand.seeds}")
    if not cand.loss_finite:
        return False
    return abs(mean(cand.loss) - mean(base.loss)) <= loss_tolerance(base, cand)


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
        if not cand.loss_finite:
            return Verdict(False, "loss is not finite", gain)
        moved = abs(mean(cand.loss) - mean(base.loss))
        tolerance = loss_tolerance(base, cand)
        return Verdict(False, f"loss moved by {moved:.4g}, more than {tolerance:.4g}", gain)
    if not is_faster(base, cand):
        return Verdict(False, f"{gain:.2f}x is within noise", gain)
    return Verdict(True, f"{gain:.2f}x faster, loss within noise", gain)
