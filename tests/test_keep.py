import math

import pytest

from agentictorch.agent.keep import decide
from agentictorch.measurement import Measurement, tail_loss


def run(step_ms=100.0, loss=1.0, seed=0, gpu="NVIDIA L4", **extra):
    return {
        "completed": True,
        "step_ms": step_ms,
        "losses": [loss] * 20,
        "seed": seed,
        "gpu": gpu,
        **extra,
    }


def config(step_ms=(100.0, 102.0), loss=(1.00, 1.05), **extra):
    return Measurement(
        [run(t, x, seed, **extra) for seed, (t, x) in enumerate(zip(step_ms, loss, strict=True))]
    )


BASE = config()


def test_keeps_a_clear_gain_with_the_same_loss():
    verdict = decide(BASE, config(step_ms=(60.0, 61.0), loss=(1.01, 1.04)))
    assert verdict.keep
    assert verdict.speedup == pytest.approx(101 / 60.5)


def test_reverts_a_gain_inside_the_noise():
    noisy = config(step_ms=(100.0, 120.0))
    assert not decide(noisy, config(step_ms=(95.0, 105.0))).keep


def test_reverts_a_gain_below_the_minimum():
    tight = config(step_ms=(100.0, 100.5))
    assert not decide(tight, config(step_ms=(98.5, 98.6))).keep


def test_reverts_when_the_loss_moves_beyond_the_seed_spread():
    verdict = decide(BASE, config(step_ms=(60.0, 61.0), loss=(1.20, 1.25)))
    assert not verdict.keep
    assert "loss moved" in verdict.reason


def test_tolerates_loss_noise_from_chaotic_training():
    # Real ResNet numbers: an edit that can't change the math (an extra loss.item()),
    # with the loss of runs sharing a seed 0.1 to 0.2 apart.
    base = config(step_ms=(60.0, 61.0), loss=(2.035, 1.902))
    assert decide(base, config(step_ms=(40.0, 41.0), loss=(2.247, 1.801))).keep


def test_uses_a_relative_floor_when_seeds_do_not_move_the_loss():
    flat = config(loss=(1.0, 1.0))
    assert decide(flat, config(step_ms=(60.0, 61.0), loss=(1.015, 1.015))).keep
    assert not decide(flat, config(step_ms=(60.0, 61.0), loss=(1.03, 1.03))).keep


def test_reverts_a_nan_loss():
    assert not decide(BASE, config(step_ms=(60.0, 61.0), loss=(math.nan, 1.0))).keep


def test_reverts_when_the_baseline_loss_is_nan():
    verdict = decide(config(loss=(math.nan, 1.0)), config(step_ms=(60.0, 61.0)))
    assert not verdict.keep
    assert "baseline" in verdict.reason


def test_reverts_a_failed_run_and_says_why():
    failed = Measurement([run(), {"completed": False, "seed": 1, "gpu": "NVIDIA L4", "oom": True}])
    verdict = decide(BASE, failed)
    assert not verdict.keep
    assert "out of memory" in verdict.reason


def test_keeps_a_change_that_makes_the_run_fit_in_memory():
    oom = Measurement(
        [{"completed": False, "oom": True, "seed": s, "gpu": "NVIDIA L4"} for s in (0, 1)]
    )
    assert decide(oom, BASE).keep


def test_refuses_to_compare_across_gpus():
    with pytest.raises(ValueError):
        decide(BASE, config(gpu="NVIDIA T4"))


def test_refuses_to_compare_different_seeds():
    shifted = Measurement([run(60.0, 1.0, seed=s) for s in (1, 2)])
    with pytest.raises(ValueError):
        decide(BASE, shifted)


def test_tail_loss_uses_the_last_quarter_of_steps():
    assert tail_loss([10.0] * 30 + [2.0] * 10) == 2.0
    assert tail_loss([None, None]) is None
    assert tail_loss([4.0, None, 2.0]) == 3.0


def in_containers(m: Measurement, names: list[str]) -> Measurement:
    return Measurement([{**r, "container": name} for r, name in zip(m.runs, names, strict=True)])


def test_pairs_cancel_out_hardware_differences_between_containers():
    # The second container is 50% slower for both configurations.
    base = in_containers(config(step_ms=(20.0, 30.0)), ["a", "b"])
    cand = in_containers(config(step_ms=(16.0, 24.0)), ["a", "b"])
    verdict = decide(base, cand)
    assert verdict.keep
    assert verdict.speedup == pytest.approx(1.25)
    # Unpaired, the same numbers overlap and the rule can't call it.
    assert not decide(config(step_ms=(20.0, 30.0)), config(step_ms=(16.0, 24.0))).keep


def test_pairs_need_the_gain_in_every_pair():
    base = in_containers(config(step_ms=(20.0, 30.0)), ["a", "b"])
    cand = in_containers(config(step_ms=(16.0, 29.5)), ["a", "b"])
    assert not decide(base, cand).keep


def test_runs_in_different_containers_are_not_pairs():
    base = in_containers(config(step_ms=(20.0, 30.0)), ["a", "b"])
    cand = in_containers(config(step_ms=(16.0, 24.0)), ["c", "d"])
    assert not decide(base, cand).keep
