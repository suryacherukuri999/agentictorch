import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hotloop.harness import runner

FIXTURES = Path(__file__).parent / "fixtures"


def train(args: str = "", *, steps: int = 20, warmup: int = 5, seed: int = 0) -> dict:
    command = f"{sys.executable} tiny_train.py {args}"
    return runner.run(command, FIXTURES, steps=steps, warmup=warmup, seed=seed, timeout=120)


def test_stops_after_warmup_plus_steps():
    result = train(steps=20, warmup=5)
    assert result["completed"], result["log_tail"]
    assert result["exit_code"] == 0
    assert result["steps"] == 20
    assert len(result["step_ms_each"]) == 20
    assert result["step_ms"] == pytest.approx(sum(result["step_ms_each"]) / 20)
    assert len(result["losses"]) == 25
    assert all(math.isfinite(loss) for loss in result["losses"])
    assert "finished" not in result["log_tail"]


def test_stops_cleanly_with_dataloader_workers():
    result = train("--workers 2")
    assert result["completed"], result["log_tail"]
    assert not result["timed_out"]


def test_counts_one_step_per_iteration_with_two_optimizers():
    result = train("--two-optimizers", steps=20, warmup=5)
    assert result["completed"], result["log_tail"]
    assert result["steps"] == 20
    # The second optimizer steps after the first, so it is one behind at the stop.
    assert result["other_optimizer_steps"] == 24


def test_rebuilt_optimizer_is_still_the_clock():
    # 16 steps per epoch, so the run crosses two optimizer rebuilds.
    result = train("--rebuild-optimizer", steps=30, warmup=5)
    assert result["completed"], result["log_tail"]
    assert result["other_optimizer_steps"] == 0


def test_seed_repeats_and_varies_the_run():
    first, again, other = train(seed=0), train(seed=0), train(seed=1)
    assert first["losses"] == again["losses"]
    assert first["losses"] != other["losses"]


def test_seed_shifts_the_scripts_own_seed():
    first, again, other = (
        train("--seed 7", seed=0),
        train("--seed 7", seed=0),
        train("--seed 7", seed=1),
    )
    assert first["losses"] == again["losses"]
    assert first["losses"] != other["losses"]


def test_records_the_unscaled_loss_with_a_grad_scaler():
    plain, scaled = train(), train("--scaler")
    assert scaled["completed"], scaled["log_tail"]
    assert scaled["losses"][:3] == pytest.approx(plain["losses"][:3], rel=1e-4)


def test_crash_is_reported_with_partial_steps():
    result = train("--crash-at 9", warmup=5)
    assert not result["completed"]
    assert result["exit_code"] != 0
    assert "planted crash" in result["error"]
    assert result["steps"] == 3


def test_run_that_ends_early_is_incomplete():
    result = train("--epochs 1", steps=50, warmup=5)  # 16 steps in an epoch
    assert not result["completed"]
    assert result["exit_code"] == 0
    assert result["error"] is None
    assert result["steps"] == 11


def test_measures_time_spent_waiting_for_data():
    result = train("--delay 0.002", steps=10, warmup=3)  # 16 samples, ~32 ms per batch
    assert result["completed"], result["log_tail"]
    assert result["data_wait_ms"] > 20
    assert result["data_wait_ms"] <= result["step_ms"]


def test_harness_is_inactive_without_an_output_path():
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOTLOOP_")}
    env["PYTHONPATH"] = str(runner.INJECT_DIR)
    out = subprocess.run(
        [sys.executable, "tiny_train.py", "--epochs", "1"],
        cwd=FIXTURES,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "finished 16" in out.stdout


def test_profile_run_summarizes_the_measured_steps():
    command = f"{sys.executable} tiny_train.py --delay 0.002"
    result = runner.run(command, FIXTURES, steps=5, warmup=3, timeout=120, profile=True)
    assert result["completed"], result["log_tail"]
    profile = result["profile"]
    assert profile["steps"] == 5
    assert profile["gpu_busy"] is None
    assert profile["data_wait"] > 0.3
    assert profile["top_ops"]
    assert "waiting on the DataLoader" in result["profile_text"]


def test_profile_names_the_exact_repo_line_behind_each_op():
    command = f"{sys.executable} tiny_train.py"
    result = runner.run(command, FIXTURES, steps=5, warmup=3, timeout=120, profile=True)
    ops = {op["name"]: op for op in result["profile"]["top_ops"]}
    forward_line = "tiny_train.py:55"  # loss = nn.functional.mse_loss(model(x), y)
    assert ops["aten::addmm"]["where"] == forward_line
    assert not ops["aten::addmm"]["backward"]
    backward = [op for op in result["profile"]["top_ops"] if op["backward"]]
    assert backward and all(op["where"] == forward_line for op in backward)
    # A range PyTorch opens itself gets the line of the call that opened it.
    assert ops["Optimizer.step#SGD.step"]["where"] == "tiny_train.py:58"
    assert "at tiny_train.py:58: scaler.step(opt)" in result["profile_text"]
