import json

import pytest

from agentictorch.scoreboard import harmful_attempt, outcome, recovered


def record(reason=None, measured=None, speedup=1.0):
    attempts = [{"change": "x", "keep": False, "reason": reason}] if reason else []
    return {"attempts": attempts, "measured_speedup": measured, "speedup": speedup}


def test_recovered_share_of_a_planted_slowdown():
    assert recovered(2.0, 2.0) == pytest.approx(1.0)
    assert recovered(1.5, 2.0) == pytest.approx((1 - 1 / 1.5) / 0.5)
    assert recovered(1.0, 1.0) is None


def test_blind_patches_ship_whatever_they_do():
    assert outcome(record("run failed: SyntaxError"), "llm") == (1.0, True)
    assert outcome(record("loss moved by 0.5, more than 0.1", 2.0), "llm-kb") == (1.0, True)
    assert outcome(record("0.80x is within noise", 0.8), "llm") == (0.8, True)
    assert outcome(record("1.01x is within noise", 1.01), "rules") == (1.01, False)
    assert outcome(record(), "rules") == (1.0, False)


def test_the_agent_ships_only_what_it_kept():
    assert outcome(record("run failed: SyntaxError", speedup=2.5), "agent") == (2.5, False)


def test_harmful_attempts_are_crashes_loss_changes_and_slowdowns():
    assert harmful_attempt({"reason": "run failed: NameError"})
    assert harmful_attempt({"reason": "loss moved by 0.4, more than 0.1", "speedup": 1.5})
    assert harmful_attempt({"reason": "0.90x is within noise", "speedup": 0.9})
    assert not harmful_attempt({"reason": "1.01x is within noise", "speedup": 1.01})
    assert not harmful_attempt({"reason": "1.50x faster, loss within noise", "speedup": 1.5})


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_headline_table_has_one_row_per_repo_from_its_final_pair(tmp_path, monkeypatch):
    from agentictorch import scoreboard

    (tmp_path / "survey").mkdir()
    (tmp_path / "survey" / "repos.csv").write_text("repo,stars\nw86763777/pytorch-ddpm,1234\n")
    save(
        tmp_path / "survey/maintainers.json",
        {"repos": [{"repo": "lucidrains/vit-pytorch", "stars": 99}]},
    )
    save(tmp_path / "fixes.json", {"vit-pytorch": "fused attention", "pytorch-ddpm": "bf16"})
    sol = scoreboard.SOL_MODEL
    pair = {"baseline_ms": [30.0, 26.0], "candidate_ms": [12.0, 10.0]}
    save(tmp_path / "eval/agent/vit-pytorch/agent-gpt-6.1-sol-20261025-010000.json",
         {"target": "vit-pytorch", "model": sol, "speedup": 2.43, "kept": ["x"],
          "final_pair": pair, "attempts": []})  # fmt: skip
    save(tmp_path / "eval/agent/detr/agent-gpt-6.1-sol-20261025-010000.json",
         {"target": "detr", "model": sol, "speedup": 1.0, "kept": [], "final_pair": None,
          "baseline_ms": 207.9, "attempts": []})  # fmt: skip
    # A run that lost its report counts through its final pair, measured again.
    save(tmp_path / "final-pairs/pytorch-ddpm.json",
         {"target": "pytorch-ddpm", "speedup": 2.5, "keep": True,
          "baseline": [{"step_ms": 540.0}, {"step_ms": 560.0}],
          "candidate": [{"step_ms": 210.0}, {"step_ms": 230.0}]})  # fmt: skip
    monkeypatch.setattr(scoreboard, "RESULTS", tmp_path)

    ddpm, vit, detr = scoreboard.headline().splitlines()[2:]
    assert ddpm.startswith(
        "| [w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) |"
    )
    assert ddpm.endswith("| 1,234 | 550.0 → 220.0 | 2.50x | bf16 |")
    assert vit.endswith("| 99 | 28.0 → 11.0 | 2.43x | fused attention |")
    assert detr.endswith("|  | 207.9 | nothing kept |  |")


def test_runs_are_ordered_by_time_and_kept_apart_by_model(tmp_path, monkeypatch):
    from agentictorch import scoreboard

    folder = tmp_path / "eval" / "agent" / "ddpm"
    folder.mkdir(parents=True)
    for label, model, speedup in [
        ("agent-ornith-9b-20261025-010000", "ornith-1.5:9b", 1.1),
        ("agent-gpt-6-luna-20261025-020000", "gpt-6-luna", 1.3),
        ("agent-gemini-flash-20261025-030000", "gemini-3.8-flash", 1.2),
    ]:
        record = {"target": "ddpm", "model": model, "speedup": speedup, "attempts": []}
        (folder / f"{label}.json").write_text(json.dumps(record))
    monkeypatch.setattr(scoreboard, "RESULTS", tmp_path)

    assert scoreboard.latest("agent")["ddpm"]["speedup"] == 1.2
    assert scoreboard.latest("agent", "gpt-6-luna")["ddpm"]["speedup"] == 1.3
    assert scoreboard.setup_runs("agent")["ddpm"]["speedup"] == 1.3
    assert scoreboard.setup_runs("agent", scoreboard.LOCAL_MODEL)["ddpm"]["speedup"] == 1.1


def test_compare_models_counts_harmful_blind_patches_and_averages(tmp_path, monkeypatch):
    from agentictorch import scoreboard

    luna, sol = scoreboard.MAIN_MODEL, scoreboard.MODELS["gpt-6.1-sol"].name
    for target, a, b in [("ddpm", 1.29, 2.54), ("nerf", 1.18, None)]:
        run = {"target": target, "model": luna, "speedup": a, "kept": ["x"], "attempts": []}
        save(tmp_path / f"eval/agent/{target}/agent-gpt-6-luna-20261025-010000.json", run)
        if b:
            save(tmp_path / f"final-pairs/{target}-sol.json",
                 {"target": target, "speedup": b, "keep": True})  # fmt: skip
    save(tmp_path / "eval/agent/nerf/agent-gpt-6.1-sol-20261025-020000.json",
         {"target": "nerf", "model": sol, "speedup": 1.0, "kept": [], "attempts": []})  # fmt: skip
    crash = {"change": "x", "keep": False, "reason": "run failed: SyntaxError"}
    save(tmp_path / "eval/llm/ddpm/llm-gpt-6.1-sol-20261025-030000.json",
         {"target": "ddpm", "model": sol, "measured_speedup": None, "speedup": 1.0,
          "attempts": [crash]})  # fmt: skip
    save(tmp_path / "eval/llm/nerf/llm-gpt-6.1-sol-20261025-030000.json",
         {"target": "nerf", "model": sol, "measured_speedup": 1.5, "speedup": 1.5,
          "attempts": [{"change": "x", "keep": True, "reason": "1.50x faster"}]})  # fmt: skip
    loss = {"change": "x", "keep": False, "reason": "loss moved by 0.5, more than 0.1"}
    save(tmp_path / "eval/llm/ddpm/llm-gpt-6-luna-20261025-040000.json",
         {"target": "ddpm", "model": luna, "measured_speedup": 1.2, "speedup": 1.2,
          "attempts": [{"change": "x", "keep": True, "reason": "1.20x faster"}]})  # fmt: skip
    save(tmp_path / "eval/llm/nerf/llm-gpt-6-luna-20261025-040000.json",
         {"target": "nerf", "model": luna, "measured_speedup": 2.0, "speedup": 1.0,
          "attempts": [loss]})  # fmt: skip
    monkeypatch.setattr(scoreboard, "RESULTS", tmp_path)

    lines = scoreboard.compare_models().splitlines()
    assert lines[2] == "| ddpm | 2.54x | 1.00x harmful | 1.29x | 1.20x |"
    assert lines[3] == "| nerf | 1.00x | 1.50x | 1.18x | 1.00x harmful |"
    assert lines[4] == "| **Geometric mean** | **1.59x** | **1.22x** | **1.23x** | **1.10x** |"
    assert lines[5] == "| Harmful changes shipped | 0 | 1 | 0 | 1 |"

    gain = scoreboard.loop_gain().splitlines()
    assert gain[2] == "| gpt-6.1-sol | 1.59x | 1.22x | 1.30x | 0, 1 |"
    assert gain[3] == "| gpt-6-luna | 1.23x | 1.10x | 1.13x | 0, 1 |"
