import pytest

from hotloop.scoreboard import harmful_attempt, outcome, recovered


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


def headline_record(target, recorded, speedup, keep=True, reason="1.50x faster, loss within noise"):
    return {
        "target": target,
        "repo": "https://github.com/someone/" + target,
        "change": "bf16 autocast",
        "gpu": "NVIDIA A100-SXM4-40GB",
        "recorded": recorded,
        "baseline": [{"step_ms": 100.0}, {"step_ms": 110.0}],
        "candidate": [{"step_ms": 60.0}, {"step_ms": 70.0}],
        "speedup": speedup,
        "keep": keep,
        "reason": reason,
    }


def test_headline_table_uses_the_latest_run_per_repo(tmp_path, monkeypatch):
    import json

    from hotloop import scoreboard

    (tmp_path / "survey").mkdir()
    (tmp_path / "survey" / "repos.csv").write_text("repo,stars\nsomeone/ddpm,1234\n")
    checks = [{"repo": "someone/vit", "stars": 99, "outside_merges_12_months": 0}]
    (tmp_path / "survey" / "maintainers.json").write_text(json.dumps({"repos": checks}))
    (tmp_path / "headline").mkdir()
    runs = [
        headline_record("ddpm", "2026-10-25T01:00:00", 1.2),
        headline_record("ddpm", "2026-10-25T02:00:00", 1.6),
        headline_record("vit", "2026-10-25T01:00:00", 1.01, False, "1.01x is within noise"),
    ]
    for i, record in enumerate(runs):
        (tmp_path / "headline" / f"{i}.json").write_text(json.dumps(record))
    monkeypatch.setattr(scoreboard, "RESULTS", tmp_path)

    rows = scoreboard.headline().splitlines()[2:]
    assert len(rows) == 2
    assert "1,234" in rows[0] and "A100-SXM4-40GB" in rows[0]
    assert "| 105.0 | 65.0 | 1.60x | within noise | bf16 autocast | not sent |" in rows[0]
    assert "| 99 |" in rows[1] and "1.01x (1.01x is within noise)" in rows[1]
    assert rows[1].endswith("| not sent: no outside pull request merged in 12 months |")


def test_headline_table_lists_repos_where_the_agent_kept_nothing(tmp_path, monkeypatch):
    import json

    from hotloop import scoreboard

    (tmp_path / "survey").mkdir()
    (tmp_path / "survey" / "repos.csv").write_text("repo,stars\n")
    (tmp_path / "survey" / "maintainers.json").write_text('{"repos": []}')
    (tmp_path / "headline").mkdir()
    folder = tmp_path / "eval" / "agent" / "ganilla"
    folder.mkdir(parents=True)
    run = {"target": "ganilla", "model": scoreboard.HEADLINE_MODEL, "gpu": "NVIDIA L4",
           "baseline_ms": 130.5, "kept": [], "speedup": 1.0, "attempts": []}  # fmt: skip
    (folder / "agent-gpt-6-luna-20261025-045305.json").write_text(json.dumps(run))
    monkeypatch.setattr(scoreboard, "RESULTS", tmp_path)

    [row] = scoreboard.headline().splitlines()[2:]
    assert row.startswith("| [giddyyupp/ganilla](https://github.com/giddyyupp/ganilla) |")
    assert "| L4 | 130.5 |  | nothing kept |" in row


def test_runs_are_ordered_by_time_and_kept_apart_by_model(tmp_path, monkeypatch):
    import json

    from hotloop import scoreboard

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
    assert scoreboard.setup_runs("agent")["ddpm"]["speedup"] == 1.1
