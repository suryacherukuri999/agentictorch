import subprocess

import pytest

from agentictorch.agent import session as session_module
from agentictorch.agent.session import Session
from agentictorch.targets import Target
from agentictorch.tools.files import ToolError, apply_edit


def fake_run(repo, seed, container):
    fast = "FAST" in (repo / "train.py").read_text()
    return {
        "completed": True,
        "step_ms": 50.0 if fast else 100.0,
        "losses": [1.0] * 20,
        "seed": seed,
        "gpu": "NVIDIA L4",
        "container": container,
    }


@pytest.fixture
def session(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("speed = 'SLOW'\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "start"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)

    def ab(target, base, cand, gpu):
        runs = [
            (fake_run(base.repo_dir, s, f"c{s}"), fake_run(cand.repo_dir, s, f"c{s}"))
            for s in (0, 1)
        ]
        return [b for b, _ in runs], [c for _, c in runs]

    monkeypatch.setattr(session_module.modal_runner, "ab", ab)
    monkeypatch.setattr(session_module.workspace, "WORK", tmp_path / "work")
    return Session(Target(name="t", command="python train.py"), repo, rounds=2)


def test_keeps_a_faster_change_by_committing_it(session):
    apply_edit(session.repo, "train.py", "SLOW", "FAST")
    attempt = session.benchmark("make it fast")
    assert attempt.verdict.keep
    assert attempt.verdict.speedup == pytest.approx(2.0)
    assert session.diff() == ""
    assert "FAST" in (session.repo / "train.py").read_text()


def test_reverts_a_change_that_is_not_faster(session):
    apply_edit(session.repo, "train.py", "SLOW", "STILL SLOW")
    attempt = session.benchmark("rename")
    assert not attempt.verdict.keep
    assert (session.repo / "train.py").read_text() == "speed = 'SLOW'\n"


def test_refuses_to_benchmark_without_a_change_or_past_the_last_round(session):
    with pytest.raises(ToolError, match="nothing has changed"):
        session.benchmark("nothing")
    for word in ("A", "B"):
        apply_edit(session.repo, "train.py", "SLOW", word)
        session.benchmark(word)
    apply_edit(session.repo, "train.py", "SLOW", "C")
    with pytest.raises(ToolError, match="no benchmark rounds left"):
        session.benchmark("C")


def test_final_compares_the_original_with_everything_kept(session):
    apply_edit(session.repo, "train.py", "SLOW", "FAST")
    session.benchmark("make it fast")
    final = session.final()
    assert final.verdict.keep
    assert "+speed = 'FAST'" in final.diff


def test_syntax_errors_are_caught_before_using_a_round(session):
    apply_edit(session.repo, "train.py", "speed = 'SLOW'", "speed = ('SLOW'")
    with pytest.raises(ToolError, match="syntax error in train.py line 1"):
        session.benchmark("broken")
    assert session.attempts == []


def test_quick_check_uses_no_round_and_returns_the_log_on_a_crash(session, monkeypatch):
    calls = []

    def measure(target, repo, gpu, seeds, steps, warmup):
        calls.append(repo)
        crashed = "BROKEN" in (repo / "train.py").read_text()
        run = fake_run(repo, 0, "c0") | {"completed": not crashed}
        if crashed:
            run |= {
                "error": "NameError: x",
                "log_tail": 'File "/work/repo0/train.py", line 1\nNameError: x',
            }
        return [run]

    monkeypatch.setattr(session_module.modal_runner, "measure", measure)
    with pytest.raises(ToolError, match="nothing has changed"):
        session.check()
    apply_edit(session.repo, "train.py", "SLOW", "BROKEN")
    text = session.check()
    assert text.startswith("failed: NameError: x")
    assert 'File "train.py", line 1' in text
    assert session.rounds_left == session.rounds
    assert calls == [session.repo]  # the working tree itself, edits included


@pytest.mark.parametrize(
    "line",
    [
        "with torch.autocast(device_type='cuda', dtype=torch.float16):",
        "with torch.autocast('cuda'):",
        "model = model.half()",
    ],
)
def test_float16_without_a_grad_scaler_is_refused(session, line):
    (session.repo / "train.py").write_text(f"speed = 'SLOW'\n{line}\n")
    with pytest.raises(ToolError, match="GradScaler"):
        session.check_precision()


def test_bfloat16_or_float16_with_a_grad_scaler_is_allowed(session):
    (session.repo / "train.py").write_text(
        "speed = 'SLOW'\nwith torch.autocast(device_type='cuda', dtype=torch.bfloat16):\n"
    )
    session.check_precision()
    (session.repo / "train.py").write_text(
        "speed = 'SLOW'\nscaler = torch.amp.GradScaler()\n"
        "with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
    )
    session.check_precision()
