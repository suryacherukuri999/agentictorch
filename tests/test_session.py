import subprocess

import pytest

from hotloop.agent import session as session_module
from hotloop.agent.session import Session
from hotloop.targets import Target
from hotloop.tools.files import ToolError, apply_edit


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


def test_refuses_to_benchmark_without_a_change_or_past_the_budget(session):
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


def test_syntax_errors_are_caught_before_spending_a_round(session):
    apply_edit(session.repo, "train.py", "speed = 'SLOW'", "speed = ('SLOW'")
    with pytest.raises(ToolError, match="syntax error in train.py line 1"):
        session.benchmark("broken")
    assert session.attempts == []
