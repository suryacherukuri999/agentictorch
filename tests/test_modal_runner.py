from pathlib import Path

from agentictorch.harness import modal_runner
from agentictorch.harness.modal_runner import Job
from agentictorch.targets import Target


def fake_run(calls):
    def run(target, job_lists, seeds, *, gpu, verbose):
        calls.append((job_lists, seeds))
        return [
            [{"repo": job.repo_dir.name, "seed": seed} for job in jobs]
            for jobs, seed in zip(job_lists, seeds, strict=True)
        ]

    return run


def test_pairs_get_one_container_per_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(modal_runner, "_run", fake_run(calls))
    base, cand = modal_runner.ab(Target("t", "python train.py"), Job(Path("a")), Job(Path("b")))
    [(job_lists, seeds)] = calls
    assert len(job_lists) == 2 and seeds == (0, 1)
    assert [r["repo"] for r in base] == ["a", "a"] and [r["seed"] for r in cand] == [0, 1]
