from pathlib import Path

from agentictorch.harness import modal_runner
from agentictorch.harness.modal_runner import Job
from agentictorch.targets import Target


def fake_run(calls):
    def run(target, job_lists, seeds, *, gpu, verbose):
        calls.append((job_lists, seeds))
        return [
            [{"repo": job.repo_dir.name, "seed": job.seed if job.seed is not None else seed}
             for job in jobs]
            for jobs, seed in zip(job_lists, seeds, strict=True)
        ]  # fmt: skip

    return run


def test_pairs_get_one_container_per_seed_on_a_single_gpu_model(monkeypatch):
    calls = []
    monkeypatch.setattr(modal_runner, "_run", fake_run(calls))
    base, cand = modal_runner.ab(Target("t", "python train.py"), Job(Path("a")), Job(Path("b")))
    [(job_lists, seeds)] = calls
    assert len(job_lists) == 2 and seeds == (0, 1)
    assert [r["repo"] for r in base] == ["a", "a"] and [r["seed"] for r in cand] == [0, 1]


def test_pairs_on_a_mixed_gpu_pool_share_one_container(monkeypatch):
    calls = []
    monkeypatch.setattr(modal_runner, "_run", fake_run(calls))
    target = Target("t", "python train.py")
    base, cand = modal_runner.ab(target, Job(Path("a")), Job(Path("b")), gpu="A100-40GB")
    [(job_lists, _)] = calls
    [jobs] = job_lists
    assert [(j.repo_dir.name, j.seed) for j in jobs] == [("a", 0), ("b", 0), ("b", 1), ("a", 1)]
    assert [(r["repo"], r["seed"]) for r in base] == [("a", 0), ("a", 1)]
    assert [(r["repo"], r["seed"]) for r in cand] == [("b", 0), ("b", 1)]
