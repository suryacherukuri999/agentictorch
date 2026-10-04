from agentictorch import cli


def test_resolves_real_repos_and_bench_cases(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli.workspace, "working_copy", lambda t, label: calls.append((t.name, label)) or tmp_path
    )
    repo = cli.load_target("pytorch-ddpm")
    assert cli.working_copy(repo, "baseline") == tmp_path
    assert calls == [("pytorch-ddpm", "baseline")]

    case = cli.load_target("bench-resnet-workers0")
    assert case.name == "bench-resnet-workers0"
    assert "bench-resnet-workers0" in cli.target_names()


def test_cpu_means_no_gpu():
    assert cli.gpu_type("cpu") is None
    assert cli.gpu_type("L4") == "L4"
