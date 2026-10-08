import shutil
import sys
from pathlib import Path

from agentictorch.harness import local_runner
from agentictorch.harness.runner import Job
from agentictorch.targets import Target

FIXTURES = Path(__file__).parent / "fixtures"


def use_this_python(monkeypatch, tmp_path):
    # Building a fresh environment with torch is slow; the test environment has one.
    monkeypatch.setattr(local_runner, "environment", lambda target: Path(sys.executable).parent)
    monkeypatch.setattr(local_runner, "DATA", tmp_path / "data")


def tiny_target(**fields) -> Target:
    defaults = {"command": "python tiny_train.py", "steps": 3, "warmup": 1, "timeout": 120}
    return Target(name="tiny", **{**defaults, **fields})


def test_ab_keeps_each_side_with_its_own_runs_when_the_order_alternates(tmp_path, monkeypatch):
    use_this_python(monkeypatch, tmp_path)
    base = shutil.copytree(FIXTURES, tmp_path / "base")
    cand = shutil.copytree(FIXTURES, tmp_path / "cand")
    (cand / "tiny_train.py").write_text("raise SystemExit(3)\n")

    base_runs, cand_runs = local_runner.ab(tiny_target(), Job(base), Job(cand))

    assert [r["completed"] for r in base_runs] == [True, True]
    assert [r["exit_code"] for r in cand_runs] == [3, 3]
    # Both runs of a seed count as one pair on the same machine.
    assert [r["container"] for r in base_runs] == [r["container"] for r in cand_runs]


def test_runs_find_the_dataset_where_the_repo_expects_it(tmp_path, monkeypatch):
    use_this_python(monkeypatch, tmp_path)
    (tmp_path / "data" / "toy").mkdir(parents=True)
    (tmp_path / "data" / "toy" / "marker").write_text("x")
    repo = shutil.copytree(FIXTURES, tmp_path / "repo")
    target = tiny_target(
        command="test -f inputs/marker && python tiny_train.py", dataset="toy", data_path="inputs"
    )

    [run] = local_runner.measure(target, repo, seeds=(0,))

    assert run["completed"], run["log_tail"]


def test_download_fills_the_dataset_folder_once(tmp_path, monkeypatch):
    use_this_python(monkeypatch, tmp_path)
    target = tiny_target(dataset="toy", data_path="inputs", download="echo hello >> file.txt")

    assert local_runner.download(target) == "1 files, 0 MiB"
    local_runner.download(target)

    assert (tmp_path / "data" / "toy" / "file.txt").read_text() == "hello\n"
