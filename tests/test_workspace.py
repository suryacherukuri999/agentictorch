import subprocess

from hotloop import targets, workspace
from hotloop.targets import Target


def git(repo, *args):
    command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, name, text):
    (repo / name).write_text(text)
    git(repo, "add", "--all")
    git(repo, "commit", "-qm", name)
    return git(repo, "rev-parse", "HEAD")


def test_diff_covers_kept_commits_and_edits_but_not_the_compatibility_tweaks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    pinned = commit(repo, "train.py", "speed = 'SLOW'\n")
    commit(repo, "compat.py", "import shim\n")
    commit(repo, "train.py", "speed = 'FAST'\n")  # a change the agent kept
    (repo / "loader.py").write_text("workers = 4\n")  # an edit not yet committed
    git(repo, "add", "loader.py")
    monkeypatch.setattr(targets, "TARGETS_DIR", tmp_path / "targets")
    (tmp_path / "targets" / "patched").mkdir(parents=True)
    (tmp_path / "targets" / "patched" / "compat.patch").write_text("")

    diff = workspace.diff(repo, Target(name="patched", command="python train.py", commit=pinned))
    assert "+speed = 'FAST'" in diff and "+workers = 4" in diff
    assert "shim" not in diff

    unpatched = Target(name="unpatched", command="python train.py", commit=pinned)
    assert "+import shim" in workspace.diff(repo, unpatched)
    planted = Target(name="planted", command="python train.py")
    assert "+import shim" in workspace.diff(repo, planted)
