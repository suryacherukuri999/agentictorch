import pytest

from hotloop.tools.files import ToolError, apply_edit, list_files, read_file, search


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "train.py").write_text(
        "loader = DataLoader(ds, shuffle=True)\nfor x in loader:\n    step(x)\n"
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "model.py").write_text("import torch\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("")
    return tmp_path


def test_apply_edit_replaces_one_occurrence_and_returns_a_diff(repo):
    diff = apply_edit(repo, "train.py", "shuffle=True)", "shuffle=True, num_workers=4)")
    assert "+loader = DataLoader(ds, shuffle=True, num_workers=4)" in diff
    assert "num_workers=4" in (repo / "train.py").read_text()


def test_apply_edit_refuses_missing_or_ambiguous_text(repo):
    with pytest.raises(ToolError, match="not found"):
        apply_edit(repo, "train.py", "num_workers=0", "num_workers=4")
    (repo / "twice.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(ToolError, match="2 times"):
        apply_edit(repo, "twice.py", "x = 1", "x = 2")
    assert (repo / "twice.py").read_text() == "x = 1\nx = 1\n"


def test_tools_stay_inside_the_repo(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.py"
    outside.write_text("token = 1\n")
    with pytest.raises(ToolError, match="outside the repo"):
        read_file(repo, f"../{outside.parent.name}/secret.py")


def test_read_file_numbers_lines(repo):
    assert read_file(repo, "train.py", 2, 3) == "    2  for x in loader:\n    3      step(x)"


def test_list_files_and_search_skip_git(repo):
    assert list_files(repo) == ["pkg/model.py", "train.py"]
    assert search(repo, r"DataLoader\(") == ["train.py:1: loader = DataLoader(ds, shuffle=True)"]
