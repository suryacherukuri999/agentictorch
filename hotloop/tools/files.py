"""File tools for the agent. They only ever touch the working copy."""

import difflib
import re
from pathlib import Path

MAX_LINES = 400


class ToolError(Exception):
    """A mistake the model can fix; its message goes back to the model."""


def resolve(repo: Path, path: str) -> Path:
    full = (repo / path).resolve()
    if not full.is_relative_to(repo.resolve()):
        raise ToolError(f"{path} is outside the repo")
    if not full.is_file():
        raise ToolError(f"{path} does not exist")
    return full


def list_files(repo: Path, pattern: str = "**/*.py") -> list[str]:
    root = repo.resolve()
    return sorted(
        str(p.relative_to(root))
        for p in root.glob(pattern)
        if p.is_file() and ".git" not in p.relative_to(root).parts
    )


def read_file(repo: Path, path: str, start: int = 1, end: int | None = None) -> str:
    """Return lines start..end (1-based, inclusive) of a file, numbered."""
    lines = resolve(repo, path).read_text(errors="replace").splitlines()
    end = min(end or len(lines), len(lines), start + MAX_LINES - 1)
    # A tab after the number keeps the line's own indentation unambiguous.
    numbered = [f"{i:5d}\t{lines[i - 1]}" for i in range(max(start, 1), end + 1)]
    more = f"\n... {len(lines) - end} more lines" if end < len(lines) else ""
    return "\n".join(numbered) + more


def _replace_ignoring_indent(text: str, old: str, new: str) -> str | None:
    lines = text.splitlines(keepends=True)
    want = [line.strip() for line in old.strip("\n").splitlines()]
    if not any(want):
        return None
    starts = [
        i
        for i in range(len(lines) - len(want) + 1)
        if all(lines[i + j].strip() == w for j, w in enumerate(want))
    ]
    if len(starts) != 1:
        return None
    first = lines[starts[0]]
    old_first = old.strip("\n").splitlines()[0]
    shift = _indent(first) - _indent(old_first)
    shifted = [
        " " * max(_indent(line) + shift, 0) + line.lstrip(" ") if line.strip() else line
        for line in new.strip("\n").splitlines()
    ]
    replacement = "".join(f"{line}\n" for line in shifted)
    return "".join(lines[: starts[0]]) + replacement + "".join(lines[starts[0] + len(want) :])


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def search(repo: Path, pattern: str, glob: str = "**/*.py", limit: int = 50) -> list[str]:
    """Lines matching a regular expression, as path:line: text."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regular expression: {e}") from None
    hits = []
    for rel in list_files(repo, glob):
        for i, line in enumerate((repo / rel).read_text(errors="replace").splitlines(), 1):
            if regex.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
                if len(hits) == limit:
                    return hits
    return hits


def apply_edit(repo: Path, path: str, old: str, new: str) -> str:
    """Replace the one occurrence of `old` with `new` and return the diff.

    If `old` doesn't match exactly, lines that match apart from indentation are
    accepted when they occur exactly once, and `new` is shifted by the same amount.
    Small models often get the indentation of copied lines wrong.
    """
    file = resolve(repo, path)
    text = file.read_text()
    count = text.count(old)
    if count > 1:
        raise ToolError(
            f"the text to replace appears {count} times in {path}; include more context"
        )
    updated = text.replace(old, new) if count == 1 else _replace_ignoring_indent(text, old, new)
    if updated is None:
        raise ToolError(f"the text to replace was not found in {path}; read the file again")
    file.write_text(updated)
    return "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
