"""The rules-only setup: fixed edits for problems the static detectors find.

TorchFix (checked 2026-10-24, version 0.7.0) only flags deprecated and unsafe
APIs, so these few AST rules stand in for a rule-based fixer.
"""

import ast
from pathlib import Path

from agentictorch.survey.detect import _enclosing_loop, _is_backward, _is_training_loader

WORKERS = 4


def apply(repo: Path) -> list[str]:
    """Apply every rule to every Python file in the repo; return what was changed."""
    changes = []
    for path in sorted(repo.rglob("*.py")):
        if ".git" in path.relative_to(repo).parts:
            continue
        source = path.read_text(errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        edits = _loader_edits(tree, source) + _sync_edits(tree, source)
        # Apply from the end so earlier offsets stay valid.
        for start, end, text, change in sorted(edits, key=lambda e: -e[0]):
            source = source[:start] + text + source[end:]
            line = source.count("\n", 0, start) + 1
            changes.append(f"{path.relative_to(repo)}:{line}: {change}")
        if edits:
            path.write_text(source)
    return changes


def _offset(source: str, lineno: int, col: int) -> int:
    """Character offset of an AST position (1-based line, UTF-8 byte column)."""
    lines = source.splitlines(keepends=True)
    before = sum(len(line) for line in lines[: lineno - 1])
    return before + len(lines[lineno - 1].encode()[:col].decode(errors="ignore"))


def _loader_edits(tree: ast.AST, source: str):
    """Give main-thread training DataLoaders worker processes and pinned memory."""
    edits = []
    for node in ast.walk(tree):
        if not _is_training_loader(node) or any(kw.arg is None for kw in node.keywords):
            continue
        given = {kw.arg: kw.value for kw in node.keywords}
        workers = given.get("num_workers")
        if workers is not None and not (isinstance(workers, ast.Constant) and workers.value == 0):
            continue  # set by the repo, maybe from a config: leave it alone
        if workers is not None:
            start = _offset(source, workers.lineno, workers.col_offset)
            end = _offset(source, workers.end_lineno, workers.end_col_offset)
            edits.append((start, end, str(WORKERS), "DataLoader workers"))
        added = ([] if workers is not None else [f"num_workers={WORKERS}"]) + (
            [] if "pin_memory" in given else ["pin_memory=True"]
        )
        if added:
            last = max([*node.args, *node.keywords], key=lambda n: (n.end_lineno, n.end_col_offset))
            at = _offset(source, last.end_lineno, last.end_col_offset)
            edits.append((at, at, "".join(f", {a}" for a in added), "DataLoader workers"))
    return edits


def _sync_edits(tree: ast.AST, source: str):
    """Drop torch.cuda.synchronize() and empty_cache() statements from training loops."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    loops = {_enclosing_loop(n, parents) for n in ast.walk(tree) if _is_backward(n)} - {None}
    edits = []
    for loop in loops:
        for stmt in ast.walk(loop):
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            name = ast.unparse(stmt.value.func)
            if name.endswith(("cuda.synchronize", "cuda.empty_cache")) and not _alone(
                stmt, parents[stmt]
            ):
                start, end = (
                    _line_start(source, stmt.lineno),
                    _line_start(source, stmt.end_lineno + 1),
                )
                edits.append((start, end, "", f"removed {name}()"))
    return edits


def _alone(stmt: ast.stmt, parent: ast.AST) -> bool:
    """True if the statement is its block's only one, so removing it would empty the block."""
    for field in ("body", "orelse", "finalbody"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and stmt in block:
            return len(block) == 1
    return False


def _line_start(source: str, lineno: int) -> int:
    return sum(len(line) for line in source.splitlines(keepends=True)[: lineno - 1])
