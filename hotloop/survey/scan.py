"""Scan one cloned repo with the detectors."""

import ast
import warnings
from pathlib import Path

from hotloop.survey.detect import Findings, argparse_defaults, scan_python, scan_text

TEXT_SUFFIXES = {".sh", ".yaml", ".yml", ".json", ".cfg", ".toml", ".ini", ".md"}
TEST_DIRS = {"test", "tests", "testing"}
MAX_FILE_BYTES = 1_000_000


def is_test(path: Path) -> bool:
    name = path.name
    return (
        bool(TEST_DIRS & set(path.parts[:-1]))
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def scan_repo(root: Path) -> Findings:
    trees = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if ".git" in rel.parts or is_test(rel) or path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)  # e.g. invalid escapes
                trees[rel] = ast.parse(path.read_bytes())
        except (SyntaxError, ValueError):
            continue  # Python 2 or broken files

    defaults = {}
    for tree in trees.values():
        defaults.update(argparse_defaults(tree))

    findings = Findings()
    for rel, tree in trees.items():
        findings.merge(scan_python(tree, str(rel), defaults))
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if (
            path.suffix in TEXT_SUFFIXES
            and ".git" not in rel.parts
            and path.is_file()
            and path.stat().st_size <= MAX_FILE_BYTES
        ):
            findings.merge(scan_text(path.read_text(errors="replace"), str(rel)))
    return findings
