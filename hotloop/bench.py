"""The planted test suite: our own fast scripts, each case breaking one thing on purpose.

The cases are listed in bench/cases.toml. They test the agent and let setups be
compared; they're not the headline results.
"""

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from hotloop.targets import Target, parse
from hotloop.workspace import WORK

BENCH = Path(__file__).parent.parent / "bench"


@dataclass(frozen=True)
class Case:
    name: str
    script: str
    mistake: str  # empty for a clean control
    edits: tuple[tuple[str, str], ...] = ()
    args: str = ""

    @property
    def control(self) -> bool:
        return not self.edits


def cases() -> list[Case]:
    config = tomllib.loads((BENCH / "cases.toml").read_text())
    return [
        Case(
            name=c["name"],
            script=c["script"],
            mistake=c["mistake"],
            edits=tuple((e["old"], e["new"]) for e in c.get("edit", ())),
            args=c.get("args", ""),
        )
        for c in config["case"]
    ]


def case(name: str) -> Case:
    for c in cases():
        if c.name == name:
            return c
    raise KeyError(f"no bench case named {name}")


def planted_source(case: Case) -> str:
    source = (BENCH / case.script / "train.py").read_text()
    for old, new in case.edits:
        matches = source.count(old)
        if matches != 1:
            raise ValueError(f"{case.name}: an edit matches {matches} times, not once")
        source = source.replace(old, new)
    return source


def target(case: Case) -> Target:
    config = tomllib.loads((BENCH / case.script / "target.toml").read_text())
    config["command"] = f"{config['command']} {case.args}".strip()
    return parse(f"bench-{case.name}", config)


def working_copy(case: Case, label: str) -> Path:
    """A git working copy of the planted script, so `git diff` shows only later changes."""
    dest = WORK / "work" / "bench" / case.name / label
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    (dest / "train.py").write_text(planted_source(case))
    for args in (
        ["init", "--quiet"],
        ["add", "train.py"],
        ["-c", "user.name=hotloop", "-c", "user.email=hotloop@localhost", "commit", "--quiet",
         "-m", f"planted: {case.mistake or 'clean control'}"],
    ):  # fmt: skip
        subprocess.run(["git", "-C", str(dest), *args], check=True)
    return dest
