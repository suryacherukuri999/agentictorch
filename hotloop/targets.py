"""Target repos: one `targets/<name>/target.toml` per repo, plus an optional compat.patch."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TARGETS_DIR = Path(__file__).parent.parent / "targets"


@dataclass(frozen=True)
class Target:
    name: str
    repo: str
    commit: str
    command: str
    steps: int = 50
    warmup: int = 10
    timeout: int = 1800
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    # The dataset lives in the Modal Volume under `dataset`; `data_path` is
    # where the repo expects it; `download` fills the Volume folder once.
    dataset: str | None = None
    data_path: str | None = None
    download: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return TARGETS_DIR / self.name

    @property
    def compat_patch(self) -> Path | None:
        patch = self.dir / "compat.patch"
        return patch if patch.exists() else None


def load(name: str) -> Target:
    config = tomllib.loads((TARGETS_DIR / name / "target.toml").read_text())
    image = config.pop("image", {})
    data = config.pop("data", {})
    return Target(
        name=name,
        pip=tuple(image.get("pip", ())),
        apt=tuple(image.get("apt", ())),
        dataset=data.get("dataset"),
        data_path=data.get("path"),
        download=data.get("download"),
        **config,
    )


def names() -> list[str]:
    return sorted(p.parent.name for p in TARGETS_DIR.glob("*/target.toml"))
