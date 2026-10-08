"""The tools the model can call, as JSON schemas plus the code that runs them."""

import json

from agentictorch.agent.session import Session
from agentictorch.tools import files
from agentictorch.tools.files import ToolError

MAX_PROFILES = 2
MAX_CHECKS = 5
DOC_CHARS = 700
# Tools that run on Modal, where a call can fail for reasons outside the code (a lost
# container, a remote exception); such a failure shouldn't end a run that has already
# measured and kept changes.
GPU_TOOLS = {"profile", "quick_check", "run_benchmark"}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SCHEMAS = [
    _schema(
        "list_files",
        "List files in the repo matching a glob pattern.",
        {"pattern": {"type": "string", "description": "for example **/*.py"}},
        [],
    ),
    _schema(
        "read_file",
        "Read numbered lines of a file in the repo.",
        {
            "path": {"type": "string"},
            "start": {"type": "integer", "description": "first line, from 1"},
            "end": {"type": "integer", "description": "last line"},
        },
        ["path"],
    ),
    _schema(
        "search_code",
        "Find lines in the repo's Python files that match a regular expression.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    _schema(
        "apply_edit",
        "Replace one exact occurrence of `old` with `new` in a file. Include enough "
        "surrounding lines in `old` to match exactly once.",
        {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        ["path", "old", "new"],
    ),
    _schema(
        "search_docs",
        "Search PyTorch and NVIDIA performance docs and resolved PyTorch issues.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "profile",
        "Profile a short run of the best version so far: GPU busy share, data-loading "
        "wait, host-device syncs and top ops. Slower than a normal run; at most twice per run.",
        {},
        [],
    ),
    _schema(
        "quick_check",
        "Run your edits for a few steps on the GPU to catch crashes before run_benchmark. "
        "Uses no benchmark round and says nothing about speed. At most 5 per run.",
        {},
        [],
    ),
    _schema(
        "run_benchmark",
        "Measure your edits against the best version so far on the GPU. The change is "
        "kept only if it is faster beyond noise with the same loss; otherwise your "
        "edits are reverted.",
        {"description": {"type": "string", "description": "one line: what changed and why"}},
        ["description"],
    ),
    _schema(
        "finish",
        "Stop when no promising changes are left.",
        {"summary": {"type": "string", "description": "what was wrong and what helped"}},
        ["summary"],
    ),
]


class Toolbox:
    def __init__(self, session: Session, index=None, allowed: tuple[str, ...] | None = None):
        self.session = session
        self.allowed = allowed
        self.index = index  # the knowledge base, loaded lazily because the models are slow to load
        self.sources: dict[str, str] = {}  # url -> title, for every doc the model was shown
        self.profiles = 0  # the model's own profile calls; the baseline profile doesn't count
        self.checks = 0
        self.finished: str | None = None

    def call(self, name: str, arguments: str) -> str:
        if self.allowed is not None and name not in self.allowed:
            return f"error: {name} is not available"
        try:
            args = json.loads(arguments or "{}")
            return getattr(self, f"_{name}")(**args)
        except ToolError as e:
            return f"error: {e}"
        except (AttributeError, TypeError, json.JSONDecodeError) as e:
            return f"error: bad call to {name}: {e}"
        except Exception as e:
            if name not in GPU_TOOLS:
                raise
            return f"error: {name} failed: {type(e).__name__}: {e}"

    def _list_files(self, pattern: str = "**/*.py") -> str:
        return "\n".join(files.list_files(self.session.repo, pattern)[:200]) or "no files"

    def _read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        return files.read_file(self.session.repo, path, start, end)

    def _search_code(self, pattern: str) -> str:
        return "\n".join(files.search(self.session.repo, pattern)) or "no matches"

    def _apply_edit(self, path: str, old: str, new: str) -> str:
        return files.apply_edit(self.session.repo, path, old, new)

    def _search_docs(self, query: str) -> str:
        if self.index is None:
            from agentictorch.kb.search import Index

            self.index = Index()
        out = []
        for hit in self.index.search(query, k=4):
            c = hit.chunk
            self.sources[c.url] = f"{c.source}: {c.heading}"
            out.append(f"[{c.source}: {c.heading}]({c.url})\n{c.text[:DOC_CHARS]}")
        return "\n\n".join(out)

    def _profile(self) -> str:
        if self.profiles >= MAX_PROFILES:
            raise ToolError("already profiled twice in this run")
        self.profiles += 1
        return self.session.profile()

    def _quick_check(self) -> str:
        if self.checks >= MAX_CHECKS:
            raise ToolError(f"already ran {MAX_CHECKS} quick checks in this run")
        self.session.check_precision()
        self.checks += 1
        return self.session.check()

    def _run_benchmark(self, description: str) -> str:
        self.session.check_precision()
        attempt = self.session.benchmark(description)
        v = attempt.verdict
        outcome = "kept" if v.keep else "reverted"
        text = (
            f"{outcome}: {v.reason}. Best version now runs at "
            f"{(attempt.candidate if v.keep else attempt.baseline).mean_step_ms:.1f} ms/step. "
            f"{self.session.rounds_left} benchmark rounds left."
        )
        if not attempt.candidate.completed:
            text += f"\nlast lines of the failed run's log:\n{attempt.candidate.failure_log()}"
        return text

    def _finish(self, summary: str) -> str:
        self.finished = summary
        return "finished"
