"""One agent run on one target: the working copy, its measurements and its benchmark rounds.

The working copy is a git repo. HEAD holds every kept change; the model edits
the working tree. Each benchmark runs HEAD and the working tree as a pair, and
the keep/revert rule, not the model, decides what happens to the change.
"""

import contextlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentictorch import workspace
from agentictorch.agent.keep import Verdict, decide
from agentictorch.harness import local_runner, modal_runner
from agentictorch.harness.runner import Job
from agentictorch.measurement import Measurement
from agentictorch.targets import Target
from agentictorch.tools.files import ToolError

# A quick check only has to get through start-up and a few optimizer steps.
CHECK_WARMUP = 2
CHECK_STEPS = 3
NO_CHANGE = "nothing has changed since the last kept version; make an edit with apply_edit first"
# Without loss scaling, small float16 gradients underflow over a long run, which a short
# run can't show. CUDA autocast without a dtype means float16.
FLOAT16 = re.compile(r"(?<!b)float16|torch\.half\b|\.half\(\)")


@dataclass(frozen=True)
class Attempt:
    description: str
    diff: str
    verdict: Verdict
    baseline: Measurement
    candidate: Measurement


class Session:
    def __init__(
        self,
        target: Target,
        repo: Path,
        gpu: str = "L4",
        rounds: int = 5,
        label: str = "agent",
        backend: str = "modal",
    ):
        self.target = target
        self.repo = repo
        self.label = label  # names the raw runs saved for each pair
        self.gpu = gpu
        self.runner = local_runner if backend == "local" else modal_runner
        self.rounds = rounds
        self.original = self._git("rev-parse", "HEAD").strip()
        self.attempts: list[Attempt] = []
        self.baseline: Measurement | None = None
        self.profiles: list[str] = []

    @property
    def rounds_left(self) -> int:
        return self.rounds - len(self.attempts)

    @property
    def kept(self) -> list[Attempt]:
        return [a for a in self.attempts if a.verdict.keep]

    def measure_baseline(self) -> Measurement:
        with self._exported("HEAD") as best:
            self.baseline = Measurement(self.runner.measure(self.target, best, gpu=self.gpu))
        return self.baseline

    def profile(self) -> str:
        """Profile the best version so far in a short run."""
        with self._exported("HEAD") as best:
            [run] = self.runner.measure(
                self.target, best, gpu=self.gpu, seeds=(0,), steps=10, profile=True
            )
        text = run.get("profile_text") or f"profiling failed: {Measurement([run]).failure()}"
        self.profiles.append(text)
        return text

    def diff(self) -> str:
        return self._git("diff")

    def check(self) -> str:
        """Run the working tree for a few steps to catch crashes. Uses no benchmark round."""
        if not self.diff():
            raise ToolError(NO_CHANGE)
        self._check_syntax()
        [run] = self.runner.measure(
            self.target,
            self.repo,
            gpu=self.gpu,
            seeds=(0,),
            steps=CHECK_STEPS,
            warmup=CHECK_WARMUP,
        )
        result = Measurement([run])
        if result.completed:
            steps = CHECK_WARMUP + CHECK_STEPS
            return f"ran {steps} steps without errors (too short to measure speed)"
        return f"failed: {result.failure()}\nlast lines of the log:\n{result.failure_log()}"

    def benchmark(self, description: str) -> Attempt:
        """Measure the working tree against the best version, then keep or revert it."""
        if self.rounds_left <= 0:
            raise ToolError("no benchmark rounds left; call finish")
        diff = self.diff()
        if not diff:
            raise ToolError(NO_CHANGE)
        self._check_syntax()
        with self._exported("HEAD") as best:
            base, cand = self.runner.ab(self.target, Job(best), Job(self.repo), gpu=self.gpu)
        round_label = f"{self.label}-round{len(self.attempts) + 1}"
        workspace.save_ab(self.target, f"{self.label}-best", round_label, base, cand, "", "")
        base, cand = Measurement(base), Measurement(cand)
        verdict = decide(base, cand)
        if verdict.keep:
            self._git("add", "--all")
            self._git(
                "-c", "user.name=agentictorch", "-c", "user.email=agentictorch@localhost",
                "commit", "--quiet", "-m", description,
            )  # fmt: skip
        else:
            self._git("checkout", "--", ".")
        attempt = Attempt(description, diff, verdict, base, cand)
        self.attempts.append(attempt)
        return attempt

    def final(self) -> Attempt:
        """The original code against everything kept, as one last pair."""
        with self._exported(self.original) as original, self._exported("HEAD") as best:
            base, cand = self.runner.ab(self.target, Job(original), Job(best), gpu=self.gpu)
        workspace.save_ab(self.target, "original", f"{self.label}-final", base, cand, "", "")
        base, cand = Measurement(base), Measurement(cand)
        diff = self._git("diff", self.original, "HEAD")
        return Attempt("all kept changes", diff, decide(base, cand), base, cand)

    def check_precision(self) -> None:
        """Refuse a change that adds float16 when the repo has no GradScaler."""
        added = [
            line for line in self.diff().splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]  # fmt: skip
        if not any(_uses_float16(line) for line in added):
            return
        grep = ["git", "-C", str(self.repo), "grep", "--quiet", "--untracked", "GradScaler"]
        if subprocess.run(grep).returncode != 0:
            raise ToolError(
                "float16 needs a GradScaler: without loss scaling, small gradients underflow to "
                "zero over a long run, which a short benchmark can't show. Use bfloat16, or add "
                "torch.amp.GradScaler around backward and the optimizer step."
            )

    def _check_syntax(self) -> None:
        """Catch Python syntax errors locally instead of wasting a GPU round on them."""
        for path in self._git("diff", "--name-only").split():
            if path.endswith(".py"):
                try:
                    compile((self.repo / path).read_bytes(), path, "exec")
                except SyntaxError as e:
                    raise ToolError(f"syntax error in {path} line {e.lineno}: {e.msg}") from None

    @contextlib.contextmanager
    def _exported(self, rev: str):
        """A temporary copy of the working copy at `rev`, for upload."""
        with tempfile.TemporaryDirectory(prefix="agentictorch-export-") as tmp:
            copy = Path(tmp) / "repo"
            shutil.copytree(self.repo, copy, symlinks=True)
            subprocess.run(["git", "-C", str(copy), "reset", "--quiet", "--hard", rev], check=True)
            yield copy

    def _git(self, *args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True
        )
        return out.stdout


def _uses_float16(line: str) -> bool:
    if FLOAT16.search(line):
        return True
    return "autocast(" in line and "bfloat16" not in line and "bf16" not in line
