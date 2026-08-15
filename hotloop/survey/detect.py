"""Static detectors for likely slow training code. Nothing here runs repo code.

- A repo has training code if it calls `.backward()` or uses a trainer
  framework such as Lightning.
- A training loop is the innermost loop that calls `.backward()`.
- A per-step sync is a call inside a training loop, not under an `if`, to
  `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `cuda.synchronize()` or
  `cuda.empty_cache()`.
- A training DataLoader has `shuffle=True`. It runs on the main thread when
  `num_workers` is missing, is 0, or comes from an argparse option whose
  default is 0. DataLoaders that take `**kwargs` are skipped.
- Mixed precision counts if any Python file uses it or can switch it on (an
  fp16 dtype that isn't only compared against, autocast, GradScaler, a
  Lightning precision, an fp16 switch in a Python config), or if a script,
  config or README turns it on with a flag.
"""

import ast
import re
from dataclasses import dataclass, field

SYNC_METHODS = {"item", "cpu", "numpy", "tolist"}
SYNC_CUDA = {"synchronize", "empty_cache"}
LOOPS = (ast.For, ast.AsyncFor, ast.While)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
AMP_NAMES = {"autocast", "GradScaler", "auto_fp16", "AMPTrainer"}
TRAINER_MODULES = {
    "pytorch_lightning",
    "lightning",
    "mmcv",
    "mmengine",
    "detectron2",
    "ignite",
    "fastai",
    "deepspeed",
    "stable_baselines3",
}
TRAINER_CLASSES = {"Trainer", "Seq2SeqTrainer", "SFTTrainer"}
HALF_DTYPES = {"float16", "half", "bfloat16"}
LIGHTNING_HALF = {"16-mixed", "16-true", "bf16", "bf16-mixed", "bf16-true"}
LIGHTNING_RUNNERS = {"Trainer", "Fabric"}
OFF = (False, None, 0, 32, "32", "no", "fp32")
# Mixed-precision switches set in Python config files, such as mmcv's
# `fp16 = dict(loss_scale=512.)` or detectron2's `train.amp.enabled = True`.
# A bare `amp` or `use_amp` is left out: in RL code it often means
# Adversarial Motion Priors.
CONFIG_SWITCH = re.compile(r"(^|\.)(fp16|bf16|use_fp16|mixed_precision|amp\.enabled)$")

# Flags that turn on mixed precision in shell scripts, configs and READMEs.
AMP_FLAG = re.compile(
    r"--(fp16|bf16|amp|use[_-]amp)\b"
    r"|--mixed[_-]precision\b(?![ =]+['\"]?no\b)"
    r"|--precision[ =]+['\"]?(16|bf16)\b"
    r"|^\s*['\"]?(fp16|bf16|amp|use_amp)['\"]?\s*[:=]\s*(true|yes|1)\b"
    r"|['\"](fp16|bf16)['\"]\s*:\s*\{\s*['\"]enabled['\"]\s*:\s*true"
    r"|mixed_precision['\"]?\s*[:=]\s*['\"]?(fp16|bf16)\b"
    r"|^\s*precision\s*:\s*['\"]?(16|bf16)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Findings:
    backward_calls: int = 0
    training_loops: int = 0
    loop_syncs: list[str] = field(default_factory=list)
    main_thread_loaders: list[str] = field(default_factory=list)
    amp: list[str] = field(default_factory=list)
    uses_compile: bool = False
    trainer_framework: bool = False

    def merge(self, other: "Findings") -> None:
        self.backward_calls += other.backward_calls
        self.training_loops += other.training_loops
        self.loop_syncs += other.loop_syncs
        self.main_thread_loaders += other.main_thread_loaders
        self.amp += other.amp
        self.uses_compile |= other.uses_compile
        self.trainer_framework |= other.trainer_framework


def argparse_defaults(tree: ast.AST) -> dict[str, object]:
    """Map each argparse destination to its literal default (None if not literal)."""
    defaults = {}
    for node in ast.walk(tree):
        if not (_is_method_call(node, "add_argument") and node.args):
            continue
        options = [a.value for a in node.args if isinstance(a, ast.Constant)]
        longs = [o for o in options if isinstance(o, str) and o.startswith("--")]
        dest = _keyword(node, "dest")
        if isinstance(dest, ast.Constant):
            name = dest.value
        elif longs:
            name = longs[0][2:].replace("-", "_")
        else:
            continue
        default = _keyword(node, "default")
        defaults[name] = _literal(default) if default is not None else None
    return defaults


def scan_python(tree: ast.AST, path: str, defaults: dict[str, object]) -> Findings:
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    torch_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "torch"
    }
    found = Findings()

    backwards = [node for node in ast.walk(tree) if _is_backward(node)]
    found.backward_calls = len(backwards)
    loops = {_enclosing_loop(node, parents) for node in backwards}
    loops.discard(None)
    found.training_loops = len(loops)
    for loop in loops:
        for node in _walk_body(loop):
            if _is_sync(node) and not _conditional(node, loop, parents):
                found.loop_syncs.append(f"{path}:{node.lineno}: {ast.unparse(node.func)}()")

    for node in ast.walk(tree):
        if _is_training_loader(node):
            reason = _main_thread_reason(node, defaults)
            if reason:
                found.main_thread_loaders.append(f"{path}:{node.lineno}: {reason}")
        if _uses_amp(node, parents, torch_names):
            found.amp.append(f"{path}:{node.lineno}")
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.compile":
            found.uses_compile = True
        if _uses_trainer(node):
            found.trainer_framework = True
    return found


def scan_text(text: str, path: str) -> Findings:
    """Mixed-precision flags in shell scripts, configs and READMEs."""
    found = Findings()
    for match in AMP_FLAG.finditer(text):
        found.amp.append(f"{path}:{text.count(chr(10), 0, match.start()) + 1}")
    return found


def _is_method_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def _is_backward(node: ast.AST) -> bool:
    return _is_method_call(node, "backward")


def _is_sync(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    func = node.func
    if func.attr in SYNC_METHODS:
        return not node.args
    if func.attr in SYNC_CUDA:
        owner = func.value
        return (isinstance(owner, ast.Attribute) and owner.attr == "cuda") or (
            isinstance(owner, ast.Name) and owner.id == "cuda"
        )
    return False


def _enclosing_loop(node: ast.AST, parents: dict) -> ast.AST | None:
    while node in parents:
        node = parents[node]
        if isinstance(node, LOOPS):
            return node
        if isinstance(node, SCOPES):
            return None
    return None


def _walk_body(loop: ast.AST):
    """Walk the statements a loop runs each iteration, without entering
    nested functions or classes."""
    stack = list(loop.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _conditional(node: ast.AST, loop: ast.AST, parents: dict) -> bool:
    while node is not loop:
        node = parents[node]
        if isinstance(node, ast.If | ast.IfExp):
            return True
    return False


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _literal(node: ast.expr):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def _is_training_loader(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if _call_name(node) != "DataLoader":
        return False
    shuffle = _keyword(node, "shuffle")
    if shuffle is None and len(node.args) >= 3:
        shuffle = node.args[2]
    return isinstance(shuffle, ast.Constant) and shuffle.value is True


def _main_thread_reason(loader: ast.Call, defaults: dict[str, object]) -> str | None:
    if any(kw.arg is None for kw in loader.keywords):
        return None  # **kwargs may set num_workers
    workers = _keyword(loader, "num_workers")
    if workers is None:
        return None if len(loader.args) >= 6 else "num_workers not set (defaults to 0)"
    if isinstance(workers, ast.Constant):
        return "num_workers=0" if workers.value == 0 else None
    if isinstance(workers, ast.Attribute) and defaults.get(workers.attr) == 0:
        return f"num_workers from --{workers.attr}, default 0"
    return None


def _uses_amp(node: ast.AST, parents: dict, torch_names: set[str]) -> bool:
    if isinstance(node, ast.Name | ast.Attribute):
        name = node.id if isinstance(node, ast.Name) else node.attr
        text = ast.unparse(node)
        if name in AMP_NAMES or text == "amp.initialize":
            return True
        half = (
            isinstance(node, ast.Attribute)
            and node.attr in HALF_DTYPES
            and isinstance(node.value, ast.Name)
            and node.value.id in torch_names
        )
        return half and not _in_comparison(node, parents)
    if isinstance(node, ast.Assign):
        return not _off(node.value) and any(
            CONFIG_SWITCH.search(ast.unparse(t)) for t in node.targets
        )
    if not isinstance(node, ast.Call):
        return False
    if _is_method_call(node, "half") and not node.args:
        return True
    for kw in node.keywords:
        if kw.arg in {"fp16", "bf16", "mixed_precision"} and not _off(kw.value):
            return True
        if kw.arg == "precision" and _lightning_half(node, kw.value):
            return True
    return False


def _off(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value in OFF


def _lightning_half(call: ast.Call, value: ast.expr) -> bool:
    runner = _call_name(call) in LIGHTNING_RUNNERS
    if not isinstance(value, ast.Constant):
        return runner
    if value.value in LIGHTNING_HALF:
        return True
    return runner and value.value in (16, "16")


def _in_comparison(node: ast.AST, parents: dict) -> bool:
    """True for dtype checks like `x.dtype == torch.float16`."""
    while node in parents and not isinstance(node, ast.stmt):
        node = parents[node]
        if isinstance(node, ast.Compare):
            return True
    return False


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _uses_trainer(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] in TRAINER_MODULES for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".")[0]
        if root in TRAINER_MODULES:
            return True
        return root == "transformers" and any(a.name in TRAINER_CLASSES for a in node.names)
    return False
