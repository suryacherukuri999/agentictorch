"""Measurement harness that Python loads at startup from PYTHONPATH.

It does nothing unless AGENTICTORCH_OUT is set. Then, once torch is imported, it:

- seeds Python, numpy and torch with AGENTICTORCH_SEED, and shifts the repo's own
  seed calls by the same amount, so runs with different seeds differ but each
  seed repeats its data order;
- counts steps of the first optimizer that steps (the clock), so a GAN that
  steps a discriminator and a generator counts one step per iteration;
- keeps each step's loss as a detached tensor and reads them all at the end,
  so it never adds a host-device sync to a step;
- syncs CUDA only at the start and end of the measured window;
- after AGENTICTORCH_WARMUP + AGENTICTORCH_STEPS steps, writes JSON to AGENTICTORCH_OUT and exits.

With AGENTICTORCH_PROFILE set to a directory, it also runs the PyTorch profiler over
the measured steps and saves the trace and per-op table there, with the repo
line behind each op and each host-device sync (AGENTICTORCH_REPO is the repo root).
The profiler slows training, so step times from those runs are only for diagnosis.

This file runs inside the target's environment, so it uses only the standard
library, torch and (if present) numpy.
"""

import atexit
import contextlib
import functools
import importlib.abc
import importlib.util
import itertools
import json
import multiprocessing
import operator
import os
import random
import sys
import time
import traceback
import warnings

SYNC_WARNING = "called a synchronizing CUDA operation"
LINE_TAG = "agentictorch:line:"


class _State:
    def __init__(self) -> None:
        self.out = os.environ["AGENTICTORCH_OUT"]
        self.warmup = max(1, int(os.environ.get("AGENTICTORCH_WARMUP", "10")))
        self.steps = int(os.environ.get("AGENTICTORCH_STEPS", "50"))
        self.seed = int(os.environ.get("AGENTICTORCH_SEED", "0"))
        self.profile_dir = os.environ.get("AGENTICTORCH_PROFILE")
        self.repo = os.path.realpath(os.environ.get("AGENTICTORCH_REPO") or os.getcwd())
        self.profiler = None
        self.lines = None  # the torch function mode that tags ops with repo lines
        self.syncs = {}  # repo line -> host-device syncs seen there while profiling
        self.show_warning = None  # warnings.showwarning while syncs are being watched
        self.pid = os.getpid()
        self.clock = None
        self.step = 0
        self.other_steps = 0
        self.stamps = []  # perf_counter at each step of the measured window
        self.data_wait = 0.0
        self.pending = []  # losses since the last clock step
        self.losses = []  # one list of loss tensors per step
        self.error = None
        self.oom = False
        self.finished = False

    @property
    def in_window(self) -> bool:
        return bool(self.stamps) and self.step < self.warmup + self.steps


_state = None


class _AfterImport(importlib.abc.MetaPathFinder):
    """Calls `callback(module)` right after the named top-level module is imported."""

    def __init__(self, name, callback):
        self.name = name
        self.callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_then_callback(module):
            exec_module(module)
            try:
                self.callback(module)
            except Exception:
                print("agentictorch: harness setup failed", file=sys.stderr)
                traceback.print_exc()

        spec.loader.exec_module = exec_then_callback
        return spec


def _shift_seed(seed_fn, modulus=None):
    def seeded(*args, **kwargs):
        if args:
            args = (_shifted(args[0], modulus), *args[1:])
        for name in ("seed", "a"):  # random.seed calls its argument `a`
            if name in kwargs:
                kwargs[name] = _shifted(kwargs[name], modulus)
        return seed_fn(*args, **kwargs)

    return seeded


def _shifted(seed, modulus):
    try:
        seed = operator.index(seed) + _state.seed
    except TypeError:
        return seed  # None or a sequence: leave it alone
    return seed % modulus if modulus else seed


def _on_numpy(np):
    np.random.seed(_state.seed)
    if _state.seed:
        np.random.seed = _shift_seed(np.random.seed, modulus=2**32)


def _on_torch(torch):
    torch.manual_seed(_state.seed)
    if _state.seed:
        torch.manual_seed = torch.random.manual_seed = _shift_seed(torch.manual_seed)
        torch.cuda.manual_seed = _shift_seed(torch.cuda.manual_seed)
        torch.cuda.manual_seed_all = _shift_seed(torch.cuda.manual_seed_all)

    from torch.optim.optimizer import register_optimizer_step_post_hook

    register_optimizer_step_post_hook(_on_step)

    backward = torch.Tensor.backward

    def recording_backward(self, *args, **kwargs):
        if self.numel() == 1 and not getattr(self, "_agentictorch_scaled", False):
            _state.pending.append(self.detach().clone())
        return backward(self, *args, **kwargs)

    torch.Tensor.backward = recording_backward

    # With a GradScaler, record the loss before scaling and skip the scaled
    # tensor when its backward() runs.
    scale = torch.amp.GradScaler.scale

    def recording_scale(self, outputs):
        scaled = scale(self, outputs)
        if isinstance(outputs, torch.Tensor) and outputs.numel() == 1:
            _state.pending.append(outputs.detach().clone())
            scaled._agentictorch_scaled = True
        return scaled

    torch.amp.GradScaler.scale = recording_scale

    from torch.utils.data.dataloader import _BaseDataLoaderIter

    next_batch = _BaseDataLoaderIter.__next__

    def timed_next(self):
        if not _state.in_window:
            return next_batch(self)
        start = time.perf_counter()
        try:
            return next_batch(self)
        finally:
            _state.data_wait += time.perf_counter() - start

    _BaseDataLoaderIter.__next__ = timed_next
    atexit.register(_write_at_exit)


def _clock_key(optimizer):
    # Keyed by the first parameter, so an optimizer that the repo rebuilds
    # every epoch is still the same clock.
    return id(optimizer.param_groups[0]["params"][0])


def _on_step(optimizer, args, kwargs):
    key = _clock_key(optimizer)
    if _state.clock is None:
        _state.clock = key
    if key != _state.clock:
        _state.other_steps += 1
        return
    if _state.finished:
        return

    _state.losses.append(_state.pending)
    _state.pending = []
    _state.step += 1
    if _state.step == _state.warmup:
        _sync()
        _state.stamps.append(time.perf_counter())
        if _state.profile_dir:
            _start_profiler()
    elif _state.step == _state.warmup + _state.steps:
        _stop_watching_syncs()  # our own sync below is not the repo's
        _sync()
        _state.stamps.append(time.perf_counter())
        _finish()
    elif _state.step > _state.warmup:
        _state.stamps.append(time.perf_counter())


def _cuda_in_use():
    torch = sys.modules.get("torch")
    return torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized()


def _sync():
    if _cuda_in_use():
        sys.modules["torch"].cuda.synchronize()


def _results(completed):
    stamps = _state.stamps
    measured = max(len(stamps) - 1, 0)
    step_times = [(b - a) * 1000 for a, b in itertools.pairwise(stamps)]
    result = {
        "completed": completed,
        "warmup": _state.warmup,
        "steps": measured,
        "step_ms": sum(step_times) / measured if measured else None,
        "step_ms_each": step_times,
        "data_wait_ms": _state.data_wait * 1000 / measured if measured else None,
        "other_optimizer_steps": _state.other_steps,
        "losses": _read_losses(),
        "seed": _state.seed,
        "error": _state.error,
        "oom": _state.oom,
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch"] = torch.__version__
    if _cuda_in_use():
        result["device"] = torch.cuda.get_device_name()
        result["peak_mem_allocated"] = torch.cuda.max_memory_allocated()
        result["peak_mem_reserved"] = torch.cuda.max_memory_reserved()
    else:
        result["device"] = "cpu"
    return result


def _read_losses():
    try:
        return [sum(map(float, step)) / len(step) if step else None for step in _state.losses]
    except Exception:  # e.g. the CUDA context is broken after a device-side error
        return None


def _write(result):
    tmp = _state.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, _state.out)


def _start_profiler():
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if _cuda_in_use():
        activities.append(ProfilerActivity.CUDA)
    _state.profiler = profile(activities=activities)
    _state.profiler.start()
    if not _compiled():
        _state.lines = _repo_lines_mode()
        _state.lines.__enter__()
    if _cuda_in_use():
        _watch_syncs()


def _compiled():
    """True if the repo already uses torch.compile; the line-tagging mode is skipped
    then, since compiled code shouldn't be traced through it."""
    try:
        from torch._dynamo.utils import counters

        return counters["stats"]["unique_graphs"] > 0
    except Exception:
        return False


def _repo_lines_mode():
    """A torch function mode that runs each torch call made from repo code inside
    a profiler range named after the calling line, so every op in the trace can
    be traced back to the exact line.

    Stacks from with_stack=True can't do this: on PyTorch 2.14 they name each
    frame by its function's def line, or by the line it was on when profiling
    started, not by the line that ran the op.
    """
    import torch

    class RepoLines(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            where = _calling_line(sys._getframe(1))
            if where is None:
                return func(*args, **(kwargs or {}))
            with torch.profiler.record_function(LINE_TAG + where):
                return func(*args, **(kwargs or {}))

    return RepoLines()


def _calling_line(frame):
    while frame is not None:
        where = _in_repo(frame.f_code.co_filename, frame.f_lineno)
        if where:
            return where
        frame = frame.f_back
    return None


def _watch_syncs():
    """Count each host-device sync under the repo line that caused it.

    CUDA's sync debug mode warns on every synchronizing call, which also
    catches implicit syncs such as float(loss) or `if loss < best:`.
    """
    previous = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if SYNC_WARNING not in str(message):
            return previous(message, category, filename, lineno, file, line)
        where = _calling_line(sys._getframe(1))
        _state.syncs[where] = _state.syncs.get(where, 0) + 1

    _state.show_warning = previous
    warnings.showwarning = show
    warnings.filterwarnings("always", message=f".*{SYNC_WARNING}")
    sys.modules["torch"].cuda.set_sync_debug_mode("warn")


def _stop_watching_syncs():
    if _state.show_warning is None:
        return
    sys.modules["torch"].cuda.set_sync_debug_mode("default")
    warnings.showwarning = _state.show_warning
    _state.show_warning = None


def _in_repo(path, line):
    """`path:line` relative to the repo root if the file is inside the repo, else None."""
    full = _resolve(path)
    if full and full.startswith(_state.repo + os.sep):
        return f"{os.path.relpath(full, _state.repo)}:{line}"
    return None


@functools.cache
def _resolve(path):
    # A script run as `python train.py` has the relative path train.py.
    candidates = [path] if os.path.isabs(path) else [os.path.join(b or ".", path) for b in sys.path]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def _op_lines(events):
    """The repo line behind each profiled event: its enclosing tagged range, or, for
    backward ops, the forward op that created them (matched by autograd's
    sequence number, since backward ops run on another thread)."""

    def tagged(e):
        while e is not None:
            if e.name.startswith(LINE_TAG):
                return e.name[len(LINE_TAG) :]
            e = e.cpu_parent
        return None

    forward = {
        (e.thread, e.sequence_nr): tagged(e)
        for e in events
        if e.sequence_nr >= 0 and not e.fwd_thread
    }
    lines = {}
    for e in events:
        node, where, backward = e, None, False
        while node is not None:
            if node.fwd_thread and node.sequence_nr >= 0:
                where, backward = forward.get((node.fwd_thread, node.sequence_nr)), True
                break
            if node.name.startswith(LINE_TAG):
                where = node.name[len(LINE_TAG) :]
                break
            node = node.cpu_parent
        lines[e.id] = (where, backward)
    # Ranges that PyTorch opens itself (an optimizer step, a DataLoader fetch) sit
    # outside any tagged call; give them the line of the first tagged op inside.
    for e in events:
        if lines[e.id][0] is None:
            lines[e.id] = (_first_tagged_child(e, lines), False)
    return lines


def _first_tagged_child(event, lines):
    pending = list(event.cpu_children)
    while pending:
        child = pending.pop(0)
        where, backward = lines.get(child.id, (None, False))
        if where and not backward:
            return where
        pending.extend(child.cpu_children)
    return None


def _save_profile():
    from torch.autograd import DeviceType

    _stop_watching_syncs()
    if _state.lines is not None:
        _state.lines.__exit__(None, None, None)
    prof = _state.profiler
    prof.stop()
    os.makedirs(_state.profile_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(_state.profile_dir, "trace.json"))
    # Kernels also appear as their own events, and our line tags as ranges; keep
    # only the ops that launch kernels, so device time isn't counted twice.
    events = prof.events()
    lines = _op_lines(events)
    ops = {}
    for e in events:
        if e.device_type != DeviceType.CPU or e.name.startswith(LINE_TAG):
            continue
        where, backward = lines[e.id]
        key = (e.name, where, backward)
        op = ops.setdefault(
            key,
            {"name": e.name, "where": where, "backward": backward, "count": 0,
             "self_cpu_us": 0.0, "self_device_us": 0.0},
        )  # fmt: skip
        op["count"] += 1
        op["self_cpu_us"] += e.self_cpu_time_total
        op["self_device_us"] += e.self_device_time_total
    syncs = [{"where": where, "count": n} for where, n in _state.syncs.items()]
    for name, data in (("ops.json", list(ops.values())), ("syncs.json", syncs)):
        with open(os.path.join(_state.profile_dir, name), "w") as f:
            json.dump(data, f)


def _finish():
    _state.finished = True
    if _state.profiler:
        _save_profile()  # first, so reading the losses doesn't show up as syncs
    _write(_results(completed=True))
    for child in multiprocessing.active_children():  # DataLoader workers
        child.terminate()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _write_at_exit():
    # The run ended before the last measured step: it crashed or ran out of data.
    # Child processes (DataLoader workers, a torchrun launcher's trainer) also
    # load this file, so only write when this process stepped or is the main one.
    # A forked child inherits the parent's state, so skip those too.
    if _state.finished or os.getpid() != _state.pid:
        return
    if _state.step or (_state.error and multiprocessing.parent_process() is None):
        with contextlib.suppress(Exception):
            _sync()
        _write(_results(completed=False))


def _record_error(exc_type, exc, tb):
    _state.error = "".join(traceback.format_exception_only(exc_type, exc)).strip()[-2000:]
    _state.oom = exc_type.__name__ == "OutOfMemoryError" or "out of memory" in str(exc)
    _excepthook(exc_type, exc, tb)


if os.environ.get("AGENTICTORCH_OUT"):
    _state = _State()
    random.seed(_state.seed)
    if _state.seed:
        random.seed = _shift_seed(random.seed)
    _excepthook = sys.excepthook
    sys.excepthook = _record_error
    sys.meta_path.insert(0, _AfterImport("numpy", _on_numpy))
    sys.meta_path.insert(0, _AfterImport("torch", _on_torch))
