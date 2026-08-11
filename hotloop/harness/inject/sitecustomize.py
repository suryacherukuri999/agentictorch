"""Measurement harness that Python loads at startup from PYTHONPATH.

It does nothing unless HOTLOOP_OUT is set. Then, once torch is imported, it:

- seeds Python, numpy and torch with HOTLOOP_SEED, and shifts the repo's own
  seed calls by the same amount, so runs with different seeds differ but each
  seed repeats its data order;
- counts steps of the first optimizer that steps (the clock), so a GAN that
  steps a discriminator and a generator counts one step per iteration;
- keeps each step's loss as a detached tensor and reads them all at the end,
  so it never adds a host-device sync to a step;
- syncs CUDA only at the start and end of the measured window;
- after HOTLOOP_WARMUP + HOTLOOP_STEPS steps, writes JSON to HOTLOOP_OUT and exits.

This file runs inside the target's environment, so it uses only the standard
library, torch and (if present) numpy.
"""

import atexit
import contextlib
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


class _State:
    def __init__(self) -> None:
        self.out = os.environ["HOTLOOP_OUT"]
        self.warmup = max(1, int(os.environ.get("HOTLOOP_WARMUP", "10")))
        self.steps = int(os.environ.get("HOTLOOP_STEPS", "50"))
        self.seed = int(os.environ.get("HOTLOOP_SEED", "0"))
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
                print("hotloop: harness setup failed", file=sys.stderr)
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
        if self.numel() == 1 and not getattr(self, "_hotloop_scaled", False):
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
            scaled._hotloop_scaled = True
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
    elif _state.step == _state.warmup + _state.steps:
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


def _finish():
    _state.finished = True
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


if os.environ.get("HOTLOOP_OUT"):
    _state = _State()
    random.seed(_state.seed)
    if _state.seed:
        random.seed = _shift_seed(random.seed)
    _excepthook = sys.excepthook
    sys.excepthook = _record_error
    sys.meta_path.insert(0, _AfterImport("numpy", _on_numpy))
    sys.meta_path.insert(0, _AfterImport("torch", _on_torch))
