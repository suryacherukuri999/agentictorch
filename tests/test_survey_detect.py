import ast
from textwrap import dedent

import pytest

from hotloop.survey.detect import argparse_defaults, scan_python, scan_text


def scan(source: str):
    tree = ast.parse(dedent(source))
    return scan_python(tree, "train.py", argparse_defaults(tree))


def test_counts_an_unconditional_sync_in_the_training_loop():
    found = scan("""
        for x, y in loader:
            loss = model(x, y)
            loss.backward()
            opt.step()
            total += loss.item()
    """)
    assert found.training_loops == 1
    assert found.loop_syncs == ["train.py:6: loss.item()"]


def test_ignores_syncs_under_an_if():
    found = scan("""
        for i, (x, y) in enumerate(loader):
            loss = model(x, y)
            loss.backward()
            if i % 100 == 0:
                print(loss.item())
            value = loss.item() if verbose else None
    """)
    assert found.loop_syncs == []


def test_counts_backward_outside_a_loop_as_training_code_without_a_loop():
    found = scan("""
        def optimisation_step(optimizer, loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)
    assert found.backward_calls == 1
    assert found.training_loops == 0


def test_uses_the_innermost_loop_that_calls_backward():
    found = scan("""
        for epoch in range(10):
            for x in loader:
                model(x).backward()
            print(loss.item())
    """)
    assert found.training_loops == 1
    assert found.loop_syncs == []


def test_counts_cuda_synchronize_and_empty_cache():
    found = scan("""
        import torch
        while True:
            loss.backward()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    """)
    assert len(found.loop_syncs) == 2


def test_ignores_syncs_in_functions_defined_inside_the_loop():
    found = scan("""
        for x in loader:
            loss.backward()
            def log():
                return loss.item()
    """)
    assert found.loop_syncs == []


@pytest.mark.parametrize(
    ("call", "flagged"),
    [
        ("DataLoader(ds, batch_size=8, shuffle=True)", "num_workers not set (defaults to 0)"),
        ("DataLoader(ds, shuffle=True, num_workers=0)", "num_workers=0"),
        ("torch.utils.data.DataLoader(ds, shuffle=True, num_workers=4)", None),
        ("DataLoader(ds, shuffle=False)", None),
        ("DataLoader(ds, shuffle=True, **kwargs)", None),
        (
            "DataLoader(ds, shuffle=True, num_workers=args.n_cpu)",
            "num_workers from --n_cpu, default 0",
        ),
        ("DataLoader(ds, shuffle=True, num_workers=args.workers)", None),
    ],
)
def test_flags_training_dataloaders_on_the_main_thread(call, flagged):
    found = scan(f"""
        parser.add_argument("--n-cpu", type=int, default=0)
        parser.add_argument("--workers", type=int, default=4)
        loader = {call}
    """)
    assert found.main_thread_loaders == ([f"train.py:4: {flagged}"] if flagged else [])


@pytest.mark.parametrize(
    ("source", "amp"),
    [
        ("with torch.cuda.amp.autocast():\n    pass", True),
        ("scaler = GradScaler()", True),
        ("model = model.half()", True),
        ("import torch\nx = torch.zeros(3, dtype=torch.bfloat16)", True),
        ("trainer = pl.Trainer(precision=16)", True),
        ("trainer = pl.Trainer(precision=32)", False),
        ("args = TrainingArguments(fp16=True)", True),
        ("args = TrainingArguments(fp16=False)", False),
        ("np.set_printoptions(precision=3)", False),
        ("import torch\nsame = x.dtype == torch.float16", False),
        ("import torch\nhalf = x.dtype in {torch.float16, torch.bfloat16}", False),
        ("import torch\ndtype = torch.float16 if use_fp16 else torch.float32", True),
        ("import torch as th\nmodel = model.to(th.bfloat16)", True),
        ("import numpy as np\nx = x.astype(np.float16)", False),
        ('fabric = L.Fabric(precision="bf16-mixed")', True),
        ("fp16 = dict(loss_scale=512.0)", True),
        ("fp16 = False", False),
        ("train.amp.enabled = True", True),
        ("self.use_amp = use_amp  # adversarial motion priors in RL code", False),
        ("@auto_fp16()\ndef forward(self, x):\n    return x", True),
    ],
)
def test_detects_mixed_precision_in_python(source, amp):
    assert bool(scan(source).amp) is amp


@pytest.mark.parametrize(
    ("text", "amp"),
    [
        ("python train.py --fp16 --epochs 10", True),
        ("accelerate launch --mixed_precision=bf16 train.py", True),
        ("accelerate launch --mixed_precision no train.py", False),
        ("training:\n  fp16: true\n", True),
        ("training:\n  fp16: false\n", False),
        ('{"fp16": {"enabled": true}}', True),
        ("precision: 16\n", True),
        ("precision: 32\n", False),
        ("Use --amp to train faster.", True),
        ("--ample-mode", False),
    ],
)
def test_detects_mixed_precision_flags_in_scripts_and_configs(text, amp):
    assert bool(scan_text(text, "run.sh").amp) is amp


def test_detects_torch_compile_and_trainer_frameworks():
    found = scan("""
        import pytorch_lightning as pl
        model = torch.compile(model)
    """)
    assert found.uses_compile
    assert found.trainer_framework
    assert not scan("from transformers import AutoModel").trainer_framework
    assert scan("from transformers import Trainer").trainer_framework
