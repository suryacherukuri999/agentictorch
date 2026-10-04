import pytest

from agentictorch.profile.summary import render, summarize


def event(cat, name, ts, dur):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


TRACE = {
    "traceEvents": [
        event("cpu_op", "aten::conv2d", 0, 100),
        event(
            "user_annotation", "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", 0, 40
        ),
        event("kernel", "conv_kernel", 10, 30),
        event("kernel", "add_kernel", 30, 20),  # overlaps the first kernel by 10
        event("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", 80, 5),
        event("cpu_op", "aten::_local_scalar_dense", 80, 10),
        event("cuda_runtime", "cudaStreamSynchronize", 85, 5),
        event("cpu_op", "aten::_local_scalar_dense", 92, 1),  # .item() on a CPU tensor: no sync
        event("cpu_op", "aten::add", 190, 10),
    ]
}
OPS = [
    {"name": "aten::conv2d", "where": "model.py:2", "self_device_us": 30.0, "self_cpu_us": 50.0},
    {"name": "aten::add", "where": None, "self_device_us": 10.0, "self_cpu_us": 5.0},
    {
        "name": "aten::mm",
        "where": "model.py:2",
        "backward": True,
        "self_device_us": 0.0,
        "self_cpu_us": 0.5,
    },  # fmt: skip
    {"name": "aten::copy_", "where": None, "self_device_us": 0.0, "self_cpu_us": 1.0},
]


def test_summarizes_gpu_busy_data_wait_and_syncs():
    s = summarize(TRACE, OPS, steps=2)
    assert s.window_ms == pytest.approx(0.2)
    assert s.gpu_busy == pytest.approx((40 + 5) / 200)
    assert s.data_wait == pytest.approx(40 / 200)
    assert s.syncs_per_step == pytest.approx(0.5)
    assert s.copies_to_host_per_step == pytest.approx(0.5)
    assert [(op["name"], op["share"]) for op in s.top_ops] == [
        ("aten::conv2d", 0.75),
        ("aten::add", 0.25),
    ]


def test_uses_cpu_time_when_no_gpu_ran():
    cpu_only = {
        "traceEvents": [
            e for e in TRACE["traceEvents"] if e["cat"] != "kernel" and e["cat"] != "gpu_memcpy"
        ]
    }
    s = summarize(cpu_only, OPS, steps=2)
    assert s.gpu_busy is None
    assert s.top_ops[0]["share"] == pytest.approx(50 / 56.5)
    assert "top ops by CPU time" in render(s)


def test_names_the_repo_line_behind_each_top_op_and_sync(tmp_path):
    (tmp_path / "model.py").write_text("def forward(x):\n    return self.conv(x)\n")
    (tmp_path / "train.py").write_text("for x in loader:\n    total += loss.item()\n")
    syncs = [{"where": "train.py:2", "count": 4}, {"where": None, "count": 2}]
    s = summarize(TRACE, OPS, steps=2, syncs=syncs, repo=tmp_path)
    assert s.syncs_per_step == pytest.approx(3.0)
    text = render(s)
    assert "2.0 per step at train.py:2: total += loss.item()" in text
    assert "aten::conv2d  at model.py:2: return self.conv(x)" in text
    assert "aten::add  (no repo line" in text
    cpu = render(summarize({"traceEvents": TRACE["traceEvents"][:1]}, OPS, 2, repo=tmp_path))
    assert "aten::mm  backward of model.py:2: return self.conv(x)" in cpu
