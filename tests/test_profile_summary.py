import pytest

from hotloop.profile.summary import render, summarize


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
        event("cpu_op", "aten::add", 190, 10),
    ]
}
OPS = [
    {"name": "aten::conv2d", "self_device_us": 30.0, "self_cpu_us": 50.0},
    {"name": "aten::add", "self_device_us": 10.0, "self_cpu_us": 5.0},
    {"name": "aten::copy_", "self_device_us": 0.0, "self_cpu_us": 1.0},
]


def test_summarizes_gpu_busy_data_wait_and_syncs():
    s = summarize(TRACE, OPS, steps=2)
    assert s.window_ms == pytest.approx(0.2)
    assert s.gpu_busy == pytest.approx((40 + 5) / 200)
    assert s.data_wait == pytest.approx(40 / 200)
    assert s.syncs_per_step == pytest.approx(1.0)
    assert s.copies_to_host_per_step == pytest.approx(0.5)
    assert s.top_ops == [("aten::conv2d", 0.75), ("aten::add", 0.25)]


def test_uses_cpu_time_when_no_gpu_ran():
    cpu_only = {
        "traceEvents": [
            e for e in TRACE["traceEvents"] if e["cat"] != "kernel" and e["cat"] != "gpu_memcpy"
        ]
    }
    s = summarize(cpu_only, OPS, steps=2)
    assert s.gpu_busy is None
    assert s.top_ops[0] == ("aten::conv2d", pytest.approx(50 / 56))
    assert "top ops by CPU time" in render(s)
