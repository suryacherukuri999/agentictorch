import json
import subprocess

import pytest

from hotloop.agent import llm, loop
from hotloop.agent import session as session_module
from hotloop.agent.session import Session
from hotloop.agent.tools import MAX_CHECKS, Toolbox
from hotloop.measurement import Measurement
from hotloop.targets import Target
from hotloop.tools.files import apply_edit


def run_for(repo, seed, container):
    fast = "FAST" in (repo / "train.py").read_text()
    return {"completed": True, "step_ms": 50.0 if fast else 100.0, "losses": [1.0] * 20,
            "seed": seed, "gpu": "NVIDIA L4", "container": container}  # fmt: skip


def call(name, **args):
    return {"id": f"call-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}  # fmt: skip


SCRIPT = [
    {"role": "assistant", "content": "Let me look first."},
    {"role": "assistant", "tool_calls": [call("read_file", path="train.py")]},
    {
        "role": "assistant",
        "tool_calls": [call("apply_edit", path="train.py", old="SLOW", new="FAST")],
    },
    {"role": "assistant", "tool_calls": [call("run_benchmark", description="make it fast")]},
    {"role": "assistant", "tool_calls": [call("finish", summary="it was slow")]},
]


@pytest.fixture
def session(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("speed = 'SLOW'\nfor x in loader:\n    loss.backward()\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "start"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)

    def ab(target, base, cand, gpu):
        pairs = [
            (run_for(base.repo_dir, s, f"c{s}"), run_for(cand.repo_dir, s, f"c{s}")) for s in (0, 1)
        ]
        return [b for b, _ in pairs], [c for _, c in pairs]

    monkeypatch.setattr(session_module.modal_runner, "ab", ab)
    monkeypatch.setattr(session_module.workspace, "WORK", tmp_path / "work")
    s = Session(Target(name="t", command="python train.py"), repo, rounds=3)
    s.baseline = Measurement([run_for(repo, 0, "x"), run_for(repo, 1, "y")])
    return s


def test_runs_tools_until_finish_and_measures_the_result(session, monkeypatch):
    replies = iter(SCRIPT)
    seen = []

    def chat(model, messages, tools):
        seen.append(list(messages))
        return llm.Reply(next(replies), 10, 5, 0.0, False)

    monkeypatch.setattr(loop.llm, "chat", chat)
    result = loop.run(session, llm.Model("fake", "http://x"))
    assert result.summary == "it was slow"
    assert result.turns == 5
    assert seen[1][-1] == {"role": "user", "content": loop.NUDGE}
    assert "kept: 2.00x faster" in seen[-1][-1]["content"]
    assert result.final.verdict.speedup == pytest.approx(2.0)
    assert result.tokens == {"input": 50, "output": 25}


def test_context_reports_the_baseline_and_static_hints(session):
    text = loop.context(session)
    assert "100.0 ms/step" in text
    assert "no mixed precision anywhere in the code" in text
    assert "You have 3 benchmark rounds." in text


def test_blind_setup_makes_one_patch_and_measures_it_afterwards(session, monkeypatch):
    calls = [
        call("run_benchmark", description="not allowed"),
        call("apply_edit", path="train.py", old="SLOW", new="FAST"),
        call("finish", summary="done"),
    ]
    script = iter({"role": "assistant", "tool_calls": [c]} for c in calls)
    seen = []

    def chat(model, messages, tools):
        seen.append(([t["function"]["name"] for t in tools], list(messages)))
        return llm.Reply(next(script), 10, 5, 0.0, False)

    monkeypatch.setattr(loop.llm, "chat", chat)
    result = loop.run(session, llm.Model("fake", "http://x"), setup=loop.SETUPS["llm"])
    assert "run_benchmark" not in seen[0][0]
    assert "ms/step" not in seen[0][1][1]["content"]  # no measurements for a blind setup
    assert seen[1][1][-1]["content"] == "error: run_benchmark is not available"
    assert len(session.attempts) == 1
    assert result.final.verdict.speedup == pytest.approx(2.0)


def test_a_failed_benchmark_shows_the_model_the_end_of_the_log(session, monkeypatch):
    def ab(target, base, cand, gpu):
        log = 'File "/work/repo1/train.py", line 2, in step\nRuntimeError: boom'
        crashed = {"completed": False, "seed": 0, "gpu": "NVIDIA L4", "container": "c0",
                   "error": "RuntimeError: boom", "log_tail": log}  # fmt: skip
        good = run_for(base.repo_dir, 0, "c0")
        return [good, run_for(base.repo_dir, 1, "c1")], [
            crashed,
            crashed | {"seed": 1, "container": "c1"},
        ]

    monkeypatch.setattr(session_module.modal_runner, "ab", ab)
    toolbox = Toolbox(session)
    apply_edit(session.repo, "train.py", "SLOW", "FAST")
    text = toolbox.call("run_benchmark", json.dumps({"description": "x"}))
    assert text.startswith("reverted: run failed: RuntimeError: boom")
    assert 'File "train.py", line 2, in step' in text


def test_quick_checks_are_capped(session, monkeypatch):
    monkeypatch.setattr(session_module.Session, "check", lambda self: "ok")
    toolbox = Toolbox(session)
    outputs = [toolbox.call("quick_check", "{}") for _ in range(MAX_CHECKS + 1)]
    assert outputs[:-1] == ["ok"] * MAX_CHECKS
    assert outputs[-1].startswith("error: already ran")


def test_a_failed_gpu_call_is_an_error_for_the_model_not_a_crash(session, monkeypatch):
    def lost(self, description):
        raise KeyError("seed")

    monkeypatch.setattr(session_module.Session, "benchmark", lost)
    apply_edit(session.repo, "train.py", "SLOW", "FAST")
    output = Toolbox(session).call("run_benchmark", '{"description": "x"}')
    assert output == "error: run_benchmark failed: KeyError: 'seed'"


def test_a_model_failure_ends_the_run_but_keeps_what_was_measured(session, monkeypatch):
    script = iter(SCRIPT[1:4])

    def chat(model, messages, tools):
        message = next(script, None)
        if message is None:
            raise RuntimeError("ornith: timed out")
        return llm.Reply(message, 10, 5, 0.0, False)

    monkeypatch.setattr(loop.llm, "chat", chat)
    run = loop.run(session, llm.Model("m", "http://x"))

    assert run.stopped == "the model failed: ornith: timed out"
    assert len(session.kept) == 1
    assert run.final is not None and run.final.verdict.keep
