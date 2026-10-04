from agentictorch.measurement import Measurement

TRACEBACK = "\n".join(
    [f"step {i}" for i in range(20)]
    + [
        "Traceback (most recent call last):",
        '  File "/work/repo1/train.py", line 85, in <module>',
        "    model = torch.compile(model, reduce-overhead=True)",
        "SyntaxError: expression cannot contain assignment",
    ]
)


def test_failure_log_gives_the_end_of_the_log_with_repo_paths():
    m = Measurement([{"completed": False, "error": "SyntaxError: x", "log_tail": TRACEBACK}])
    log = m.failure_log(lines=15)
    assert len(log.splitlines()) == 15
    assert 'File "train.py", line 85, in <module>' in log
    assert log.endswith("SyntaxError: expression cannot contain assignment")


def test_failure_log_is_empty_when_every_run_completed():
    assert Measurement([{"completed": True, "log_tail": "fine"}]).failure_log() == ""
