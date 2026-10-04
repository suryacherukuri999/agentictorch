import pytest

from agentictorch.bench import cases, planted_source, target


@pytest.mark.parametrize("case", cases(), ids=lambda c: c.name)
def test_each_case_plants_its_edits_and_still_compiles(case):
    source = planted_source(case)
    compile(source, f"{case.name}.py", "exec")
    for _, new in case.edits:
        assert new in source


def test_every_script_has_a_clean_control():
    scripts = {c.script for c in cases()}
    controls = {c.script for c in cases() if c.control}
    assert scripts == controls


def test_case_arguments_extend_the_scripts_command():
    [oom] = [c for c in cases() if c.name == "gpt-out-of-memory"]
    assert target(oom).command.endswith("--block 1024")
    assert target(oom).name == "bench-gpt-out-of-memory"
