import json

import pytest

from hotloop.agent import llm

PAID = llm.Model("cheap-model", "https://api.example.com/v1", "EXAMPLE_KEY", 0.5, 2.0)
FREE = llm.Model("local", "http://localhost:11434/v1")
BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(llm, "KEYS", tmp_path / ".env")
    calls = []
    monkeypatch.setattr(llm, "_post", lambda model, request: calls.append(request) or BODY)
    return calls


def test_costs_come_from_token_usage_and_are_recorded(isolated):
    reply = llm.chat(PAID, [{"role": "user", "content": "hello"}], tools=[])
    assert reply.cost == pytest.approx((1000 * 0.5 + 500 * 2.0) / 1e6)
    assert llm.paid_spend() == pytest.approx(reply.cost)


def test_identical_requests_are_served_from_the_cache(isolated):
    messages = [{"role": "user", "content": "hello"}]
    first = llm.chat(FREE, messages, tools=[])
    second = llm.chat(FREE, messages, tools=[])
    assert len(isolated) == 1
    assert not first.cached and second.cached and second.cost == 0


def test_paid_calls_stop_at_the_budget(isolated):
    llm.LEDGER.write_text(json.dumps({"cost": llm.PAID_BUDGET - 0.001}) + "\n")
    with pytest.raises(llm.BudgetExceeded):
        llm.chat(PAID, [{"role": "user", "content": "x" * 30000}], tools=[], max_tokens=4096)
    assert isolated == []


def test_free_models_ignore_the_budget(isolated):
    llm.LEDGER.write_text(json.dumps({"cost": 100.0}) + "\n")
    assert (
        llm.chat(FREE, [{"role": "user", "content": "hello"}], tools=[]).message["content"] == "hi"
    )


def test_keys_are_read_from_the_env_file(monkeypatch):
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    llm.KEYS.write_text("OTHER=1\nEXAMPLE_KEY=sk-test\n")
    assert llm.api_key("EXAMPLE_KEY") == "sk-test"
    assert llm.api_key("MISSING") is None


def test_sends_sampling_settings_for_thinking_models(isolated):
    model = llm.Model("thinker", "http://x", temperature=0.6, reasoning_effort="low")
    llm.chat(model, [{"role": "user", "content": "hi"}], tools=[])
    llm.chat(FREE, [{"role": "user", "content": "hi"}], tools=[])
    assert isolated[0]["temperature"] == 0.6 and isolated[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in isolated[1]
