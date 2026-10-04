import json

import pytest

from agentictorch.agent import llm

PAID = llm.Model("cheap-model", "https://api.example.com/v1", "EXAMPLE_KEY", 0.5, 2.0)
FREE = llm.Model("local", "http://localhost:11434/v1")
# The fixture below replaces _post in every test; the retry tests need the real one.
REAL_POST = llm._post
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


def test_a_timeout_is_retried_once(monkeypatch):
    import io
    import urllib.request

    replies = [TimeoutError("timed out"), io.BytesIO(json.dumps(BODY).encode())]

    def urlopen(request, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    assert REAL_POST(FREE, {"model": "m"}) == BODY
    assert not replies

    replies[:] = [TimeoutError("timed out"), TimeoutError("timed out")]
    with pytest.raises(RuntimeError, match="timed out"):
        REAL_POST(FREE, {"model": "m"})


def test_responses_models_send_items_and_get_chat_shaped_replies(monkeypatch):
    sent = []
    output = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "call_id": "call_1", "name": "read_file",
         "arguments": '{"path": "train.py"}'},
    ]  # fmt: skip
    reply = {"output": output, "usage": {"input_tokens": 100, "output_tokens": 20}}
    monkeypatch.setattr(llm, "_post", lambda model, request: sent.append(request) or reply)
    model = llm.Model("sol", "https://api.example.com/v1", "EXAMPLE_KEY", 2.0, 10.0,
                      reasoning_effort="medium", api="responses")  # fmt: skip
    tool = {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    messages = [{"role": "system", "content": "be fast"}, {"role": "user", "content": "go"}]

    first = llm.chat(model, messages, [tool])
    assert sent[0]["input"] == [
        {"role": "developer", "content": "be fast"},
        {"role": "user", "content": "go"},
    ]
    assert sent[0]["tools"] == [{"type": "function", "name": "read_file", "parameters": {}}]
    assert sent[0]["reasoning"] == {"effort": "medium"}
    assert first.message["tool_calls"][0]["id"] == "call_1"
    assert first.cost == pytest.approx((100 * 2.0 + 20 * 10.0) / 1e6)

    messages += [first.message, {"role": "tool", "tool_call_id": "call_1", "content": "x = 1"}]
    llm.chat(model, messages, [tool])
    assert sent[1]["input"][2:] == [
        *output,
        {"type": "function_call_output", "call_id": "call_1", "output": "x = 1"},
    ]


def test_cached_prompt_tokens_cost_the_cached_price(monkeypatch):
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 10,
                  "prompt_tokens_details": {"cached_tokens": 900}},
    }  # fmt: skip
    monkeypatch.setattr(llm, "_post", lambda model, request: body)
    model = llm.Model("m", "https://api.example.com/v1", "EXAMPLE_KEY", 2.0, 10.0, 0.10)
    reply = llm.chat(model, [{"role": "user", "content": "x"}], [])
    assert reply.cost == pytest.approx((100 * 2.0 + 900 * 0.10 + 10 * 10.0) / 1e6)


def test_a_reconciled_total_restarts_the_spend():
    entries = [{"cost": 3.0}, {"reconciled_total": 0.41, "cost": 0.0}, {"cost": 0.05}]
    llm.LEDGER.write_text("".join(json.dumps(e) + "\n" for e in entries))
    assert llm.paid_spend() == pytest.approx(0.46)
