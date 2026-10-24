"""A thin client for OpenAI-compatible chat APIs.

Ollama, Gemini and OpenAI all accept the same chat-completions request with
tools, so one client covers every model. Responses are cached on disk, and 429s
are retried with backoff.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

WORK = Path.home() / "agentictorch-work"
CACHE = WORK / "llm-cache"
KEYS = WORK / ".env"


@dataclass(frozen=True)
class Model:
    name: str
    base_url: str
    key_env: str | None = None
    temperature: float = 0.0
    reasoning_effort: str | None = None  # for thinking models; omitted when None
    max_tokens_field: str = "max_tokens"  # newer OpenAI models want max_completion_tokens
    api: str = "chat"  # "responses" for models that take tools only through /v1/responses


@dataclass(frozen=True)
class Reply:
    message: dict  # the assistant message, with "content" and maybe "tool_calls"
    input_tokens: int
    output_tokens: int
    cached: bool


def chat(model: Model, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> Reply:
    if model.api == "responses":
        request = _responses_request(model, messages, tools, max_tokens)
    else:
        request = {
            "model": model.name,
            "messages": messages,
            "tools": tools,
            model.max_tokens_field: max_tokens,
            "temperature": model.temperature,
        }
        if model.reasoning_effort:
            request["reasoning_effort"] = model.reasoning_effort
    key = hashlib.sha256(json.dumps([model.base_url, request], sort_keys=True).encode())
    cached = CACHE / f"{key.hexdigest()}.json"
    if cached.exists():
        body = json.loads(cached.read_text())
        return _reply(model, body, cached=True)

    body = _post(model, request)
    if model.api == "responses":
        body = _as_chat(body)
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(body))
    return _reply(model, body, cached=False)


def api_key(name: str) -> str | None:
    """Read a key from the environment or ~/agentictorch-work/.env, never from the repo."""
    if os.environ.get(name):
        return os.environ[name]
    if KEYS.exists():
        for line in KEYS.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip()
    return None


def _post(model: Model, request: dict, attempts: int = 6) -> dict:
    headers = {"Content-Type": "application/json"}
    if model.key_env:
        key = api_key(model.key_env)
        if not key:
            raise RuntimeError(f"{model.key_env} is not set in the environment or {KEYS}")
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(request).encode()
    path = "/responses" if model.api == "responses" else "/chat/completions"
    url = model.base_url.rstrip("/") + path
    for attempt in range(attempts):
        try:
            http = urllib.request.Request(url, data=data, headers=headers)
            # A local model serving several agents can take minutes per turn.
            with urllib.request.urlopen(http, timeout=1200) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            # Rate limits and overload pass; a 500 may not, so it gets one retry.
            transient = e.code in (429, 502, 503) or (e.code == 500 and attempt == 0)
            if not transient or attempt == attempts - 1:
                raise RuntimeError(f"{model.name}: HTTP {e.code}: {e.read()[:500]!r}") from None
            time.sleep(min(2**attempt * 5, 120))
        except (TimeoutError, urllib.error.URLError) as e:
            # A timeout or a dropped connection gets one retry.
            if attempt > 0:
                raise RuntimeError(f"{model.name}: {e}") from None
            time.sleep(5)
    raise AssertionError("unreachable")


def _responses_request(model: Model, messages: list[dict], tools: list[dict], max_tokens: int):
    """The same conversation as a Responses API request. An assistant turn that came from
    this API goes back as its raw output items: reasoning models need their reasoning
    items returned with the tool calls."""
    items = []
    for m in messages:
        if m["role"] == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m["content"],
                }
            )
        elif m.get("items"):
            items.extend(m["items"])
        else:
            role = "developer" if m["role"] == "system" else m["role"]
            items.append({"role": role, "content": m.get("content") or ""})
    request = {
        "model": model.name,
        "input": items,
        "tools": [{"type": "function", **t["function"]} for t in tools],
        "max_output_tokens": max_tokens,
    }
    if model.reasoning_effort:
        request["reasoning"] = {"effort": model.reasoning_effort}
    return request


def _as_chat(body: dict) -> dict:
    """A Responses API reply in the shape of a chat completion, keeping the raw output items."""
    output = body.get("output") or []
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content") or []
    )
    calls = [
        {
            "id": item["call_id"],
            "type": "function",
            "function": {"name": item["name"], "arguments": item["arguments"]},
        }
        for item in output
        if item.get("type") == "function_call"
    ]
    message = {"role": "assistant", "content": text or None, "items": output}
    if calls:
        message["tool_calls"] = calls
    usage = body.get("usage") or {}
    return {
        "choices": [{"message": message}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
        },
    }


def _reply(model: Model, body: dict, cached: bool) -> Reply:
    usage = body.get("usage") or {}
    tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return Reply(
        message=body["choices"][0]["message"],
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cached=cached,
    )
