"""The agent loop: the model reads code, edits it and measures each change on a GPU."""

from collections.abc import Callable
from dataclasses import dataclass, field

from hotloop.agent import llm
from hotloop.agent.session import Attempt, Session
from hotloop.agent.tools import SCHEMAS, Toolbox
from hotloop.survey.scan import scan_repo

SYSTEM = """You make PyTorch training code faster without changing what it computes.

How to work:
- Start from the measurements and hints you are given, and read the training code before editing.
- Make one focused change at a time, then call run_benchmark. It measures your edits against
  the best version so far on the GPU and keeps them only if they are faster beyond noise with
  the same loss. Otherwise they are reverted.
- Use search_docs to check how to apply a fix correctly, for example mixed precision,
  DataLoader workers, pinned memory, fused optimizers or torch.compile.
- Don't change the model, the data, the batch size, the learning rate, the number of steps,
  or what gets printed or saved. Only make the same training run faster.
- Don't add command-line flags; make the change directly in the code.
- Call finish when no promising change is left."""

NUDGE = "Use a tool. Call finish when no promising change is left."


@dataclass
class Run:
    session: Session
    model: str
    messages: list[dict]
    summary: str | None
    final: Attempt | None
    sources: dict[str, str]
    cost: float = 0.0
    turns: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})


def hints(session: Session) -> list[str]:
    found = scan_repo(session.repo)
    lines = []
    if not found.amp:
        lines.append("no mixed precision anywhere in the code")
    lines += [f"training DataLoader on the main thread: {x}" for x in found.main_thread_loaders]
    lines += [f"host-device sync on every step: {x}" for x in found.loop_syncs]
    if not found.uses_compile:
        lines.append("torch.compile is not used")
    return lines


def context(session: Session) -> str:
    base = session.baseline
    lines = [
        f"Training command: {session.target.command}",
        f"GPU: {session.gpu}",
        f"Baseline over {len(base.runs)} runs: {base.mean_step_ms:.1f} ms/step"
        + (f", peak memory {base.peak_mem / 1e9:.1f} GB" if base.peak_mem else "")
        + (
            f", waiting on the DataLoader {base.data_wait_share:.0%} of the time"
            if base.data_wait_share is not None
            else ""
        ),
    ]
    if session.profiles:
        lines += ["", "Profile of the baseline:", session.profiles[0]]
    found = hints(session)
    if found:
        lines += ["", "Hints from scanning the code (likely problems, not measured):"]
        lines += [f"- {h}" for h in found]
    lines += ["", f"You have {session.rounds} benchmark rounds."]
    return "\n".join(lines)


def run(
    session: Session,
    model: llm.Model,
    max_turns: int = 40,
    index=None,
    log: Callable[[str], None] = lambda line: None,
) -> Run:
    toolbox = Toolbox(session, index)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": context(session)},
    ]
    result = Run(session, model.name, messages, None, None, toolbox.sources)
    for _ in range(max_turns):
        reply = llm.chat(model, messages, SCHEMAS)
        result.turns += 1
        result.cost += reply.cost
        result.tokens["input"] += reply.input_tokens
        result.tokens["output"] += reply.output_tokens
        message = {k: v for k, v in reply.message.items() if k in ("role", "content", "tool_calls")}
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": NUDGE})
            continue
        for call in calls:
            name, arguments = call["function"]["name"], call["function"].get("arguments")
            log(f"{name}({(arguments or '')[:120]})")
            output = toolbox.call(name, arguments)
            log(f"  -> {output.splitlines()[0][:160] if output else ''}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        if toolbox.finished is not None or session.rounds_left == 0:
            break
    result.summary = toolbox.finished
    if session.kept:
        result.final = session.final()
    return result
