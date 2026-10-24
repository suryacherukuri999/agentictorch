"""The agent loop: the model reads code, edits it and measures each change on a GPU.

The same loop runs the setups the agent is compared with. They get fewer tools
and no measurements, make one patch blind, and the patch is measured afterwards
only to score it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from agentictorch.agent import llm
from agentictorch.agent.session import Attempt, Session
from agentictorch.agent.tools import SCHEMAS, Toolbox
from agentictorch.survey.scan import scan_repo

SYSTEM = """You make PyTorch training code faster without changing what it computes.

How to work:
- Start from the measurements and hints you are given, and read the training code before editing.
- Make one focused change at a time. Use quick_check to make sure it runs (it uses no round),
  then call run_benchmark. It measures your edits against the best version so far on the GPU
  and keeps them only if they are faster beyond noise with the same loss. Otherwise they are
  reverted.
- Use search_docs to check how to apply a fix correctly, for example mixed precision,
  DataLoader workers, pinned memory, fused optimizers or torch.compile.
- Don't change the model, the data, the batch size, the learning rate, the number of steps,
  or what gets printed or saved. Only make the same training run faster.
- Don't add command-line flags; make the change directly in the code.
- For mixed precision, use bfloat16. float16 is refused unless the code also uses a GradScaler.
- Call finish when no promising change is left."""

BLIND = """You make PyTorch training code faster without changing what it computes.

How to work:
- Read the training code and the hints you are given.
- Make every change you are confident will make training faster, with apply_edit. You can't
  run or measure anything, so there is no second chance to fix a mistake.
- Don't change the model, the data, the batch size, the learning rate, the number of steps,
  or what gets printed or saved. Only make the same training run faster.
- Don't add command-line flags; make the change directly in the code.
- For mixed precision, use bfloat16. float16 needs a GradScaler.
- Call finish when you are done editing."""

NUDGE = "Use a tool. Call finish when no promising change is left."


@dataclass(frozen=True)
class Setup:
    name: str
    tools: tuple[str, ...]
    measured: bool  # measurements and run_benchmark; otherwise one blind patch


READ_AND_EDIT = ("list_files", "read_file", "search_code", "apply_edit", "finish")
SETUPS = {
    "llm": Setup("llm", READ_AND_EDIT, measured=False),
    "llm-kb": Setup("llm-kb", (*READ_AND_EDIT, "search_docs"), measured=False),
    "agent": Setup(
        "agent", (*READ_AND_EDIT, "search_docs", "profile", "quick_check", "run_benchmark"), True
    ),
}


@dataclass
class Run:
    session: Session
    model: str
    messages: list[dict]
    summary: str | None
    final: Attempt | None
    sources: dict[str, str]
    turns: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    stopped: str | None = None  # why the run ended before the model finished, if it did


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


def context(session: Session, measured: bool = True) -> str:
    if not measured:
        lines = [f"Training command: {session.target.command}"]
        found = hints(session)
        if found:
            lines += ["", "Hints from scanning the code (likely problems, not measured):"]
            lines += [f"- {h}" for h in found]
        return "\n".join(lines)
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
    setup: Setup = SETUPS["agent"],
    max_turns: int = 40,
    index=None,
    log: Callable[[str], None] = lambda line: None,
) -> Run:
    toolbox = Toolbox(session, index, allowed=setup.tools)
    schemas = [s for s in SCHEMAS if s["function"]["name"] in setup.tools]
    messages = [
        {"role": "system", "content": SYSTEM if setup.measured else BLIND},
        {"role": "user", "content": context(session, setup.measured)},
    ]
    result = Run(session, model.name, messages, None, None, toolbox.sources)
    for _ in range(max_turns):
        try:
            reply = llm.chat(model, messages, schemas)
        except (RuntimeError, OSError) as e:
            # The changes already kept still count.
            result.stopped = f"the model failed: {e}"
            log(f"stopped: {result.stopped}")
            break
        result.turns += 1
        result.tokens["input"] += reply.input_tokens
        result.tokens["output"] += reply.output_tokens
        # "items" holds a Responses API turn's raw output, which goes back in the next request.
        keep = ("role", "content", "tool_calls", "items")
        message = {k: v for k, v in reply.message.items() if k in keep}
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
    try:
        if not setup.measured:
            if session.diff():
                result.final = session.benchmark(f"{setup.name} patch, made without measuring")
        elif session.kept:
            result.final = session.final()
    except Exception as e:
        # The report still lists every attempt; `agentictorch ab` can rerun the final pair.
        result.stopped = f"the final measurement failed: {type(e).__name__}: {e}"
        log(f"stopped: {result.stopped}")
    return result
