"""Command-line entry point."""

import argparse
import json
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

from hotloop import bench, targets, workspace
from hotloop.agent.keep import decide
from hotloop.measurement import Measurement, tail_loss

RESULTS = Path(__file__).parent.parent / "results"


def target_names() -> list[str]:
    return targets.names() + [f"bench-{c.name}" for c in bench.cases()]


def load_target(name: str) -> targets.Target:
    if name.startswith("bench-"):
        return bench.target(bench.case(name.removeprefix("bench-")))
    return targets.load(name)


def working_copy(target: targets.Target, label: str) -> Path:
    if target.name.startswith("bench-"):
        return bench.working_copy(bench.case(target.name.removeprefix("bench-")), label)
    return working_copy(target, label)


def describe(label: str, m: Measurement) -> str:
    if not m.completed:
        return f"{label:<10} failed: {m.failure()}"
    parts = [f"{label:<10} {m.mean_step_ms:7.1f} ms/step (noise {m.noise:.1%})"]
    if m.peak_mem:
        parts.append(f"peak {m.peak_mem / 1e9:.1f} GB")
    if m.data_wait_share is not None:
        parts.append(f"data wait {m.data_wait_share:.0%}")
    losses = ", ".join(f"{x:.4g}" if x is not None else "none" for x in m.loss)
    parts.append(f"loss {losses}")
    parts.append(m.gpu)
    return "   ".join(parts)


def cmd_clone(args) -> None:
    target = load_target(args.target)
    print(working_copy(target, args.label))


def cmd_download(args) -> None:
    from hotloop.harness import modal_runner

    print(modal_runner.download(load_target(args.target)))


def cmd_measure(args) -> None:
    from hotloop.harness import modal_runner

    target = load_target(args.target)
    repo = working_copy(target, args.label)
    runs = modal_runner.measure(
        target,
        repo,
        gpu=args.gpu,
        extra_args=args.extra_args,
        steps=args.steps,
        warmup=args.warmup,
        verbose=args.verbose,
    )
    path = workspace.save_runs(target, args.label, runs)
    print(describe(args.label, Measurement(runs)))
    print(f"saved {path}")


def cmd_profile(args) -> None:
    from hotloop.harness import modal_runner

    target = load_target(args.target)
    repo = working_copy(target, args.label)
    [run] = modal_runner.measure(
        target,
        repo,
        gpu=args.gpu,
        extra_args=args.extra_args,
        seeds=(0,),
        steps=args.steps,
        profile=True,
        verbose=args.verbose,
    )
    path = workspace.save_runs(target, f"profile-{args.label}", [run])
    print(run.get("profile_text") or f"no profile: {Measurement([run]).failure()}")
    print(f"saved {path}")


def cmd_ab(args) -> None:
    from hotloop.harness import modal_runner

    target = load_target(args.target)
    base_runs, cand_runs = modal_runner.ab(
        target,
        modal_runner.Job(working_copy(target, args.baseline), args.base_args),
        modal_runner.Job(working_copy(target, args.candidate), args.cand_args),
        gpu=args.gpu,
        verbose=args.verbose,
    )
    path = workspace.save_ab(
        target, args.baseline, args.candidate, base_runs, cand_runs, args.base_args, args.cand_args
    )
    base, cand = Measurement(base_runs), Measurement(cand_runs)
    print(describe(args.baseline, base))
    print(describe(args.candidate, cand))
    verdict = decide(base, cand)
    print(f"{'keep' if verdict.keep else 'revert'}: {verdict.reason}")
    print(f"saved {path}")


def cmd_run(args) -> None:
    from hotloop.agent import loop, report
    from hotloop.agent.models import MODELS
    from hotloop.agent.session import Session

    target = load_target(args.target)
    label = f"agent-{args.model}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    session = Session(target, working_copy(target, label), gpu=args.gpu, rounds=args.rounds)
    print(describe("baseline", session.measure_baseline()))
    print(session.profile())
    result = loop.run(session, MODELS[args.model], log=print)
    path = report.write(result, RESULTS / "agent" / target.name, label)
    final = result.final.verdict.reason if result.final else "no change kept"
    print(f"result: {final}")
    print(f"report: {path}")


def summarize_runs(m: Measurement) -> list[dict]:
    return [
        {
            "seed": r.get("seed"),
            "completed": r.get("completed"),
            "step_ms": r.get("step_ms"),
            "data_wait_ms": r.get("data_wait_ms"),
            "peak_mem_allocated": r.get("peak_mem_allocated"),
            "loss_tail": tail_loss(r.get("losses")),
            "gpu": r.get("gpu"),
            "cpu": r.get("cpu"),
            "container": r.get("container"),
            "torch": r.get("torch"),
        }
        for r in m.runs
    ]


def cmd_record(args) -> None:
    """Save the latest paired comparison, with the candidate's diff, as a result file."""
    target = load_target(args.target)
    pair = workspace.load_ab(target, args.baseline, args.candidate)
    base = Measurement(pair["baseline"]["runs"])
    cand = Measurement(pair["candidate"]["runs"])
    verdict = decide(base, cand)
    record = {
        "target": target.name,
        "repo": target.repo,
        "commit": target.commit,
        "command": target.command,
        "baseline_args": pair["baseline"]["extra_args"],
        "candidate_args": pair["candidate"]["extra_args"],
        "change": args.change,
        "gpu": base.gpu,
        "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
        "baseline": summarize_runs(base),
        "candidate": summarize_runs(cand),
        "speedup": verdict.speedup,
        "keep": verdict.keep,
        "reason": verdict.reason,
        "diff": workspace.diff(working_copy(target, args.candidate)),
    }
    path = RESULTS / args.folder / f"{target.name}-{args.candidate}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{'keep' if verdict.keep else 'revert'}: {verdict.reason}")
    print(f"saved {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hotloop")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("clone", help="make a working copy of a target repo")
    p.add_argument("target", choices=target_names())
    p.add_argument("--label", default="baseline")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("download", help="download a target's dataset into the Modal volume")
    p.add_argument("target", choices=target_names())
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("measure", help="measure a working copy on Modal, once per seed")
    p.add_argument("target", choices=target_names())
    p.add_argument("--label", default="baseline")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--extra-args", default="", help="appended to the target's command")
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("profile", help="profile a short run of a working copy on Modal")
    p.add_argument("target", choices=target_names())
    p.add_argument("--label", default="baseline")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--extra-args", default="", help="appended to the target's command")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("ab", help="run two working copies in the same containers and compare")
    p.add_argument("target", choices=target_names())
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--base-args", default="", help="appended to the baseline's command")
    p.add_argument("--cand-args", default="", help="appended to the candidate's command")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("run", help="let the agent find and fix slow training code")
    p.add_argument("target", choices=target_names())
    p.add_argument("--model", default="ornith-9b")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--rounds", type=int, default=5)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("record", help="save the latest paired comparison under results/")
    p.add_argument("target", choices=target_names())
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--change", required=True, help="one line on what the candidate changes")
    p.add_argument("--folder", default="runs")
    p.set_defaults(func=cmd_record)

    args = parser.parse_args(argv)
    # Exit through the normal path when killed, so Modal apps stop and the cost
    # ledger records the run instead of leaving containers running.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
