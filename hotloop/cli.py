"""Command-line entry point."""

import argparse
import sys

from hotloop import targets, workspace
from hotloop.agent.keep import decide
from hotloop.measurement import Measurement


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
    target = targets.load(args.target)
    print(workspace.working_copy(target, args.label))


def cmd_download(args) -> None:
    from hotloop.harness import modal_runner

    print(modal_runner.download(targets.load(args.target)))


def cmd_measure(args) -> None:
    from hotloop.harness import modal_runner

    target = targets.load(args.target)
    repo = workspace.working_copy(target, args.label)
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

    target = targets.load(args.target)
    repo = workspace.working_copy(target, args.label)
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


def cmd_compare(args) -> None:
    target = targets.load(args.target)
    base = Measurement(workspace.load_runs(target, args.baseline))
    cand = Measurement(workspace.load_runs(target, args.candidate))
    print(describe(args.baseline, base))
    print(describe(args.candidate, cand))
    verdict = decide(base, cand)
    print(f"{'keep' if verdict.keep else 'revert'}: {verdict.reason}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hotloop")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("clone", help="make a working copy of a target repo")
    p.add_argument("target", choices=targets.names())
    p.add_argument("--label", default="baseline")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("download", help="download a target's dataset into the Modal volume")
    p.add_argument("target", choices=targets.names())
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("measure", help="measure a working copy on Modal, once per seed")
    p.add_argument("target", choices=targets.names())
    p.add_argument("--label", default="baseline")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--extra-args", default="", help="appended to the target's command")
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("profile", help="profile a short run of a working copy on Modal")
    p.add_argument("target", choices=targets.names())
    p.add_argument("--label", default="baseline")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--extra-args", default="", help="appended to the target's command")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("compare", help="apply the keep/revert rule to two saved measurements")
    p.add_argument("target", choices=targets.names())
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
