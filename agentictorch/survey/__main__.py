"""Rebuild the survey: python -m agentictorch.survey [--pinned results/survey/repos.csv]

Without --pinned it searches GitHub and scans each repo's default branch.
With --pinned it rescans the exact repos and commits of an earlier run.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from agentictorch.survey import github, report
from agentictorch.survey.scan import scan_repo

CLONES = Path.home() / "agentictorch-work" / "survey"
RESULTS = Path(__file__).parent.parent.parent / "results" / "survey"


def from_search() -> list[dict]:
    repos, skipped = github.search()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "skipped.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["repo", "reason"])
        writer.writerows(sorted(skipped.items()))
    return [
        {
            "repo": r["full_name"],
            "url": r["html_url"],
            "stars": r["stargazers_count"],
            "pushed_at": r["pushed_at"][:10],
            "description": (r["description"] or "")[:100],
            "commit": None,
        }
        for r in repos
    ]


def from_pinned(path: Path) -> list[dict]:
    with path.open() as f:
        return [{**r, "stars": int(r["stars"])} for r in csv.DictReader(f)]


def scan_one(repo: dict) -> dict | None:
    dest = CLONES / repo["repo"].replace("/", "__")
    try:
        commit = github.clone(repo["repo"], dest, repo["commit"])
    except Exception as e:  # deleted, renamed or too slow; leave it out and say so
        print(f"skip {repo['repo']}: {e}")
        return None
    return report.row(repo, commit, scan_repo(dest))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m agentictorch.survey")
    parser.add_argument("--pinned", type=Path, help="rescan the repos and commits in this CSV")
    args = parser.parse_args(argv)

    repos = from_pinned(args.pinned) if args.pinned else from_search()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(scan_one, repos) if r]
    rows.sort(key=lambda r: -r["stars"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "repos.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text = report.summary(rows, datetime.now(UTC).date())
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
