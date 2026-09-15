"""Does a repo's maintainer merge pull requests from other people?

python -m hotloop.survey.activity [--save results/survey/maintainers.json] owner/repo ...

Pull requests only go to repos that do. Read-only GitHub API calls with no login,
two per repo (the unauthenticated limit is 60 an hour).
"""

import json
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Authors who are the maintainers themselves.
INSIDERS = {"OWNER", "MEMBER", "COLLABORATOR"}


def _get(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def outside_merges(repo: str) -> list[dict]:
    """Merged pull requests by people other than the maintainers, newest first,
    from the 100 most recently updated closed ones."""
    pulls = _get(
        f"https://api.github.com/repos/{repo}/pulls"
        "?state=closed&sort=updated&direction=desc&per_page=100"
    )
    merged = [
        {
            "number": p["number"],
            "merged": p["merged_at"][:10],
            "author": p["user"]["login"],
            "title": p["title"],
        }
        for p in pulls
        if p["merged_at"] and p["author_association"] not in INSIDERS and p["user"]["type"] != "Bot"
    ]
    return sorted(merged, key=lambda p: p["merged"], reverse=True)


def summary(repo: str, today: datetime) -> dict:
    merges = outside_merges(repo)

    def since(months: int) -> int:
        cutoff = (today - timedelta(days=30 * months)).date().isoformat()
        return sum(1 for m in merges if m["merged"] >= cutoff)

    return {
        "repo": repo,
        "stars": _get(f"https://api.github.com/repos/{repo}")["stargazers_count"],
        "last_outside_merge": merges[0]["merged"] if merges else None,
        "outside_merges_6_months": since(6),
        "outside_merges_12_months": since(12),
        "latest": merges[:3],
    }


def main(argv: list[str]) -> None:
    save = argv[argv.index("--save") + 1] if "--save" in argv else None
    repos = [a for a in argv if "/" in a and a != save]
    today = datetime.now(UTC)
    found = []
    for repo in repos:
        s = summary(repo, today)
        found.append(s)
        latest = ", ".join(f"#{m['number']} {m['merged']} by {m['author']}" for m in s["latest"])
        counts = f"6 mo {s['outside_merges_6_months']:>2}  12 mo {s['outside_merges_12_months']:>2}"
        print(f"{repo:<48} last {s['last_outside_merge'] or 'never':<10} {counts}  {latest}")
        time.sleep(1)
    if save:
        record = {"checked": today.date().isoformat(), "repos": found}
        Path(save).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
