"""Find candidate repos with GitHub's search API and make shallow, sparse clones."""

import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

# (query, pages of 100 results), sorted by stars.
QUERIES = [
    (
        '"pytorch implementation" in:name,description language:Python stars:>500 '
        "pushed:>2024-01-01",
        2,
    ),  # fmt: skip
    ('"official pytorch" in:description language:Python stars:>200 pushed:>2024-01-01', 1),
    ("pytorch fine-tuning in:name,description language:Python stars:>500 pushed:>2024-01-01", 1),
    ("pytorch diffusion in:name,description language:Python stars:>500 pushed:>2024-01-01", 1),
]
SKIP_WORDS = re.compile(
    r"\b(awesome|tutorials?|course|lectures?|book|cheat ?sheet|collection|list of"
    r"|library|toolbox|toolkit|zoo)\b",
    re.IGNORECASE,
)
# Frameworks whose names and descriptions don't say so.
FRAMEWORKS = {"huggingface/diffusers"}
MAX_SIZE_KB = 300 * 1024
# Code, plus the scripts, configs and READMEs where mixed precision is often switched on.
SPARSE_PATTERNS = [
    "*.py",
    "*.sh",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.cfg",
    "*.toml",
    "*.ini",
    "*.md",
    "requirements*.txt",
]


def search() -> tuple[list[dict], dict[str, str]]:
    """Return the repos to scan and, for the rest, why they were skipped."""
    found = {}
    for query, pages in QUERIES:
        for page in range(1, pages + 1):
            params = {"q": query, "sort": "stars", "per_page": 100, "page": page}
            url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                for item in json.load(response)["items"]:
                    found.setdefault(item["full_name"], item)
            time.sleep(7)  # unauthenticated search allows 10 requests a minute

    keep, skipped = [], {}
    for repo in found.values():
        reason = skip_reason(repo)
        if reason:
            skipped[repo["full_name"]] = reason
        else:
            keep.append(repo)
    return keep, skipped


def skip_reason(repo: dict) -> str | None:
    if repo["archived"]:
        return "archived"
    if repo["size"] > MAX_SIZE_KB:
        return "over 300 MB"
    if repo["full_name"] in FRAMEWORKS:
        return "framework"
    if SKIP_WORDS.search(f"{repo['name']} {repo['description'] or ''}"):
        return "list, tutorial, course or library"
    return None


def clone(full_name: str, dest: Path, commit: str | None = None) -> str:
    """Clone the default branch (or `commit`) with only the sparse patterns checked out."""
    url = f"https://github.com/{full_name}.git"
    if not dest.exists():
        _git("clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse", url, dest)
        if commit:
            _git("-C", dest, "fetch", "--quiet", "--depth", "1", "origin", commit)
            _git("-C", dest, "checkout", "--quiet", commit)
        _git("-C", dest, "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS)
    return _git("-C", dest, "rev-parse", "HEAD").strip()


def _git(*args) -> str:
    out = subprocess.run(
        ["git", *map(str, args)], check=True, capture_output=True, text=True, timeout=600
    )
    return out.stdout
