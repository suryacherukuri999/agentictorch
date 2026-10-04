"""Download the knowledge base sources and split them into sections with URLs.

The downloaded corpus lives in ~/agentictorch-work/kb and is never committed:
python -m agentictorch.kb.fetch
"""

import html.parser
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

KB = Path.home() / "agentictorch-work" / "kb"

DOCS = {
    "PyTorch performance tuning guide": "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
    "PyTorch AMP recipe": "https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html",
    "torch.amp": "https://docs.pytorch.org/docs/2.14/amp.html",
    "PyTorch AMP examples": "https://docs.pytorch.org/docs/2.14/notes/amp_examples.html",
    "PyTorch CUDA semantics": "https://docs.pytorch.org/docs/2.14/notes/cuda.html",
    "torch.utils.data": "https://docs.pytorch.org/docs/2.14/data.html",
    "torch.optim": "https://docs.pytorch.org/docs/2.14/optim.html",
    "torch.utils.checkpoint": "https://docs.pytorch.org/docs/2.14/checkpoint.html",
    "torch.compile tutorial": "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    "Pinned memory and non_blocking": "https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html",
    "Channels last memory format": "https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html",
    "Scaled dot product attention": "https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html",
    "NVIDIA GPU performance background": "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html",
    "NVIDIA mixed precision training": "https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html",
    "NVIDIA matrix multiplication background": "https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html",
    "NVIDIA convolutional layers": "https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html",
    "NVIDIA memory-limited layers": "https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html",
}  # fmt: skip

ISSUES_QUERY = 'repo:pytorch/pytorch is:issue is:closed label:"module: performance"'
MAX_ISSUES = 40  # each needs a comments request; unauthenticated GitHub allows 60 an hour

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "form", "aside", "button", "svg"}
HEADINGS = {"h1", "h2", "h3", "h4"}
BLOCKS = {"p", "li", "pre", "div", "tr", "dt", "dd", "br", "table", "blockquote"}


@dataclass
class Section:
    source: str
    heading: str
    url: str
    text: str


class _SectionParser(html.parser.HTMLParser):
    """Collect (heading, anchor, text) from a doc page's <article>, or <main> without one."""

    def __init__(self, root: str):
        super().__init__(convert_charrefs=True)
        self.root = root
        self.inside = 0
        self.skipping = 0
        self.section_ids: list[str | None] = []
        self.heading: list[str] | None = None
        self.anchor: str | None = None
        self.sections: list[tuple[str, str | None, list[str]]] = [("", None, [])]

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == self.root:
            self.inside += 1
        if not self.inside:
            return
        if tag in SKIP_TAGS:
            self.skipping += 1
        if tag == "section":
            self.section_ids.append(attrs.get("id"))
        if tag in HEADINGS and not self.skipping:
            self.heading = []
            self.anchor = attrs.get("id") or next(
                (i for i in reversed(self.section_ids) if i), None
            )
        elif tag == "a" and self.heading is not None:
            href = attrs.get("href") or ""
            self.anchor = attrs.get("id") or (href[1:] if href.startswith("#") else self.anchor)
        elif tag in BLOCKS:
            self.sections[-1][2].append("\n")

    def handle_endtag(self, tag):
        if not self.inside:
            return
        if tag in SKIP_TAGS and self.skipping:
            self.skipping -= 1
        if tag == "section" and self.section_ids:
            self.section_ids.pop()
        if tag in HEADINGS and self.heading is not None:
            title = re.sub(r"\s+", " ", "".join(self.heading)).strip(" #¶")
            self.sections.append((title, self.anchor, []))
            self.heading = None
        if tag == self.root:
            self.inside -= 1

    def handle_data(self, data):
        if not self.inside or self.skipping:
            return
        if self.heading is not None:
            self.heading.append(data)
        else:
            self.sections[-1][2].append(data)


def sections(source: str, url: str, page: str) -> list[Section]:
    parser = _SectionParser("article" if "<article" in page else "main")
    parser.feed(page)
    found = []
    for heading, anchor, parts in parser.sections:
        text = re.sub(r"[ \t]+", " ", "".join(parts))
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        if heading and len(text) > 40:
            found.append(Section(source, heading, f"{url}#{anchor}" if anchor else url, text))
    return found


def issue_sections(issue: dict, comments: list[dict]) -> list[Section]:
    """One section per issue: the report, then the last comments, where fixes usually land."""
    body = (issue.get("body") or "")[:4000]
    replies = "\n\n".join((c.get("body") or "")[:1500] for c in comments[-4:])
    text = f"{body}\n\nLast comments:\n{replies}" if replies else body
    return [Section("PyTorch GitHub issues", issue["title"], issue["html_url"], text)]


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "agentictorch-kb"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _cached(name: str, url: str) -> bytes:
    path = KB / "raw" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(url))
    return path.read_bytes()


def fetch_docs() -> list[Section]:
    found = []
    for source, url in DOCS.items():
        name = re.sub(r"\W+", "-", source.lower()) + ".html"
        found += sections(source, url, _cached(name, url).decode(errors="replace"))
    return found


def fetch_issues() -> list[Section]:
    params = {"q": ISSUES_QUERY, "sort": "reactions", "per_page": MAX_ISSUES}
    search = "https://api.github.com/search/issues?" + urllib.parse.urlencode(params)
    issues = json.loads(_cached("issues.json", search))["items"]
    found = []
    for issue in issues:
        name = f"issue-{issue['number']}-comments.json"
        fresh = not (KB / "raw" / name).exists()
        comments = json.loads(_cached(name, issue["comments_url"] + "?per_page=100"))
        found += issue_sections(issue, comments)
        if fresh:
            time.sleep(61)  # stay under GitHub's unauthenticated limit of 60 requests an hour
    return found


def main() -> None:
    found = fetch_docs() + fetch_issues()
    out = KB / "sections.jsonl"
    out.write_text("".join(json.dumps(asdict(s)) + "\n" for s in found))
    print(f"{len(found)} sections from {len({s.source for s in found})} sources -> {out}")


if __name__ == "__main__":
    main()
