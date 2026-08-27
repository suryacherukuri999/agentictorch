"""Hybrid search over the knowledge base: BM25 and embeddings, fused, then reranked.

Build the index once after fetching: python -m hotloop.kb.search build
Both models are small and run locally, so search costs nothing.
"""

import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from hotloop.kb.fetch import KB

EMBEDDER = "BAAI/bge-small-en-v1.5"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
# BGE models embed queries with this prefix and passages without one.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 20


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    url: str
    text: str

    @property
    def searchable(self) -> str:
        return f"{self.source}: {self.heading}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    # Keeps identifiers like num_workers and torch.compile whole.
    return [t.strip(".") for t in re.findall(r"[a-z0-9_.]+", text.lower()) if t.strip(".")]


def chunk_section(section: dict, words: int = 180, overlap: int = 30) -> list[Chunk]:
    tokens = section["text"].split(" ")
    step = words - overlap
    pieces = [
        " ".join(tokens[i : i + words]) for i in range(0, max(len(tokens) - overlap, 1), step)
    ]
    return [Chunk(section["source"], section["heading"], section["url"], p) for p in pieces]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_length = sum(self.lengths) / len(docs)
        frequency = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for counts, length in zip(self.counts, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out.append(
                sum(
                    self.idf[t] * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                    for t in query
                    if t in counts
                )
            )
        return out


def best_per_url(hits: list[Hit]) -> list[Hit]:
    """Keep the best-ranked chunk of each section, so the results cover more ground."""
    seen = set()
    return [h for h in hits if not (h.chunk.url in seen or seen.add(h.chunk.url))]


def fuse(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: merge rankings of document ids."""
    scores = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] += 1 / (k + rank + 1)
    return [doc for doc, _ in scores.most_common()]


class Index:
    def __init__(self):
        lines = (KB / "index" / "chunks.jsonl").read_text().splitlines()
        self.chunks = [Chunk(**json.loads(line)) for line in lines]
        self.embeddings = np.load(KB / "index" / "embeddings.npy")
        self.bm25 = BM25([tokenize(c.searchable) for c in self.chunks])

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDER)

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANKER)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        lexical = np.argsort(self.bm25.scores(tokenize(query)))[::-1][:50]
        vector = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        dense = np.argsort(self.embeddings @ vector)[::-1][:50]
        candidates = fuse(lexical.tolist(), dense.tolist())[:CANDIDATES]
        scores = self.reranker.predict([(query, self.chunks[i].searchable) for i in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -p[1])
        return best_per_url([Hit(self.chunks[i], float(s)) for i, s in ranked])[:k]


def build() -> None:
    from sentence_transformers import SentenceTransformer

    lines = (KB / "sections.jsonl").read_text().splitlines()
    chunks = [c for line in lines for c in chunk_section(json.loads(line))]
    model = SentenceTransformer(EMBEDDER)
    embeddings = model.encode(
        [c.searchable for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    out = KB / "index"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in chunks))
    np.save(out / "embeddings.npy", embeddings)
    print(f"indexed {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        for hit in Index().search(" ".join(sys.argv[1:])):
            print(f"{hit.score:6.2f}  {hit.chunk.source}: {hit.chunk.heading}")
            print(f"        {hit.chunk.url}")
