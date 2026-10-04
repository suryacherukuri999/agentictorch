"""Retrieval check: does each symptom query find the section that explains its fix?

python -m agentictorch.kb.evaluate
"""

from agentictorch.kb.search import Index

# (query as the agent might phrase it, words that must appear in "source: heading" of a hit)
QUERIES = [
    ("DataLoader num_workers=0 and the GPU waits for data", "asynchronous data loading"),
    ("loss.item() called every step slows training down", "synchronization"),
    ("pin_memory and non_blocking copies to the GPU", "pin"),
    ("train with bf16 autocast mixed precision", "mixed precision"),
    ("GradScaler loss scaling for fp16 training", "scal"),
    ("fused AdamW optimizer kernel is faster", "fused"),
    ("replace hand-written attention softmax with a fused attention kernel", "attention"),
    ("cudnn.benchmark for convolution networks with fixed input sizes", "cudnn"),
    ("torch.cuda.empty_cache() called in the training loop", "memory management"),
    ("validation loop without torch.no_grad uses extra memory", "gradient calculation"),
    ("out of memory: trade compute for memory by recomputing activations", "checkpoint"),
    ("channels_last memory format for convolutional networks", "channels"),
    ("torch.compile to fuse pointwise operations", "fuse"),
    ("zero_grad set_to_none instead of zeroing gradients", "zero"),
    ("TF32 for float32 matmuls on Ampere GPUs", "tf32"),
]


def hits(index: Index, k: int = 4) -> list[tuple[str, bool, str]]:
    out = []
    for query, expected in QUERIES:
        found = index.search(query, k=k)
        titles = [f"{h.chunk.source}: {h.chunk.heading}" for h in found]
        out.append((query, any(expected in t.lower() for t in titles), titles[0] if titles else ""))
    return out


if __name__ == "__main__":
    results = hits(Index())
    for query, ok, top in results:
        print(f"{'hit ' if ok else 'MISS'}  {query}  ->  {top}")
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} found in the top 4")
