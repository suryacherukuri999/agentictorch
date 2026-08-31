"""Train a character-level GPT on tiny Shakespeare."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, block: int, layers: int, dim: int, heads: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(block, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.blocks(self.token(idx) + self.position(positions))
        return self.head(self.norm(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(args.data).read_text()
    chars = sorted(set(text))
    encoded = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    split = int(0.9 * len(encoded))
    # The whole dataset is about 1 MB, so it lives on the GPU and batches never touch the CPU.
    train, val = encoded[:split].to(device), encoded[split:].to(device)

    def batch(data):
        starts = torch.randint(len(data) - args.block - 1, (args.batch_size,), device=device)
        offsets = torch.arange(args.block, device=device)
        x = data[starts[:, None] + offsets]
        y = data[starts[:, None] + offsets + 1]
        return x, y

    model = GPT(len(chars), args.block, args.layers, args.dim, args.heads, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1, fused=device == "cuda"
    )
    for step in range(1, args.steps + 1):
        x, y = batch(train)
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                x, y = batch(val)
                val_loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            model.train()
            print(f"step {step} train {loss.item():.3f} val {val_loss.item():.3f}")


if __name__ == "__main__":
    main()
