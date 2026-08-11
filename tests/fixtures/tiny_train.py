"""A tiny CPU training script for testing the harness."""

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Noise(Dataset):
    def __init__(self, n: int, delay: float):
        self.x = torch.randn(n, 16)
        self.y = torch.randn(n, 1)
        self.delay = delay

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.delay:
            time.sleep(self.delay)
        return self.x[i], self.y[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--two-optimizers", action="store_true")
    parser.add_argument("--rebuild-optimizer", action="store_true")
    parser.add_argument("--scaler", action="store_true")
    parser.add_argument("--crash-at", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    data = DataLoader(Noise(256, args.delay), batch_size=16, shuffle=True, num_workers=args.workers)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    critic = nn.Linear(16, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=args.scaler)

    step = 0
    for _ in range(args.epochs):
        if args.rebuild_optimizer:
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for x, y in data:
            step += 1
            if step == args.crash_at:
                raise RuntimeError("planted crash")
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if args.two_optimizers:
                critic_loss = critic(x).pow(2).mean()
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
    print("finished", step)


if __name__ == "__main__":
    main()
