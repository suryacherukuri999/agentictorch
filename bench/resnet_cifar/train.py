"""Train ResNet-18 on CIFAR-10."""

import argparse

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def resnet18() -> nn.Module:
    model = torchvision.models.resnet18(num_classes=10)
    # CIFAR images are 32x32: a 3x3 stem and no max-pool keep the spatial detail.
    model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def evaluate(model, loader, device, batches):
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i, (x, y) in enumerate(loader):
            if i == batches:
                break
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum()
            total += len(y)
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    augment = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR10(
        args.data, train=True, download=True, transform=augment
    )
    test_set = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=plain)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = resnet18().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4
    )
    step = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.3f}")
            if step % args.eval_every == 0:
                accuracy = evaluate(model, test_loader, device, args.eval_batches)
                print(f"step {step} test accuracy {accuracy.item():.3f}")


if __name__ == "__main__":
    main()
