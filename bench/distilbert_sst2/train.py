"""DistilBERT fine-tuned on SST-2, written to be fast. The planted cases each break one thing."""

import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased"


class Sentences(torch.utils.data.Dataset):
    """Tokenizes each sentence as it is loaded, like many text pipelines do."""

    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        encoded = self.tokenizer(
            row["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0], row["label"]


def evaluate(model, loader, device, batches):
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i, (ids, mask, labels) in enumerate(loader):
            if i == batches:
                break
            ids, mask, labels = (t.to(device, non_blocking=True) for t in (ids, mask, labels))
            logits = model(input_ids=ids, attention_mask=mask).logits
            correct += (logits.argmax(1) == labels).sum()
            total += len(labels)
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sst2 = load_dataset("nyu-mll/glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    loader = torch.utils.data.DataLoader(
        Sentences(sst2["train"], tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        Sentences(sst2["validation"], tokenizer, args.max_length),
        batch_size=256,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, fused=device == "cuda")
    total_steps = args.epochs * len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 - s / total_steps)
    step = 0
    for epoch in range(args.epochs):
        for ids, mask, labels in loader:
            ids, mask, labels = (t.to(device, non_blocking=True) for t in (ids, mask, labels))
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            step += 1
            if step % args.eval_every == 0:
                accuracy = evaluate(model, val_loader, device, args.eval_batches)
                print(f"epoch {epoch} step {step} loss {loss.item():.3f} val {accuracy.item():.3f}")


if __name__ == "__main__":
    main()
