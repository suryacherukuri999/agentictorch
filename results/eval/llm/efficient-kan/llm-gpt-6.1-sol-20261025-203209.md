# AgenticTorch report: efficient-kan

**Result: 2.17x faster, loss within noise** on an NVIDIA L4.

- Command: `PYTHONPATH=src:$PYTHONPATH python examples/mnist.py`
- Baseline: 17.4 ms/step over 2 runs
- Model: gpt-6.1-sol, 5 turns, 16291 input and 2038 output tokens

Optimized examples/mnist.py with two persistent DataLoader workers, pinned memory and nonblocking CUDA transfers, a single label transfer per batch, supported-CUDA bfloat16 autocast, CUDA torch.compile, fused CUDA AdamW, and explicit set_to_none gradient clearing. Combined training metric transfers into one synchronization while retaining every progress update; accumulated validation metrics on-device in float64 and transferred once per epoch, preserving batch-mean weighting. Added a main guard for worker-safe startup. Model architecture, data, batch size, learning rate, epoch count, and printed fields remain unchanged. No execution or benchmarking performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 17.4 ms/step | 8.0 ms/step | kept: 2.17x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/examples/mnist.py b/examples/mnist.py
index 0e10c0c..970f1a8 100644
--- a/examples/mnist.py
+++ b/examples/mnist.py
@@ -9,62 +9,77 @@ import torchvision.transforms as transforms
 from torch.utils.data import DataLoader
 from tqdm import tqdm
 
-# Load MNIST
-transform = transforms.Compose(
-    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
-)
-trainset = torchvision.datasets.MNIST(
-    root="./data", train=True, download=True, transform=transform
-)
-valset = torchvision.datasets.MNIST(
-    root="./data", train=False, download=True, transform=transform
-)
-trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
-valloader = DataLoader(valset, batch_size=64, shuffle=False)
+def main():
+    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+    use_cuda = device.type == "cuda"
+    use_amp = use_cuda and torch.cuda.is_bf16_supported()
 
-# Define model
-model = KAN([28 * 28, 64, 10])
-device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-model.to(device)
-# Define optimizer
-optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
-# Define learning rate scheduler
-scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
+    # Load MNIST. Persistent workers overlap preprocessing with training.
+    transform = transforms.Compose(
+        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
+    )
+    trainset = torchvision.datasets.MNIST(
+        root="./data", train=True, download=True, transform=transform
+    )
+    valset = torchvision.datasets.MNIST(
+        root="./data", train=False, download=True, transform=transform
+    )
+    loader_kwargs = dict(num_workers=2, persistent_workers=True, pin_memory=use_cuda)
+    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, **loader_kwargs)
+    valloader = DataLoader(valset, batch_size=64, shuffle=False, **loader_kwargs)
+
+    # Define model and optimizer before compiling; parameter objects are unchanged.
+    model = KAN([28 * 28, 64, 10]).to(device)
+    optimizer = optim.AdamW(
+        model.parameters(), lr=1e-3, weight_decay=1e-4, fused=use_cuda
+    )
+    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
+    if use_cuda:
+        model = torch.compile(model)
 
-# Define loss
-criterion = nn.CrossEntropyLoss()
-for epoch in range(10):
-    # Train
-    model.train()
-    with tqdm(trainloader) as pbar:
-        for i, (images, labels) in enumerate(pbar):
-            images = images.view(-1, 28 * 28).to(device)
-            optimizer.zero_grad()
-            output = model(images)
-            loss = criterion(output, labels.to(device))
-            loss.backward()
-            optimizer.step()
-            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
-            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
+    criterion = nn.CrossEntropyLoss()
+    for epoch in range(10):
+        # Train
+        model.train()
+        with tqdm(trainloader) as pbar:
+            for i, (images, labels) in enumerate(pbar):
+                images = images.view(-1, 28 * 28).to(device, non_blocking=True)
+                labels = labels.to(device, non_blocking=True)
+                optimizer.zero_grad(set_to_none=True)
+                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
+                    output = model(images)
+                    loss = criterion(output, labels)
+                loss.backward()
+                optimizer.step()
+                accuracy = (output.argmax(dim=1) == labels).float().mean()
+                # Preserve every progress update, but transfer both scalars together.
+                loss_value, accuracy_value = torch.stack((loss.detach(), accuracy)).tolist()
+                pbar.set_postfix(loss=loss_value, accuracy=accuracy_value, lr=optimizer.param_groups[0]['lr'])
 
-    # Validation
-    model.eval()
-    val_loss = 0
-    val_accuracy = 0
-    with torch.no_grad():
-        for images, labels in valloader:
-            images = images.view(-1, 28 * 28).to(device)
-            output = model(images)
-            val_loss += criterion(output, labels.to(device)).item()
-            val_accuracy += (
-                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
-            )
-    val_loss /= len(valloader)
-    val_accuracy /= len(valloader)
+        # Validation: retain the original mean of batch means, accumulating in
+        # float64 just as Python's float additions did, without per-batch syncs.
+        model.eval()
+        val_metrics = torch.zeros(2, dtype=torch.float64, device=device)
+        with torch.no_grad():
+            for images, labels in valloader:
+                images = images.view(-1, 28 * 28).to(device, non_blocking=True)
+                labels = labels.to(device, non_blocking=True)
+                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
+                    output = model(images)
+                    loss = criterion(output, labels)
+                accuracy = (output.argmax(dim=1) == labels).float().mean()
+                val_metrics += torch.stack((loss, accuracy)).to(torch.float64)
+        val_loss, val_accuracy = val_metrics.tolist()
+        val_loss /= len(valloader)
+        val_accuracy /= len(valloader)
 
-    # Update learning rate
-    scheduler.step()
+        # Update learning rate
+        scheduler.step()
 
-    print(
-        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
-    )
+        print(
+            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
+        )
+
+
+if __name__ == "__main__":
+    main()
```
