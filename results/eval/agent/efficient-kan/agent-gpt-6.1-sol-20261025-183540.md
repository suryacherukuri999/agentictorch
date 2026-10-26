# AgenticTorch report: efficient-kan

**Result: 3.58x faster, loss within noise** on an NVIDIA L4.

- Command: `PYTHONPATH=src:$PYTHONPATH python examples/mnist.py`
- Baseline: 19.8 ms/step over 2 runs
- Model: gpt-6.1-sol, 24 turns, 200614 input and 1613 output tokens

## Diagnosis

```
profiled 10 steps, 129.7 ms/step under the profiler
GPU busy 0% of the time
waiting on the DataLoader 83% of the time
host-device syncs per step: 5.0 (copies to the CPU: 2.0)
  2.0 per step at examples/mnist.py:48: pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
  1.0 per step at examples/mnist.py:47: accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
  1.0 per step at examples/mnist.py:41: images = images.view(-1, 28 * 28).to(device)
  1.0 per step at examples/mnist.py:44: loss = criterion(output, labels.to(device))
top ops by GPU time:
   5.4%  aten::_foreach_addcdiv_  at examples/mnist.py:46: optimizer.step()
   5.0%  aten::_foreach_div_  at examples/mnist.py:46: optimizer.step()
   4.2%  aten::div  at src/efficient_kan/kan.py:97: (x - grid[:, : -(k + 1)])
   4.2%  aten::div  at src/efficient_kan/kan.py:101: (grid[:, k + 1 :] - x)
   3.8%  aten::copy_  at examples/mnist.py:41: images = images.view(-1, 28 * 28).to(device)
   3.6%  aten::sub  at src/efficient_kan/kan.py:97: (x - grid[:, : -(k + 1)])
   3.6%  aten::mm  backward of src/efficient_kan/kan.py:159: spline_output = F.linear(
   3.6%  aten::mm  at src/efficient_kan/kan.py:159: spline_output = F.linear(
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Overlap unchanged MNIST transforms with training using four persistent DataLoader workers. | 17.5 ms/step | 7.1 ms/step | kept: 2.46x faster, loss within noise |
| 2 | Compile the unchanged KAN model in reduce-overhead mode to fuse spline operations and reduce launch overhead. | 5.9 ms/step | 4.7 ms/step | kept: 1.28x faster, loss within noise |
| 3 | Use pinned DataLoader batches and nonblocking transfers; transfer labels once per batch rather than twice. | 4.6 ms/step | 4.3 ms/step | kept: 1.07x faster, loss within noise |
| 4 | Use fused CUDA AdamW with the same optimizer hyperparameters to reduce optimizer kernel launches. | 4.6 ms/step | 4.5 ms/step | reverted: 1.01x is within noise |
| 5 | Compile the existing AdamW step to fuse optimizer arithmetic without changing optimizer settings or logged metrics. | 4.5 ms/step | 4.9 ms/step | reverted: 0.92x is within noise |

## Diff of the kept changes

```diff
diff --git a/examples/mnist.py b/examples/mnist.py
index 0e10c0c..80910d6 100644
--- a/examples/mnist.py
+++ b/examples/mnist.py
@@ -19,13 +19,21 @@ trainset = torchvision.datasets.MNIST(
 valset = torchvision.datasets.MNIST(
     root="./data", train=False, download=True, transform=transform
 )
-trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
-valloader = DataLoader(valset, batch_size=64, shuffle=False)
+trainloader = DataLoader(
+    trainset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True,
+    pin_memory=torch.cuda.is_available()
+)
+valloader = DataLoader(
+    valset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True,
+    pin_memory=torch.cuda.is_available()
+)
 
 # Define model
 model = KAN([28 * 28, 64, 10])
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model.to(device)
+if device.type == "cuda":
+    model = torch.compile(model, mode="reduce-overhead")
 # Define optimizer
 optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
 # Define learning rate scheduler
@@ -38,13 +46,14 @@ for epoch in range(10):
     model.train()
     with tqdm(trainloader) as pbar:
         for i, (images, labels) in enumerate(pbar):
-            images = images.view(-1, 28 * 28).to(device)
+            images = images.view(-1, 28 * 28).to(device, non_blocking=True)
+            labels = labels.to(device, non_blocking=True)
             optimizer.zero_grad()
             output = model(images)
-            loss = criterion(output, labels.to(device))
+            loss = criterion(output, labels)
             loss.backward()
             optimizer.step()
-            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
+            accuracy = (output.argmax(dim=1) == labels).float().mean()
             pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
 
     # Validation
@@ -53,11 +62,12 @@ for epoch in range(10):
     val_accuracy = 0
     with torch.no_grad():
         for images, labels in valloader:
-            images = images.view(-1, 28 * 28).to(device)
+            images = images.view(-1, 28 * 28).to(device, non_blocking=True)
+            labels = labels.to(device, non_blocking=True)
             output = model(images)
-            val_loss += criterion(output, labels.to(device)).item()
+            val_loss += criterion(output, labels).item()
             val_accuracy += (
-                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
+                (output.argmax(dim=1) == labels).float().mean().item()
             )
     val_loss /= len(valloader)
     val_accuracy /= len(valloader)
```

## Sources the agent read

- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [torch.utils.data: torch.utils.data](https://docs.pytorch.org/docs/2.14/data.html#module-torch.utils.data)
- [torch.utils.data: Memory Pinning](https://docs.pytorch.org/docs/2.14/data.html#memory-pinning)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [PyTorch performance tuning guide: Fuse operations](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [PyTorch performance tuning guide: Use CUDA Graphs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch CUDA semantics: TensorFloat-32 (TF32) on Ampere (and later) devices](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
- [PyTorch AMP examples: Functions that need a particular dtype](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#functions-that-need-a-particular-dtype)
- [torch.optim: How to adjust learning rate](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile)
