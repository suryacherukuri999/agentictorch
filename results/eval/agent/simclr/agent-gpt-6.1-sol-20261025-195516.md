# AgenticTorch report: simclr

**Result: 2.39x faster, loss within noise** on an NVIDIA L4.

- Command: `python main.py`
- Baseline: 418.9 ms/step over 2 runs
- Model: gpt-6.1-sol, 22 turns, 171591 input and 1277 output tokens

## Diagnosis

```
profiled 10 steps, 425.6 ms/step under the profiler
GPU busy 96% of the time
waiting on the DataLoader 0% of the time
host-device syncs per step: 5.0 (copies to the CPU: 2.0)
  2.0 per step at main.py:37: loss.backward()
  1.0 per step at main.py:49: writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
  1.0 per step at main.py:52: loss_epoch += loss.item()
  1.0 per step at simclr/modules/nt_xent.py:48: labels = torch.zeros(N).to(positive_samples.device).long()
top ops by GPU time:
  17.7%  aten::convolution_backward  backward of simclr/simclr.py:30: h_i = self.encoder(x_i)
  17.5%  aten::convolution_backward  backward of simclr/simclr.py:31: h_j = self.encoder(x_j)
   6.1%  aten::cudnn_batch_norm_backward  backward of simclr/simclr.py:31: h_j = self.encoder(x_j)
   6.1%  aten::cudnn_batch_norm_backward  backward of simclr/simclr.py:30: h_i = self.encoder(x_i)
   6.1%  aten::copy_  at main.py:30: x_i = x_i.cuda(non_blocking=True)
   6.0%  aten::cudnn_convolution  at simclr/simclr.py:31: h_j = self.encoder(x_j)
   5.8%  aten::cudnn_convolution  at simclr/simclr.py:30: h_i = self.encoder(x_i)
   5.5%  aten::copy_  at main.py:31: x_j = x_j.cuda(non_blocking=True)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Use bfloat16 autocast for encoder/projector while keeping NT-Xent loss in float32. | 420.4 ms/step | 262.8 ms/step | kept: 1.60x faster, loss within noise |
| 2 | Use channels-last layout for convolution model and input transfers to improve Tensor Core convolution throughput. | 247.9 ms/step | 197.4 ms/step | kept: 1.26x faster, loss within noise |
| 3 | Pin DataLoader host batches to accelerate the existing nonblocking host-to-device input transfers. | 205.7 ms/step | 177.4 ms/step | kept: 1.16x faster, loss within noise |
| 4 | Compile the model in place to fuse pointwise operations and reduce training launch overhead without changing checkpoint keys. | 176.0 ms/step | 173.4 ms/step | reverted: 1.02x is within noise |
| 5 | Use fused Adam with identical hyperparameters to reduce optimizer kernel and memory overhead. | 175.7 ms/step | 173.8 ms/step | reverted: 1.01x is within noise |

## Diff of the kept changes

```diff
diff --git a/main.py b/main.py
index fec4778..3d7f6a1 100644
--- a/main.py
+++ b/main.py
@@ -27,13 +27,14 @@ def train(args, train_loader, model, criterion, optimizer, writer):
     loss_epoch = 0
     for step, ((x_i, x_j), _) in enumerate(train_loader):
         optimizer.zero_grad()
-        x_i = x_i.cuda(non_blocking=True)
-        x_j = x_j.cuda(non_blocking=True)
+        x_i = x_i.to(device=args.device, non_blocking=True, memory_format=torch.channels_last)
+        x_j = x_j.to(device=args.device, non_blocking=True, memory_format=torch.channels_last)
 
         # positive pair, with encoding
-        h_i, h_j, z_i, z_j = model(x_i, x_j)
+        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
+            h_i, h_j, z_i, z_j = model(x_i, x_j)
 
-        loss = criterion(z_i, z_j)
+        loss = criterion(z_i.float(), z_j.float())
         loss.backward()
 
         optimizer.step()
@@ -92,6 +93,7 @@ def main(gpu, args):
         shuffle=(train_sampler is None),
         drop_last=True,
         num_workers=args.workers,
+        pin_memory=True,
         sampler=train_sampler,
     )
 
@@ -106,7 +108,7 @@ def main(gpu, args):
             args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
         )
         model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
-    model = model.to(args.device)
+    model = model.to(device=args.device, memory_format=torch.channels_last)
 
     # optimizer / loss
     optimizer, scheduler = load_optimizer(args, model)
```

## Sources the agent read

- [Channels last memory format: Performance Gains](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains)
- [Channels last memory format: Memory Format API](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api)
- [Channels last memory format: Converting existing models](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models)
- [Channels last memory format: Channels Last Memory Format in PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#channels-last-memory-format-in-pytorch)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [PyTorch CUDA semantics: Use pinned memory buffers](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)
- [Pinned memory and non_blocking: Asynchronous vs. Synchronous Operations with non_blocking=True (CUDA cudaMemcpyAsync)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Graph Breaks](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#graph-breaks)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP recipe: Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#automatic-mixed-precision)
- [torch.amp: CPU Ops that promote to the widest input type](https://docs.pytorch.org/docs/2.14/amp.html#cpu-ops-that-promote-to-the-widest-input-type)
