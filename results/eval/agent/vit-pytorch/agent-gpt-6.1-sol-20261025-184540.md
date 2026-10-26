# AgenticTorch report: vit-pytorch

**Result: 2.43x faster, loss within noise** on an NVIDIA L4.

- Command: `python train_vit_decorr.py`
- Baseline: 26.4 ms/step over 2 runs
- Model: gpt-6.1-sol, 22 turns, 184752 input and 1141 output tokens

## Diagnosis

```
profiled 10 steps, 138.3 ms/step under the profiler
GPU busy 7% of the time
waiting on the DataLoader 38% of the time
host-device syncs per step: 4.0 (copies to the CPU: 2.0)
  2.0 per step at train_vit_decorr.py:94: for images, labels in dataloader:
  2.0 per step at train_vit_decorr.py:107: accelerator.print(f'loss: {loss.item():.3f} | decorr aux loss: {decorr_aux_loss.item():.3f}')
top ops by GPU time:
   8.4%  aten::mm  backward of vit_pytorch/vit_with_decorr.py:141: qkv = self.to_qkv(normed).chunk(3, dim = -1)
   7.1%  aten::mm  backward of vit_pytorch/vit_with_decorr.py:116: return self.net(normed), normed
   7.0%  aten::bmm  backward of vit_pytorch/vit_with_decorr.py:144: dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
   6.5%  aten::bmm  backward of vit_pytorch/vit_with_decorr.py:149: out = torch.matmul(attn, v)
   4.5%  aten::mul  backward of vit_pytorch/vit_with_decorr.py:94: loss = dist.pow(2) * (1. - eye) / ((dim - 1) * dim)
   4.3%  aten::addmm  at vit_pytorch/vit_with_decorr.py:116: return self.net(normed), normed
   4.1%  aten::mm  at vit_pytorch/vit_with_decorr.py:141: qkv = self.to_qkv(normed).chunk(3, dim = -1)
   3.9%  aten::mm  backward of vit_pytorch/vit_with_decorr.py:152: return self.to_out(out), normed
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Overlap CIFAR preprocessing with training using four persistent DataLoader workers and pinned host memory. | 26.2 ms/step | 20.3 ms/step | kept: 1.29x faster, loss within noise |
| 2 | Use PyTorch SDPA for equivalent attention with optimized kernels and fewer intermediate tensors. | 21.5 ms/step | 17.0 ms/step | kept: 1.27x faster, loss within noise |
| 3 | Compile the model in reduce-overhead mode to fuse pointwise/decorrelation operations and reduce kernel-launch overhead. | 17.0 ms/step | 11.3 ms/step | kept: 1.51x faster, loss within noise |
| 4 | Use fused Adam to consolidate optimizer updates while retaining the same Adam hyperparameters. | 10.9 ms/step | 10.8 ms/step | reverted: 1.01x is within noise |
| 5 | Enable nonblocking Accelerate batch transfers from the pinned DataLoader to avoid two host-device synchronizations per step. | 10.6 ms/step | 10.3 ms/step | reverted: 1.03x is within noise |

## Diff of the kept changes

```diff
diff --git a/train_vit_decorr.py b/train_vit_decorr.py
index 23b4051..0e89800 100644
--- a/train_vit_decorr.py
+++ b/train_vit_decorr.py
@@ -41,7 +41,10 @@ dataset = CIFAR100(
     transform = transform
 )
 
-dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
+dataloader = DataLoader(
+    dataset, batch_size = BATCH_SIZE, shuffle = True,
+    num_workers = 4, pin_memory = True, persistent_workers = True
+)
 
 # model
 
@@ -76,6 +79,7 @@ from accelerate import Accelerator
 accelerator = Accelerator()
 
 vit, optim, dataloader = accelerator.prepare(vit, optim, dataloader)
+vit = torch.compile(vit, mode = 'reduce-overhead')
 
 # experiment
 
diff --git a/vit_pytorch/vit_with_decorr.py b/vit_pytorch/vit_with_decorr.py
index 401d8fd..233c280 100644
--- a/vit_pytorch/vit_with_decorr.py
+++ b/vit_pytorch/vit_with_decorr.py
@@ -141,12 +141,11 @@ class Attention(Module):
         qkv = self.to_qkv(normed).chunk(3, dim = -1)
         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
 
-        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
-
-        attn = self.attend(dots)
-        attn = self.dropout(attn)
-
-        out = torch.matmul(attn, v)
+        out = F.scaled_dot_product_attention(
+            q, k, v,
+            dropout_p = self.dropout.p if self.training else 0.,
+            scale = self.scale
+        )
         out = rearrange(out, 'b h n d -> b n (h d)')
 
         return self.to_out(out), normed
```

## Sources the agent read

- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [torch.utils.data: Memory Pinning](https://docs.pytorch.org/docs/2.14/data.html#memory-pinning)
- [Pinned memory and non_blocking: Conclusion](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#conclusion)
- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [Scaled dot product attention: Using SDPA with attn_bias subclasses](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-attn-bias-subclasses)
- [Scaled dot product attention: Fused implementations](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#fused-implementations)
- [Scaled dot product attention: Overview](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#overview)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [PyTorch CUDA semantics: Whole-network capture](https://docs.pytorch.org/docs/2.14/notes/cuda.html#whole-network-capture)
- [PyTorch performance tuning guide: Use CUDA Graphs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [PyTorch CUDA semantics: TensorFloat-32 (TF32) on Ampere (and later) devices](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [torch.amp: CUDA Ops that promote to the widest input type](https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-promote-to-the-widest-input-type)
- [PyTorch CUDA semantics: Use pinned memory buffers](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)
