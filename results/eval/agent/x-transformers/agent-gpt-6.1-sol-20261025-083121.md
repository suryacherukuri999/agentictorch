# AgenticTorch report: x-transformers

**Result: 5.52x faster, loss within noise** on an NVIDIA L4.

- Command: `python train_enwik8.py`
- Baseline: 694.9 ms/step over 2 runs
- Model: gpt-6.1-sol, 26 turns, 211505 input and 1433 output tokens

## Diagnosis

```
profiled 10 steps, 768.3 ms/step under the profiler
GPU busy 88% of the time
waiting on the DataLoader 30% of the time
host-device syncs per step: 18.0 (copies to the CPU: 2.0)
  16.0 per step at train_enwik8.py:102: return full_seq.to(device)
  1.0 per step at train_enwik8.py:134: print(f'training loss: {loss.item()}')
  1.0 per step at train_enwik8.py:136: wandb.log(dict(loss = loss.item()))
top ops by GPU time:
  10.0%  aten::mm  backward of x_transformers/x_transformers.py:1853: out = self.ff(x)
   6.4%  aten::_softmax_backward_data  backward of x_transformers/attend.py:621: attn = self.attn_fn(sim)
   6.1%  aten::mul  backward of x_transformers/attend.py:621: attn = self.attn_fn(sim)
   6.1%  aten::bmm  backward of x_transformers/attend.py:552: sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
   5.1%  aten::addmm  at x_transformers/x_transformers.py:1853: out = self.ff(x)
   5.0%  aten::bmm  backward of x_transformers/attend.py:655: out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
   4.2%  aten::masked_fill_  at x_transformers/attend.py:596: sim = sim.masked_fill(causal_mask, mask_value)
   4.2%  aten::masked_fill_  backward of x_transformers/attend.py:596: sim = sim.masked_fill(causal_mask, mask_value)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Keep dataset samples on CPU, pin collated batches, and transfer once per batch nonblocking instead of once per sample. | 706.5 ms/step | 703.2 ms/step | reverted: 1.00x is within noise |
| 2 | Enable the existing SDPA attention implementation to fuse attention and avoid quadratic intermediate tensors. | 697.2 ms/step | 391.9 ms/step | kept: 1.78x faster, loss within noise |
| 3 | Use Accelerate bfloat16 mixed precision to enable L4 Tensor Cores and fused low-precision attention, retaining FP32 parameters. | 402.3 ms/step | 183.0 ms/step | kept: 2.20x faster, loss within noise |
| 4 | Compile the model forward with torch.compile to fuse pointwise work and reduce Python/kernel-launch overhead. | 179.3 ms/step | 130.8 ms/step | kept: 1.37x faster, loss within noise |
| 5 | Use fused CUDA Adam with unchanged optimizer hyperparameters to reduce optimizer kernel launches and memory traffic. | 131.9 ms/step | 127.8 ms/step | kept: 1.03x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/train_enwik8.py b/train_enwik8.py
index f607a4e..d619189 100644
--- a/train_enwik8.py
+++ b/train_enwik8.py
@@ -58,7 +58,7 @@ def train(
     read_gate_rank = 64,
     attn_aggregated_residuals = False
 ):
-    accelerator = Accelerator(cpu=cpu)
+    accelerator = Accelerator(cpu=cpu, mixed_precision='no' if cpu else 'bf16')
     device = accelerator.device
 
     generate_length = default(generate_length, seq_len)
@@ -72,6 +72,7 @@ def train(
             dim = 512,
             depth = 6,
             heads = 8,
+            attn_flash = True,
             rotary_pos_emb = False,
             polar_pos_emb = True,
             pre_and_post_norm = not (gated_multi_residual or attn_aggregated_residuals),
@@ -82,6 +83,8 @@ def train(
     )
 
     model = AutoregressiveWrapper(model)
+    if device.type == 'cuda':
+        model.compile()
 
     # prepare enwik8 data
 
@@ -111,7 +114,7 @@ def train(
 
     # optimizer
 
-    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
+    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, fused = device.type == 'cuda')
 
     # experiment
```

## Sources the agent read

- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [torch.utils.data: Memory Pinning](https://docs.pytorch.org/docs/2.14/data.html#memory-pinning)
- [PyTorch CUDA semantics: Use pinned memory buffers](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)
- [torch.utils.data: Multi-process data loading](https://docs.pytorch.org/docs/2.14/data.html#multi-process-data-loading)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [Scaled dot product attention: Using SDPA with attn_bias subclasses](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-attn-bias-subclasses)
- [Scaled dot product attention: NestedTensor and Dense tensor support](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#nestedtensor-and-dense-tensor-support)
- [Scaled dot product attention: Fused implementations](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#fused-implementations)
- [PyTorch AMP examples: Typical Mixed Precision Training](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#typical-mixed-precision-training)
- [NVIDIA mixed precision training: 7.1.3. Tensor Core Optimized Model Scripts For PyTorch](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#pytorch-tensor-core-ex)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [torch.amp: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html#automatic-mixed-precision-package-torch-amp)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [PyTorch GitHub issues: Subclassed Tensors Decrease Training GPU Throughput by ~40% ](https://github.com/pytorch/pytorch/issues/94160)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [PyTorch AMP recipe: Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#automatic-mixed-precision)
- [torch.amp: CUDA Ops that can autocast to float16](https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-can-autocast-to-float16)
