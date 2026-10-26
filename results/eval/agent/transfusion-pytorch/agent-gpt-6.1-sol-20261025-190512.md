# AgenticTorch report: transfusion-pytorch

**Result: 1.58x faster, loss within noise** on an NVIDIA L4.

- Command: `python train_image_only.py`
- Baseline: 83.2 ms/step over 2 runs
- Model: gpt-6.1-sol, 23 turns, 287289 input and 1553 output tokens

## Diagnosis

```
profiled 10 steps, 431.1 ms/step under the profiler
GPU busy 10% of the time
waiting on the DataLoader 5% of the time
host-device syncs per step: 4.0 (copies to the CPU: 3.0)
  2.0 per step at train_image_only.py:110: ema_model.update()
  1.0 per step at train_image_only.py:112: accelerator.print(f'{step}: {loss.item():.3f}')
  1.0 per step at transfusion_pytorch/transfusion.py:2839: modalities = modalities.to(self.device)
top ops by GPU time:
   6.1%  aten::div  at transfusion_pytorch/transfusion.py:315: return (t / value).tanh() * value
   5.9%  aten::_softmax  at transfusion_pytorch/transfusion.py:1109: attn = sim.softmax(dim = -1)
   5.7%  aten::tanh  at transfusion_pytorch/transfusion.py:315: return (t / value).tanh() * value
   5.7%  aten::mul  at transfusion_pytorch/transfusion.py:315: return (t / value).tanh() * value
   4.9%  aten::bmm  at transfusion_pytorch/transfusion.py:1113: out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
   4.8%  aten::tanh_backward  backward of transfusion_pytorch/transfusion.py:315: return (t / value).tanh() * value
   4.7%  aten::_softmax_backward_data  backward of transfusion_pytorch/transfusion.py:1109: attn = sim.softmax(dim = -1)
   4.6%  aten::mul  backward of transfusion_pytorch/transfusion.py:1109: attn = sim.softmax(dim = -1)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Compile student and EMA transformers in place to fuse pointwise operations and reduce Python/kernel launch overhead. | 98.1 ms/step | 62.9 ms/step | kept: 1.56x faster, loss within noise |
| 2 | Use existing FlexAttention implementation to fuse softcapped attention while retaining LASER value transform and all model parameters. | 62.5 ms/step | 62.1 ms/step | reverted: 1.01x is within noise |
| 3 | Enable Accelerate bfloat16 mixed precision for faster tensor-core matrix multiplications, leaving parameters and optimizer state in float32. | 62.7 ms/step | 64.7 ms/step | reverted: 0.97x is within noise |
| 4 | Use reduce-overhead transformer compilation to enable CUDA graph replay and lower host dispatch overhead. | 56.0 ms/step | failed (Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.) | reverted: run failed: Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. |
| 5 | Keep EMA step and initialized bookkeeping buffers on CPU to avoid two device-to-host synchronization points per update without changing EMA scheduling or weights. | 63.9 ms/step | 60.2 ms/step | reverted: 1.06x is within noise |

## Diff of the kept changes

```diff
diff --git a/train_image_only.py b/train_image_only.py
index dda0d90..5a8f6ec 100644
--- a/train_image_only.py
+++ b/train_image_only.py
@@ -95,6 +95,10 @@ model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
 
 ema_model.to(accelerator.device)
 
+# Compile in place so parameter names and EMA updates remain unchanged.
+model.transformer.compile()
+ema_model.ema_model.transformer.compile()
+
 # train loop
 
 for step in range(1, 100_000 + 1):
```

## Sources the agent read

- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [PyTorch performance tuning guide: Fuse operations](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [PyTorch performance tuning guide: Use CUDA Graphs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [PyTorch CUDA semantics: Why CUDA Graphs?](https://docs.pytorch.org/docs/2.14/notes/cuda.html#why-cuda-graphs)
- [PyTorch CUDA semantics: PyTorch API](https://docs.pytorch.org/docs/2.14/notes/cuda.html#pytorch-api)
- [PyTorch CUDA semantics: TensorFloat-32 (TF32) on Ampere (and later) devices](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
- [Scaled dot product attention: Using SDPA with attn_bias subclasses](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-attn-bias-subclasses)
- [Scaled dot product attention: Fused implementations](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#fused-implementations)
- [torch.amp: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html#automatic-mixed-precision-package-torch-amp)
- [PyTorch AMP recipe: All together: “Automatic Mixed Precision”](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision)
- [PyTorch AMP examples: Typical Mixed Precision Training](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#typical-mixed-precision-training)
- [PyTorch AMP recipe: Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#automatic-mixed-precision)
