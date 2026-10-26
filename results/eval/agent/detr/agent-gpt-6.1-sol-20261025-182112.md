# AgenticTorch report: detr

**Result: no change was kept.**

- Command: `tar xf data/coco/subset.tar -C data/coco && python main.py --coco_path data/coco --output_dir out`
- Baseline: 207.9 ms/step over 2 runs
- Model: gpt-6.1-sol, 21 turns, 326978 input and 1674 output tokens

## Diagnosis

```
profiled 10 steps, 579.5 ms/step under the profiler
GPU busy 23% of the time
waiting on the DataLoader 3% of the time
host-device syncs per step: 161.0 (copies to the CPU: 76.0)
  44.0 per step at util/misc.py:167: v = v.item()
  14.0 per step at engine.py:30: targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
  12.0 per step at util/box_ops.py:51: assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
  12.0 per step at util/box_ops.py:52: assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
  12.0 per step at models/detr.py:116: target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
  12.0 per step at models/detr.py:119: target_classes[idx] = target_classes_o
top ops by GPU time:
  17.1%  aten::convolution_backward  backward of models/backbone.py:73: xs = self.body(tensor_list.tensors)
  12.1%  aten::cudnn_convolution  at models/backbone.py:73: xs = self.body(tensor_list.tensors)
   5.9%  aten::add  at models/backbone.py:55: return x * scale + bias
   5.2%  aten::mul  at models/backbone.py:55: return x * scale + bias
   5.2%  aten::add_  at models/backbone.py:73: xs = self.body(tensor_list.tensors)
   4.0%  aten::threshold_backward  backward of models/backbone.py:73: xs = self.body(tensor_list.tensors)
   3.8%  aten::clamp_min_  at models/backbone.py:73: xs = self.body(tensor_list.tensors)
   3.6%  aten::copy_  at util/misc.py:291: cast_tensor = self.tensors.to(device)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Replace blocking GIoU box assertions with asynchronous device assertions, preserving validity checks without 24 host synchronizations per step. | 224.6 ms/step | 212.5 ms/step | reverted: 1.05x is within noise |
| 2 | Use bfloat16 autocast for model forward and float32 for Hungarian matching and losses; backward remains outside autocast. | 190.1 ms/step | 184.8 ms/step | reverted: 1.03x is within noise |
| 3 | Pack Hungarian assignment indices and transfer once to the prediction device, eliminating repeated implicit CPU-index transfers during criterion indexing. | 217.3 ms/step | 209.3 ms/step | reverted: 1.04x is within noise |
| 4 | Compile the ResNet backbone in place with dynamic shapes to fuse normalization/elementwise kernels while preserving model and checkpoint structure. | 206.4 ms/step | 212.3 ms/step | reverted: 0.97x is within noise |
| 5 | Use fused CUDA AdamW to reduce optimizer kernel launches without changing optimizer settings or parameter groups. | 209.6 ms/step | 201.9 ms/step | reverted: 1.04x is within noise |

## Sources the agent read

- [Pinned memory and non_blocking: Other copy directions (GPU -> CPU, CPU -> MPS)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps)
- [PyTorch CUDA semantics: Asynchronous execution](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)
- [Pinned memory and non_blocking: Asynchronous vs. Synchronous Operations with non_blocking=True (CUDA cudaMemcpyAsync)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)
- [Pinned memory and non_blocking: What you will learn](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#what-you-will-learn)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP examples: Typical Mixed Precision Training](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#typical-mixed-precision-training)
- [PyTorch AMP examples: Gradient penalty](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-penalty)
- [PyTorch AMP recipe: Adding torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Benefits over TorchScript](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#benefits-over-torchscript)
- [torch.optim: How to utilize named parameters to load optimizer state dict](https://docs.pytorch.org/docs/2.14/optim.html#how-to-utilize-named-parameters-to-load-optimizer-state-dict)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [Scaled dot product attention: Fused implementations](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#fused-implementations)
- [PyTorch GitHub issues: Training slowdown from 1.6 to 1.7.1](https://github.com/pytorch/pytorch/issues/50636)
- [PyTorch CUDA semantics: Whole-network capture](https://docs.pytorch.org/docs/2.14/notes/cuda.html#whole-network-capture)
