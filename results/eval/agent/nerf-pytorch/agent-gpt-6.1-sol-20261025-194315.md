# AgenticTorch report: nerf-pytorch

**Result: 2.43x faster, loss within noise** on an NVIDIA L4.

- Command: `python run_nerf.py --config configs/lego.txt`
- Baseline: 185.2 ms/step over 2 runs
- Model: gpt-6.1-sol, 21 turns, 291497 input and 1299 output tokens

## Diagnosis

```
profiled 10 steps, 210.5 ms/step under the profiler
GPU busy 76% of the time
waiting on the DataLoader 0% of the time
host-device syncs per step: 24.0 (copies to the CPU: 18.0)
  16.0 per step at run_nerf.py:415: if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
  2.0 per step at run_nerf.py:278: dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_sampl
  2.0 per step at run_nerf_helpers.py:10: mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
  2.0 per step at run_nerf.py:775: loss.backward()
  1.0 per step at run_nerf.py:732: target = torch.Tensor(target).to(device)
  1.0 per step at run_nerf.py:753: select_coords = coords[select_inds].long()  # (N_rand, 2)
top ops by GPU time:
  27.8%  aten::mm  backward of run_nerf_helpers.py:100: h = self.pts_linears[i](h)
  15.4%  aten::addmm  at run_nerf_helpers.py:100: h = self.pts_linears[i](h)
  15.1%  aten::threshold_backward  backward of run_nerf_helpers.py:101: h = F.relu(h)
  10.8%  aten::clamp_min  at run_nerf_helpers.py:101: h = F.relu(h)
   5.7%  aten::sum  backward of run_nerf_helpers.py:100: h = self.pts_linears[i](h)
   3.6%  aten::mm  backward of run_nerf_helpers.py:107: feature = self.feature_linear(h)
   2.7%  aten::mm  backward of run_nerf_helpers.py:111: h = self.views_linears[i](h)
   2.3%  aten::cat  at run_nerf_helpers.py:103: h = torch.cat([input_pts, h], -1)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Short-circuit disabled debug checks before evaluating CUDA reductions, eliminating unnecessary host-device synchronizations. | 175.5 ms/step | 175.2 ms/step | reverted: 1.00x is within noise |
| 2 | Use bfloat16 autocast only for NeRF MLP evaluations on CUDA; retain float32 volume integration and loss. | 182.1 ms/step | 88.6 ms/step | kept: 2.06x faster, loss within noise |
| 3 | Compile coarse and fine MLPs in place to fuse pointwise operations and optimize forward/backward without changing checkpoint format. | 88.3 ms/step | 71.6 ms/step | kept: 1.23x faster, loss within noise |
| 4 | Retest short-circuiting disabled debug reductions after accelerating the MLP, where synchronization overhead is a larger fraction of step time. | 74.5 ms/step | 71.9 ms/step | reverted: 1.04x is within noise |
| 5 | Use fused CUDA Adam to combine parameter-update kernels while retaining optimizer hyperparameters and float32 parameter states. | 70.1 ms/step | 70.4 ms/step | reverted: 1.00x is within noise |

## Diff of the kept changes

```diff
diff --git a/run_nerf.py b/run_nerf.py
index bc270be..3300fa5 100644
--- a/run_nerf.py
+++ b/run_nerf.py
@@ -46,7 +46,10 @@ def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
         embedded_dirs = embeddirs_fn(input_dirs_flat)
         embedded = torch.cat([embedded, embedded_dirs], -1)
 
-    outputs_flat = batchify(fn, netchunk)(embedded)
+    # Use tensor cores for the MLP, keeping volume integration in float32.
+    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
+        outputs_flat = batchify(fn, netchunk)(embedded)
+    outputs_flat = outputs_flat.float()
     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
     return outputs
 
@@ -234,6 +237,12 @@ def create_nerf(args):
 
     ##########################
 
+    # Compile in place so model parameters and checkpoint keys stay unchanged.
+    if device.type == 'cuda':
+        model.compile()
+        if model_fine is not None:
+            model_fine.compile()
+
     render_kwargs_train = {
         'network_query_fn' : network_query_fn,
         'perturb' : args.perturb,
```

## Sources the agent read

- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch performance tuning guide: Avoid unnecessary CPU-GPU synchronization](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [PyTorch AMP examples: DataParallel in a single process](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#dataparallel-in-a-single-process)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Graph Breaks](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#graph-breaks)
- [torch.compile tutorial: Benefits over TorchScript](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#benefits-over-torchscript)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [PyTorch AMP recipe: Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#automatic-mixed-precision)
- [torch.amp: CUDA Ops that can autocast to float32](https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-can-autocast-to-float32)
