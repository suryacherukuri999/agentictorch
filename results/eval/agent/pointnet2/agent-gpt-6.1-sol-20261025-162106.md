# AgenticTorch report: pointnet2

**Result: 4.89x faster, loss within noise** on an NVIDIA L4.

- Command: `tar xf data/modelnet40_normal_resampled/subset.tar -C data/modelnet40_normal_resampled && python train_classification.py --model pointnet2_cls_ssg --log_dir agentictorch`
- Baseline: 276.1 ms/step over 2 runs
- Model: gpt-6.1-sol, 24 turns, 269166 input and 3060 output tokens

## Diagnosis

```
profiled 10 steps, 1030.5 ms/step under the profiler
GPU busy 12% of the time
waiting on the DataLoader 0% of the time
host-device syncs per step: 1303.0 (copies to the CPU: 1285.0)
  1280.0 per step at models/pointnet2_utils.py:82: distance[mask] = dist[mask]
  5.0 per step at models/pointnet2_utils.py:58: batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
  4.0 per step at models/pointnet2_utils.py:106: group_idx[mask] = group_first[mask]
  2.0 per step at train_classification.py:186: points, target = points.cuda(), target.cuda()
  2.0 per step at models/pointnet2_utils.py:73: centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
  2.0 per step at models/pointnet2_utils.py:74: distance = torch.ones(B, N).to(device) * 1e10
top ops by GPU time:
  13.7%  aten::cudnn_batch_norm_backward  backward of models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
  11.3%  aten::nonzero  at models/pointnet2_utils.py:82: distance[mask] = dist[mask]
  11.2%  aten::convolution_backward  backward of models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   8.7%  aten::cudnn_batch_norm  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   8.1%  aten::threshold_backward  backward of models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   5.7%  aten::clamp_min  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   5.7%  aten::add_  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   4.8%  aten::cudnn_convolution  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
```

Reduced step time from 276.1 to 57.8 ms (~4.8x faster), with loss within benchmark noise. Replaced boolean gather/scatter updates with torch.where, fused farthest-point sampling in Triton while preserving CPU RNG draws and first-index tie behavior, and enabled bfloat16 autocast while retaining float32 geometric distance calculations. Sampling fusion reduced host-device synchronization from roughly 1303 to 13 per step in the intermediate profile. Channels-last layout was tested and reverted because it did not improve performance. No model, data, batch size, learning rate, step count, or logging/checkpoint behavior was changed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Replace boolean gather/scatter updates with equivalent dense torch.where to eliminate nonzero GPU synchronizations. | 288.7 ms/step | 185.2 ms/step | kept: 1.56x faster, loss within noise |
| 2 | Fuse iterative farthest-point sampling into one Triton kernel, preserving initial RNG, float32 arithmetic and first-index argmax ties. | 188.2 ms/step | 96.4 ms/step | kept: 1.95x faster, loss within noise |
| 3 | Use bfloat16 autocast for training convolutions and linear layers while preserving float32 geometry and sampling. | 96.0 ms/step | 57.8 ms/step | kept: 1.66x faster, loss within noise |
| 4 | Use channels-last layout for grouped convolution activations and model convolution weights to improve bfloat16 Tensor Core and memory efficiency. | 57.8 ms/step | 59.9 ms/step | reverted: 0.96x is within noise |

## Diff of the kept changes

```diff
diff --git a/models/pointnet2_utils.py b/models/pointnet2_utils.py
index d13986e..67af703 100644
--- a/models/pointnet2_utils.py
+++ b/models/pointnet2_utils.py
@@ -3,6 +3,36 @@ import torch.nn as nn
 import torch.nn.functional as F
 from time import time
 import numpy as np
+import triton
+import triton.language as tl
+
+
+@triton.jit
+def _fps_kernel(xyz, initial, output, N: tl.constexpr, S: tl.constexpr,
+                STRIDE_B: tl.constexpr, STRIDE_N: tl.constexpr,
+                STRIDE_C: tl.constexpr, BLOCK: tl.constexpr):
+    batch = tl.program_id(0)
+    offsets = tl.arange(0, BLOCK)
+    valid = offsets < N
+    base = xyz + batch * STRIDE_B
+    x = tl.load(base + offsets * STRIDE_N, valid, other=0)
+    y = tl.load(base + offsets * STRIDE_N + STRIDE_C, valid, other=0)
+    z = tl.load(base + offsets * STRIDE_N + 2 * STRIDE_C, valid, other=0)
+    distance = tl.full((BLOCK,), 1.0e10, tl.float32)
+    farthest = tl.load(initial + batch)
+    for i in range(S):
+        tl.store(output + batch * S + i, farthest)
+        cx = tl.load(base + farthest * STRIDE_N)
+        cy = tl.load(base + farthest * STRIDE_N + STRIDE_C)
+        cz = tl.load(base + farthest * STRIDE_N + 2 * STRIDE_C)
+        dx = x - cx
+        dy = y - cy
+        dz = z - cz
+        dist = (dx * dx + dy * dy) + dz * dz
+        distance = tl.where(dist < distance, dist, distance)
+        farthest = tl.argmax(tl.where(valid, distance, -float('inf')), 0,
+                             tie_break_left=True).to(tl.int64)
+
 
 def timeit(tag, t):
     print("{}: {}s".format(tag, time() - t))
@@ -34,9 +64,10 @@ def square_distance(src, dst):
     """
     B, N, _ = src.shape
     _, M, _ = dst.shape
-    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
-    dist += torch.sum(src ** 2, -1).view(B, N, 1)
-    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
+    with torch.autocast(device_type=src.device.type, dtype=torch.bfloat16, enabled=False):
+        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
+        dist += torch.sum(src ** 2, -1).view(B, N, 1)
+        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
     return dist
 
 
@@ -70,6 +101,14 @@ def farthest_point_sample(xyz, npoint):
     """
     device = xyz.device
     B, N, C = xyz.shape
+    if xyz.is_cuda and xyz.dtype == torch.float32 and C == 3:
+        centroids = torch.empty(B, npoint, dtype=torch.long, device=device)
+        # Keep the original CPU RNG draw so subsequent random operations agree.
+        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
+        _fps_kernel[(B,)](xyz, farthest, centroids, N, npoint,
+                          *xyz.stride(), triton.next_power_of_2(N),
+                          enable_fp_fusion=False)
+        return centroids
     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
     distance = torch.ones(B, N).to(device) * 1e10
     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
@@ -78,8 +117,7 @@ def farthest_point_sample(xyz, npoint):
         centroids[:, i] = farthest
         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
         dist = torch.sum((xyz - centroid) ** 2, -1)
-        mask = dist < distance
-        distance[mask] = dist[mask]
+        distance = torch.where(dist < distance, dist, distance)
         farthest = torch.max(distance, -1)[1]
     return centroids
 
@@ -102,8 +140,7 @@ def query_ball_point(radius, nsample, xyz, new_xyz):
     group_idx[sqrdists > radius ** 2] = N
     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
-    mask = group_idx == N
-    group_idx[mask] = group_first[mask]
+    group_idx = torch.where(group_idx == N, group_first, group_idx)
     return group_idx
 
 
diff --git a/train_classification.py b/train_classification.py
index 229f49d..c28fae7 100644
--- a/train_classification.py
+++ b/train_classification.py
@@ -185,8 +185,9 @@ def main(args):
             if not args.use_cpu:
                 points, target = points.cuda(), target.cuda()
 
-            pred, trans_feat = classifier(points)
-            loss = criterion(pred, target.long(), trans_feat)
+            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not args.use_cpu):
+                pred, trans_feat = classifier(points)
+                loss = criterion(pred, target.long(), trans_feat)
             pred_choice = pred.data.max(1)[1]
 
             correct = pred_choice.eq(target.long().data).cpu().sum()
```

## Sources the agent read

- [PyTorch performance tuning guide: Avoid unnecessary CPU-GPU synchronization](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)
- [PyTorch CUDA semantics: CUDA streams](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams)
- [Pinned memory and non_blocking: Other copy directions (GPU -> CPU, CPU -> MPS)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps)
- [PyTorch CUDA semantics: Asynchronous execution](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)
- [PyTorch CUDA semantics: Why CUDA Graphs?](https://docs.pytorch.org/docs/2.14/notes/cuda.html#why-cuda-graphs)
- [torch.compile tutorial: Graph Breaks](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#graph-breaks)
- [PyTorch CUDA semantics: Partial-network capture](https://docs.pytorch.org/docs/2.14/notes/cuda.html#partial-network-capture)
- [PyTorch performance tuning guide: Use CUDA Graphs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP recipe: Loss is inf/NaN](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#loss-is-inf-nan)
- [PyTorch AMP examples: Functions that need a particular dtype](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#functions-that-need-a-particular-dtype)
- [PyTorch AMP recipe: All together: “Automatic Mixed Precision”](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision)
- [Channels last memory format: Memory Format API](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api)
- [Channels last memory format: Performance Gains](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains)
- [PyTorch performance tuning guide: Enable channels_last memory format for computer vision models](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models)
- [Channels last memory format: Converting existing models](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models)
