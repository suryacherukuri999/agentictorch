# AgenticTorch report: pointnet2

**Result: 4.94x faster, loss within noise** on an NVIDIA L4.

- Command: `tar xf data/modelnet40_normal_resampled/subset.tar -C data/modelnet40_normal_resampled && python train_classification.py --model pointnet2_cls_ssg --log_dir agentictorch`
- Baseline: 291.0 ms/step over 2 runs
- Model: gpt-6.1-sol, 29 turns, 360710 input and 3425 output tokens

## Diagnosis

```
profiled 10 steps, 1058.1 ms/step under the profiler
GPU busy 11% of the time
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
  11.3%  aten::convolution_backward  backward of models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
  11.2%  aten::nonzero  at models/pointnet2_utils.py:82: distance[mask] = dist[mask]
   8.8%  aten::cudnn_batch_norm  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   8.1%  aten::threshold_backward  backward of models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   5.7%  aten::add_  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   5.7%  aten::clamp_min  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
   4.8%  aten::cudnn_convolution  at models/pointnet2_utils.py:198: new_points =  F.relu(bn(conv(new_points)))
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Replace dynamic boolean gather/scatter updates with fixed-shape torch.where to eliminate repeated CUDA nonzero synchronizations. | 279.3 ms/step | 186.1 ms/step | kept: 1.50x faster, loss within noise |
| 2 | Fuse FPS into a Triton kernel with unchanged CPU RNG, FP32 distance arithmetic without FMA, and first-index argmax tie handling. | 174.2 ms/step | 95.4 ms/step | kept: 1.82x faster, loss within noise |
| 3 | Use bfloat16 autocast for train/eval convolution and linear layers, with float32 neighborhood geometry; disabled scaler is a no-op for BF16. | 96.5 ms/step | 57.4 ms/step | kept: 1.68x faster, loss within noise |
| 4 | Use channels-last storage for Conv2d weights and grouped point activations to improve BF16 convolution and batchnorm throughput. | 56.3 ms/step | 58.6 ms/step | reverted: 0.96x is within noise |
| 5 | Replace full neighborhood index sort with sorted smallest-k selection; only the same sorted prefix of integer values is needed. | 57.8 ms/step | 59.8 ms/step | reverted: 0.97x is within noise |

## Diff of the kept changes

```diff
diff --git a/models/pointnet2_utils.py b/models/pointnet2_utils.py
index d13986e..2574011 100644
--- a/models/pointnet2_utils.py
+++ b/models/pointnet2_utils.py
@@ -4,6 +4,36 @@ import torch.nn.functional as F
 from time import time
 import numpy as np
 
+try:
+    import triton
+    import triton.language as tl
+except ImportError:
+    triton = None
+
+if triton is not None:
+    @triton.jit
+    def _fps_kernel(X, Start, Out, N: tl.constexpr, S: tl.constexpr,
+                    STRIDE_B: tl.constexpr, STRIDE_N: tl.constexpr,
+                    STRIDE_C: tl.constexpr, BLOCK: tl.constexpr):
+        b = tl.program_id(0)
+        ids = tl.arange(0, BLOCK)
+        base = X + b * STRIDE_B
+        x = tl.load(base + ids * STRIDE_N, ids < N, other=0)
+        y = tl.load(base + ids * STRIDE_N + STRIDE_C, ids < N, other=0)
+        z = tl.load(base + ids * STRIDE_N + 2 * STRIDE_C, ids < N, other=0)
+        distance = tl.full((BLOCK,), 1.0e10, tl.float32)
+        farthest = tl.load(Start + b).to(tl.int32)
+        for i in range(S):
+            tl.store(Out + b * S + i, farthest)
+            cx = tl.load(base + farthest * STRIDE_N)
+            cy = tl.load(base + farthest * STRIDE_N + STRIDE_C)
+            cz = tl.load(base + farthest * STRIDE_N + 2 * STRIDE_C)
+            dx, dy, dz = x - cx, y - cy, z - cz
+            dist = (dx * dx + dy * dy) + dz * dz
+            distance = tl.where(dist < distance, dist, distance)
+            best = tl.max(tl.where(ids < N, distance, -float('inf')), 0)
+            farthest = tl.min(tl.where((ids < N) & (distance == best), ids, 2147483647), 0)
+
 def timeit(tag, t):
     print("{}: {}s".format(tag, time() - t))
     return time()
@@ -34,9 +64,11 @@ def square_distance(src, dst):
     """
     B, N, _ = src.shape
     _, M, _ = dst.shape
-    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
-    dist += torch.sum(src ** 2, -1).view(B, N, 1)
-    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
+    # Neighborhood membership must retain full-precision geometry under AMP.
+    with torch.autocast(device_type=src.device.type, enabled=False):
+        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
+        dist += torch.sum(src ** 2, -1).view(B, N, 1)
+        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
     return dist
 
 
@@ -70,6 +102,14 @@ def farthest_point_sample(xyz, npoint):
     """
     device = xyz.device
     B, N, C = xyz.shape
+    if triton is not None and xyz.is_cuda and xyz.dtype == torch.float32 and C == 3:
+        centroids = torch.empty((B, npoint), dtype=torch.long, device=device)
+        # Retain the original CPU random generator and its state progression.
+        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
+        _fps_kernel[(B,)](xyz, farthest, centroids, N, npoint,
+                          *xyz.stride(), triton.next_power_of_2(N),
+                          enable_fp_fusion=False)
+        return centroids
     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
     distance = torch.ones(B, N).to(device) * 1e10
     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
@@ -78,8 +118,7 @@ def farthest_point_sample(xyz, npoint):
         centroids[:, i] = farthest
         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
         dist = torch.sum((xyz - centroid) ** 2, -1)
-        mask = dist < distance
-        distance[mask] = dist[mask]
+        distance = torch.where(dist < distance, dist, distance)
         farthest = torch.max(distance, -1)[1]
     return centroids
 
@@ -102,8 +141,7 @@ def query_ball_point(radius, nsample, xyz, new_xyz):
     group_idx[sqrdists > radius ** 2] = N
     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
-    mask = group_idx == N
-    group_idx[mask] = group_first[mask]
+    group_idx = torch.where(group_idx == N, group_first, group_idx)
     return group_idx
 
 
diff --git a/train_classification.py b/train_classification.py
index 229f49d..437a822 100644
--- a/train_classification.py
+++ b/train_classification.py
@@ -60,7 +60,8 @@ def test(model, loader, num_class=40):
             points, target = points.cuda(), target.cuda()
 
         points = points.transpose(2, 1)
-        pred, _ = classifier(points)
+        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not args.use_cpu):
+            pred, _ = classifier(points)
         pred_choice = pred.data.max(1)[1]
 
         for cat in np.unique(target.cpu()):
@@ -158,6 +159,8 @@ def main(args):
     else:
         optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
 
+    # BF16 has FP32's exponent range and does not require loss scaling.
+    scaler = torch.amp.GradScaler('cuda', enabled=False)
     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
     global_epoch = 0
     global_step = 0
@@ -185,14 +188,16 @@ def main(args):
             if not args.use_cpu:
                 points, target = points.cuda(), target.cuda()
 
-            pred, trans_feat = classifier(points)
-            loss = criterion(pred, target.long(), trans_feat)
+            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not args.use_cpu):
+                pred, trans_feat = classifier(points)
+                loss = criterion(pred, target.long(), trans_feat)
             pred_choice = pred.data.max(1)[1]
 
             correct = pred_choice.eq(target.long().data).cpu().sum()
             mean_correct.append(correct.item() / float(points.size()[0]))
-            loss.backward()
-            optimizer.step()
+            scaler.scale(loss).backward()
+            scaler.step(optimizer)
+            scaler.update()
             global_step += 1
 
         train_instance_acc = np.mean(mean_correct)
```

## Sources the agent read

- [PyTorch performance tuning guide: Avoid unnecessary CPU-GPU synchronization](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)
- [PyTorch CUDA semantics: CUDA streams](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams)
- [PyTorch CUDA semantics: Asynchronous execution](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)
- [Pinned memory and non_blocking: Other copy directions (GPU -> CPU, CPU -> MPS)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps)
- [PyTorch performance tuning guide: Fuse operations](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)
- [torch.compile tutorial: Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP recipe: All together: “Automatic Mixed Precision”](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision)
- [PyTorch AMP recipe: Adding torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
- [torch.amp: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html#automatic-mixed-precision-package-torch-amp)
- [Channels last memory format: Converting existing models](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models)
- [PyTorch performance tuning guide: Enable channels_last memory format for computer vision models](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models)
- [Channels last memory format: Memory Format API](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api)
- [Channels last memory format: Performance Gains](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains)
