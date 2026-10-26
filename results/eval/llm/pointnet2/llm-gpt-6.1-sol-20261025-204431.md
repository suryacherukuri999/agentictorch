# AgenticTorch report: pointnet2

**Result: 1.93x faster, loss within noise** on an NVIDIA L4.

- Command: `tar xf data/modelnet40_normal_resampled/subset.tar -C data/modelnet40_normal_resampled && python train_classification.py --model pointnet2_cls_ssg --log_dir agentictorch`
- Baseline: 283.8 ms/step over 2 runs
- Model: gpt-6.1-sol, 14 turns, 205598 input and 3395 output tokens

Optimized the requested classification training path: replaced synchronization-heavy boolean indexing in farthest-point sampling with torch.where; replaced full ball-query sorting with equivalent smallest-index top-k selection and broadcasted masking; removed repeated index allocations/transfers; deferred training and validation metric transfers until epoch end while preserving the original averaging rules; retained pinned CPU tensors through NumPy augmentation and enabled nonblocking CUDA copies and persistent loader workers; enabled supported CUDA BF16 autocast while keeping geometric distance calculations in FP32; and used set_to_none gradient clearing. Model structure, augmentation operations, batch size, optimizer settings, schedule, steps, log formats, and checkpoint schema remain unchanged. No execution or benchmarking was available.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 283.8 ms/step | 147.1 ms/step | kept: 1.93x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/models/pointnet2_utils.py b/models/pointnet2_utils.py
index d13986e..8286847 100644
--- a/models/pointnet2_utils.py
+++ b/models/pointnet2_utils.py
@@ -16,6 +16,7 @@ def pc_normalize(pc):
     pc = pc / m
     return pc
 
+@torch.autocast(device_type='cuda', enabled=False)
 def square_distance(src, dst):
     """
     Calculate Euclid distance between each two points.
@@ -53,9 +54,7 @@ def index_points(points, idx):
     B = points.shape[0]
     view_shape = list(idx.shape)
     view_shape[1:] = [1] * (len(view_shape) - 1)
-    repeat_shape = list(idx.shape)
-    repeat_shape[0] = 1
-    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
+    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape)
     new_points = points[batch_indices, idx, :]
     return new_points
 
@@ -70,16 +69,16 @@ def farthest_point_sample(xyz, npoint):
     """
     device = xyz.device
     B, N, C = xyz.shape
-    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
-    distance = torch.ones(B, N).to(device) * 1e10
+    centroids = torch.empty(B, npoint, dtype=torch.long, device=device)
+    distance = torch.full((B, N), 1e10, device=device)
+    # Keep the initial draw on the CPU to preserve the RNG stream.
     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
-    batch_indices = torch.arange(B, dtype=torch.long).to(device)
+    batch_indices = torch.arange(B, dtype=torch.long, device=device)
     for i in range(npoint):
         centroids[:, i] = farthest
         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
         dist = torch.sum((xyz - centroid) ** 2, -1)
-        mask = dist < distance
-        distance[mask] = dist[mask]
+        distance = torch.where(dist < distance, dist, distance)
         farthest = torch.max(distance, -1)[1]
     return centroids
 
@@ -97,14 +96,13 @@ def query_ball_point(radius, nsample, xyz, new_xyz):
     device = xyz.device
     B, N, C = xyz.shape
     _, S, _ = new_xyz.shape
-    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
+    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N)
     sqrdists = square_distance(new_xyz, xyz)
-    group_idx[sqrdists > radius ** 2] = N
-    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
-    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
-    mask = group_idx == N
-    group_idx[mask] = group_first[mask]
-    return group_idx
+    group_idx = torch.where(sqrdists > radius ** 2, N, group_idx)
+    # Select the same lowest point indices without sorting the entire cloud.
+    group_idx = group_idx.topk(min(nsample, N), dim=-1, largest=False, sorted=True).values
+    group_first = group_idx[:, :, :1]
+    return torch.where(group_idx == N, group_first, group_idx)
 
 
 def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
diff --git a/train_classification.py b/train_classification.py
index 229f49d..5d943b1 100644
--- a/train_classification.py
+++ b/train_classification.py
@@ -49,27 +49,38 @@ def inplace_relu(m):
         m.inplace=True
 
 
-def test(model, loader, num_class=40):
+def test(model, loader, num_class=40, use_amp=False):
     mean_correct = []
     class_acc = np.zeros((num_class, 3))
+    batch_stats = []
     classifier = model.eval()
 
     for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
 
         if not args.use_cpu:
-            points, target = points.cuda(), target.cuda()
+            points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)
 
         points = points.transpose(2, 1)
-        pred, _ = classifier(points)
-        pred_choice = pred.data.max(1)[1]
-
-        for cat in np.unique(target.cpu()):
-            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
-            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
+        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
+            pred, _ = classifier(points)
+        pred_choice = pred.detach().max(1)[1]
+        labels = target.long()
+        correct = pred_choice.eq(labels).long()
+        # Fixed-size scatter reductions avoid dynamic boolean indexing and
+        # per-category GPU-to-host synchronization.
+        counts = torch.zeros(num_class, dtype=torch.long, device=target.device)
+        hits = torch.zeros_like(counts)
+        counts.scatter_add_(0, labels, torch.ones_like(labels))
+        hits.scatter_add_(0, labels, correct)
+        batch_stats.append(torch.stack((hits, counts)))
+
+    # Retain the original mean of per-batch, per-class accuracies (not a
+    # dataset-wide weighted class average), with just one transfer.
+    for hits, counts in torch.stack(batch_stats).cpu().numpy():
+        for cat in np.flatnonzero(counts):
+            class_acc[cat, 0] += int(hits[cat]) / float(counts[cat])
             class_acc[cat, 1] += 1
-
-        correct = pred_choice.eq(target.long().data).cpu().sum()
-        mean_correct.append(correct.item() / float(points.size()[0]))
+        mean_correct.append(int(hits.sum()) / float(counts.sum()))
 
     class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
     class_acc = np.mean(class_acc[:, 2])
@@ -120,8 +131,8 @@ def main(args):
 
     train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
     test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
-    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
-    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
+    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True, pin_memory=not args.use_cpu, persistent_workers=True)
+    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=not args.use_cpu, persistent_workers=True)
 
     '''MODEL LOADING'''
     num_class = args.num_category
@@ -137,6 +148,7 @@ def main(args):
     if not args.use_cpu:
         classifier = classifier.cuda()
         criterion = criterion.cuda()
+    use_amp = not args.use_cpu and torch.cuda.is_bf16_supported()
 
     try:
         checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
@@ -173,33 +185,35 @@ def main(args):
 
         scheduler.step()
         for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
-            optimizer.zero_grad()
-
-            points = points.data.numpy()
-            points = provider.random_point_dropout(points)
-            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
-            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
-            points = torch.Tensor(points)
+            optimizer.zero_grad(set_to_none=True)
+
+            # Augment the CPU tensor through its NumPy view. Keeping the
+            # original tensor also keeps the DataLoader's pinned allocation.
+            points_np = points.numpy()
+            points_np = provider.random_point_dropout(points_np)
+            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
+            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
             points = points.transpose(2, 1)
 
             if not args.use_cpu:
-                points, target = points.cuda(), target.cuda()
+                points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)
 
-            pred, trans_feat = classifier(points)
-            loss = criterion(pred, target.long(), trans_feat)
-            pred_choice = pred.data.max(1)[1]
+            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
+                pred, trans_feat = classifier(points)
+                loss = criterion(pred, target.long(), trans_feat)
+            pred_choice = pred.detach().max(1)[1]
 
-            correct = pred_choice.eq(target.long().data).cpu().sum()
-            mean_correct.append(correct.item() / float(points.size()[0]))
+            correct = pred_choice.eq(target.long()).sum()
+            mean_correct.append(correct)
             loss.backward()
             optimizer.step()
             global_step += 1
 
-        train_instance_acc = np.mean(mean_correct)
+        train_instance_acc = np.mean(torch.stack(mean_correct).cpu().numpy() / float(args.batch_size))
         log_string('Train Instance Accuracy: %f' % train_instance_acc)
 
         with torch.no_grad():
-            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
+            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, use_amp=use_amp)
 
             if (instance_acc >= best_instance_acc):
                 best_instance_acc = instance_acc
```
