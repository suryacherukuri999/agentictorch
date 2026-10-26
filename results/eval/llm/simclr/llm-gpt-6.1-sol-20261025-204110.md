# AgenticTorch report: simclr

**Result: 2.50x faster, loss within noise** on an NVIDIA L4.

- Command: `python main.py`
- Baseline: 417.9 ms/step over 2 runs
- Model: gpt-6.1-sol, 14 turns, 126308 input and 2495 output tokens

Enabled bfloat16 autocast for the SimCLR model while retaining float32 contrastive loss, compiled the single-device forward without changing checkpoint keys, enabled pinned-memory loading and set-to-none gradient clearing, and deferred scalar metric transfers until epoch end while retaining step printouts and per-step TensorBoard values. Replaced broadcasted pairwise cosine similarity with normalized matrix multiplication and cached integer negative indices/labels to avoid boolean-index synchronization and repeated allocation. Removed per-batch metric synchronization in linear evaluation and disabled autograd during testing. No execution or benchmarking was performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 417.9 ms/step | 167.0 ms/step | kept: 2.50x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/linear_evaluation.py b/linear_evaluation.py
index a98038f..7a9da9d 100644
--- a/linear_evaluation.py
+++ b/linear_evaluation.py
@@ -59,11 +59,18 @@ def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size
     return train_loader, test_loader
 
 
+def _epoch_metrics(losses, correct, batch_sizes):
+    if not losses:
+        return 0, 0
+    loss_values = torch.stack(losses).cpu().tolist()
+    correct_values = torch.stack(correct).cpu().tolist()
+    return sum(loss_values), sum(n / size for n, size in zip(correct_values, batch_sizes))
+
+
 def train(args, loader, simclr_model, model, criterion, optimizer):
-    loss_epoch = 0
-    accuracy_epoch = 0
+    losses, correct, batch_sizes = [], [], []
     for step, (x, y) in enumerate(loader):
-        optimizer.zero_grad()
+        optimizer.zero_grad(set_to_none=True)
 
         x = x.to(args.device)
         y = y.to(args.device)
@@ -72,24 +79,24 @@ def train(args, loader, simclr_model, model, criterion, optimizer):
         loss = criterion(output, y)
 
         predicted = output.argmax(1)
-        acc = (predicted == y).sum().item() / y.size(0)
-        accuracy_epoch += acc
+        correct.append((predicted == y).sum())
+        batch_sizes.append(y.size(0))
 
         loss.backward()
         optimizer.step()
 
-        loss_epoch += loss.item()
+        losses.append(loss.detach())
         # if step % 100 == 0:
         #     print(
         #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
         #     )
 
-    return loss_epoch, accuracy_epoch
+    return _epoch_metrics(losses, correct, batch_sizes)
 
 
+@torch.no_grad()
 def test(args, loader, simclr_model, model, criterion, optimizer):
-    loss_epoch = 0
-    accuracy_epoch = 0
+    losses, correct, batch_sizes = [], [], []
     model.eval()
     for step, (x, y) in enumerate(loader):
         model.zero_grad()
@@ -101,12 +108,12 @@ def test(args, loader, simclr_model, model, criterion, optimizer):
         loss = criterion(output, y)
 
         predicted = output.argmax(1)
-        acc = (predicted == y).sum().item() / y.size(0)
-        accuracy_epoch += acc
+        correct.append((predicted == y).sum())
+        batch_sizes.append(y.size(0))
 
-        loss_epoch += loss.item()
+        losses.append(loss.detach())
 
-    return loss_epoch, accuracy_epoch
+    return _epoch_metrics(losses, correct, batch_sizes)
 
 
 if __name__ == "__main__":
diff --git a/main.py b/main.py
index fec4778..6fd50ae 100644
--- a/main.py
+++ b/main.py
@@ -24,16 +24,17 @@ from utils import yaml_config_hook
 
 
 def train(args, train_loader, model, criterion, optimizer, writer):
-    loss_epoch = 0
+    losses = []
     for step, ((x_i, x_j), _) in enumerate(train_loader):
-        optimizer.zero_grad()
+        optimizer.zero_grad(set_to_none=True)
         x_i = x_i.cuda(non_blocking=True)
         x_j = x_j.cuda(non_blocking=True)
 
-        # positive pair, with encoding
-        h_i, h_j, z_i, z_j = model(x_i, x_j)
+        # Keep the contrastive similarity and cross entropy in float32.
+        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
+            h_i, h_j, z_i, z_j = model(x_i, x_j)
 
-        loss = criterion(z_i, z_j)
+        loss = criterion(z_i.float(), z_j.float())
         loss.backward()
 
         optimizer.step()
@@ -45,12 +46,15 @@ def train(args, train_loader, model, criterion, optimizer, writer):
         if args.nr == 0 and step % 50 == 0:
             print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
 
-        if args.nr == 0:
-            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
-            args.global_step += 1
+        losses.append(loss.detach())
 
-        loss_epoch += loss.item()
-    return loss_epoch
+    # Transfer metrics once per epoch rather than synchronizing every step.
+    loss_values = torch.stack(losses).cpu().tolist() if losses else []
+    if args.nr == 0:
+        for value in loss_values:
+            writer.add_scalar("Loss/train_epoch", value, args.global_step)
+            args.global_step += 1
+    return sum(loss_values)
 
 
 def main(gpu, args):
@@ -92,6 +96,7 @@ def main(gpu, args):
         shuffle=(train_sampler is None),
         drop_last=True,
         num_workers=args.workers,
+        pin_memory=True,
         sampler=train_sampler,
     )
 
@@ -110,7 +115,7 @@ def main(gpu, args):
 
     # optimizer / loss
     optimizer, scheduler = load_optimizer(args, model)
-    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
+    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size).to(args.device)
 
     # DDP / DP
     if args.dataparallel:
@@ -122,6 +127,9 @@ def main(gpu, args):
             model = DDP(model, device_ids=[gpu])
 
     model = model.to(args.device)
+    if not args.dataparallel and args.nodes == 1:
+        # Compile only the callable, preserving the original checkpoint keys.
+        model.forward = torch.compile(model.forward)
 
     writer = None
     if args.nr == 0:
diff --git a/simclr/modules/nt_xent.py b/simclr/modules/nt_xent.py
index eea33f3..f5f2ce0 100644
--- a/simclr/modules/nt_xent.py
+++ b/simclr/modules/nt_xent.py
@@ -11,9 +11,13 @@ class NT_Xent(nn.Module):
         self.temperature = temperature
         self.world_size = world_size
 
-        self.mask = self.mask_correlated_samples(batch_size, world_size)
+        mask = self.mask_correlated_samples(batch_size, world_size)
+        N = mask.size(0)
+        # Fixed integer indices avoid the dynamic-size CUDA boolean indexing path.
+        negative_indices = torch.arange(N).expand(N, N)[mask].reshape(N, N - 2)
+        self.register_buffer("negative_indices", negative_indices, persistent=False)
+        self.register_buffer("labels", torch.zeros(N, dtype=torch.long), persistent=False)
         self.criterion = nn.CrossEntropyLoss(reduction="sum")
-        self.similarity_f = nn.CosineSimilarity(dim=2)
 
     def mask_correlated_samples(self, batch_size, world_size):
         N = 2 * batch_size * world_size
@@ -36,16 +40,17 @@ class NT_Xent(nn.Module):
             z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
         z = torch.cat((z_i, z_j), dim=0)
 
-        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
+        z = torch.nn.functional.normalize(z, dim=1, eps=1e-8)
+        sim = torch.mm(z, z.t()) / self.temperature
 
         sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
         sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
 
         # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
-        negative_samples = sim[self.mask].reshape(N, -1)
+        negative_samples = sim.gather(1, self.negative_indices.to(sim.device))
 
-        labels = torch.zeros(N).to(positive_samples.device).long()
+        labels = self.labels.to(sim.device)
         logits = torch.cat((positive_samples, negative_samples), dim=1)
         loss = self.criterion(logits, labels)
         loss /= N
```
