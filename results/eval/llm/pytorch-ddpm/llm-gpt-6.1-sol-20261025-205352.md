# AgenticTorch report: pytorch-ddpm

**Result: 2.52x faster, loss within noise** on an NVIDIA L4.

- Command: `python main.py --train --flagfile ./config/CIFAR10.txt`
- Baseline: 544.6 ms/step over 2 runs
- Model: gpt-6.1-sol, 12 turns, 108631 input and 2194 output tokens

Enabled bfloat16 autocast and torch.compile for the training model while preserving original model references and checkpoint keys. Replaced explicit attention matrices with equivalent scaled_dot_product_attention. Added pinned/nonblocking data transfers, cuDNN autotuning, fused Adam, foreach gradient clipping and EMA updates, set-to-none gradient clearing, and a single loss scalar transfer per step. Kept data, model architecture, batch size, learning-rate schedule, step counts, and logging/saving structure unchanged. Changes were reviewed statically; no execution or measurement was available.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 544.6 ms/step | 216.2 ms/step | kept: 2.52x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/main.py b/main.py
index c6e0774..ce9e783 100644
--- a/main.py
+++ b/main.py
@@ -55,13 +55,13 @@ flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
 device = torch.device('cuda:0')
 
 
+@torch.no_grad()
 def ema(source, target, decay):
-    source_dict = source.state_dict()
-    target_dict = target.state_dict()
-    for key in source_dict.keys():
-        target_dict[key].data.copy_(
-            target_dict[key].data * decay +
-            source_dict[key].data * (1 - decay))
+    source_tensors = list(source.state_dict().values())
+    target_tensors = list(target.state_dict().values())
+    torch._foreach_mul_(target_tensors, decay)
+    updates = torch._foreach_mul(source_tensors, 1 - decay)
+    torch._foreach_add_(target_tensors, updates)
 
 
 def infiniteloop(dataloader):
@@ -93,6 +93,9 @@ def evaluate(sampler, model):
 
 
 def train():
+    # All training batches have the same shape; reuse tuned convolution kernels.
+    torch.backends.cudnn.benchmark = True
+
     # dataset
     dataset = CIFAR10(
         root='./data', train=True, download=True,
@@ -103,15 +106,15 @@ def train():
         ]))
     dataloader = torch.utils.data.DataLoader(
         dataset, batch_size=FLAGS.batch_size, shuffle=True,
-        num_workers=FLAGS.num_workers, drop_last=True)
+        num_workers=FLAGS.num_workers, drop_last=True, pin_memory=True)
     datalooper = infiniteloop(dataloader)
 
     # model setup
     net_model = UNet(
         T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
-        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
+        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
     ema_model = copy.deepcopy(net_model)
-    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
+    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr, fused=True)
     sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
     trainer = GaussianDiffusionTrainer(
         net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
@@ -121,6 +124,9 @@ def train():
     ema_sampler = GaussianDiffusionSampler(
         ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
         FLAGS.mean_type, FLAGS.var_type).to(device)
+    # Compile only the training model call; keep the original modules for
+    # sampling, EMA, and checkpoints (including their existing state-dict keys).
+    trainer.model = torch.compile(net_model)
     if FLAGS.parallel:
         trainer = torch.nn.DataParallel(trainer)
         net_sampler = torch.nn.DataParallel(net_sampler)
@@ -147,19 +153,21 @@ def train():
     with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
         for step in pbar:
             # train
-            optim.zero_grad()
-            x_0 = next(datalooper).to(device)
-            loss = trainer(x_0).mean()
+            optim.zero_grad(set_to_none=True)
+            x_0 = next(datalooper).to(device, non_blocking=True)
+            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
+                loss = trainer(x_0).mean()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(
-                net_model.parameters(), FLAGS.grad_clip)
+                net_model.parameters(), FLAGS.grad_clip, foreach=True)
             optim.step()
             sched.step()
             ema(net_model, ema_model, FLAGS.ema_decay)
 
             # log
-            writer.add_scalar('loss', loss, step)
-            pbar.set_postfix(loss='%.3f' % loss)
+            loss_value = loss.item()
+            writer.add_scalar('loss', loss_value, step)
+            pbar.set_postfix(loss='%.3f' % loss_value)
 
             # sample
             if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
diff --git a/model.py b/model.py
index ac0c1b5..72a573d 100644
--- a/model.py
+++ b/model.py
@@ -98,16 +98,13 @@ class AttnBlock(nn.Module):
         k = self.proj_k(h)
         v = self.proj_v(h)
 
-        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
-        k = k.view(B, C, H * W)
-        w = torch.bmm(q, k) * (int(C) ** (-0.5))
-        assert list(w.shape) == [B, H * W, H * W]
-        w = F.softmax(w, dim=-1)
-
-        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
-        h = torch.bmm(w, v)
-        assert list(h.shape) == [B, H * W, C]
-        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
+        # A single attention head, with the same 1/sqrt(C) scaling as before.
+        # Contiguous head dimensions allow the fused CUDA attention kernels.
+        q = q.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
+        k = k.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
+        v = v.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
+        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
+        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
         h = self.proj(h)
 
         return x + h
```
