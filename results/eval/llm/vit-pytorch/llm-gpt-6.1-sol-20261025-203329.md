# AgenticTorch report: vit-pytorch

**Result: 1.80x faster, loss within noise** on an NVIDIA L4.

- Command: `python train_vit_decorr.py`
- Baseline: 26.0 ms/step over 2 runs
- Model: gpt-6.1-sol, 8 turns, 31124 input and 1036 output tokens

Added four DataLoader workers and CUDA pinned memory; enabled fused Adam on CUDA; replaced explicit attention operations with equivalent scaled_dot_product_attention; combined detached loss metrics into a single CPU transfer after queuing backward/optimizer work, preserving per-step W&B metrics and printed formatting; explicitly use set_to_none gradient clearing. Training configuration and precision are unchanged. No execution or benchmarking was performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 26.0 ms/step | 14.4 ms/step | kept: 1.80x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/train_vit_decorr.py b/train_vit_decorr.py
index 23b4051..0ce3d1d 100644
--- a/train_vit_decorr.py
+++ b/train_vit_decorr.py
@@ -41,7 +41,13 @@ dataset = CIFAR100(
     transform = transform
 )
 
-dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
+dataloader = DataLoader(
+    dataset,
+    batch_size = BATCH_SIZE,
+    shuffle = True,
+    num_workers = 4,
+    pin_memory = torch.cuda.is_available()
+)
 
 # model
 
@@ -67,7 +73,7 @@ vit = ViT(
 
 from torch.optim import Adam
 
-optim = Adam(vit.parameters(), lr = LEARNING_RATE)
+optim = Adam(vit.parameters(), lr = LEARNING_RATE, fused = torch.cuda.is_available())
 
 # prepare
 
@@ -102,10 +108,13 @@ for _ in range(EPOCHS):
             decorr_aux_loss * DECORR_LOSS_WEIGHT
         )
 
-        wandb.log(dict(loss = loss, decorr_loss = decorr_aux_loss))
-
-        accelerator.print(f'loss: {loss.item():.3f} | decorr aux loss: {decorr_aux_loss.item():.3f}')
+        # Queue training work before synchronizing for the per-step output.
+        metrics = torch.stack((loss.detach(), decorr_aux_loss.detach()))
 
         accelerator.backward(total_loss)
         optim.step()
-        optim.zero_grad()
+        optim.zero_grad(set_to_none = True)
+
+        loss_value, decorr_value = metrics.cpu().tolist()
+        wandb.log(dict(loss = loss_value, decorr_loss = decorr_value))
+        accelerator.print(f'loss: {loss_value:.3f} | decorr aux loss: {decorr_value:.3f}')
diff --git a/vit_pytorch/vit_with_decorr.py b/vit_pytorch/vit_with_decorr.py
index 401d8fd..1abea43 100644
--- a/vit_pytorch/vit_with_decorr.py
+++ b/vit_pytorch/vit_with_decorr.py
@@ -141,12 +141,10 @@ class Attention(Module):
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
+            dropout_p = self.dropout.p if self.training else 0.
+        )
         out = rearrange(out, 'b h n d -> b n (h d)')
 
         return self.to_out(out), normed
```
