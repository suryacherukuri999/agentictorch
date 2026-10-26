# AgenticTorch report: transfusion-pytorch

**Result: no change was kept.**

- Command: `python train_image_only.py`
- Baseline: 81.9 ms/step over 2 runs
- Model: gpt-6.1-sol, 9 turns, 76075 input and 2779 output tokens

Optimized train_image_only.py with Accelerate bfloat16 mixed precision, four persistent DataLoader workers, pinned host memory and nonblocking device transfers. The iterator now uses the prepared loader. Buffered detached losses for 50 steps to replace per-step host synchronization while retaining every formatted log line and flushing before image generation. Added a main guard for multiprocessing safety and explicit set_to_none gradient clearing. Model, batch size, learning rate, step count, and image-save schedule remain unchanged. No execution or benchmarks were performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 81.9 ms/step | 95.9 ms/step | reverted: 0.86x is within noise |

## Diff of the kept changes

```diff
diff --git a/train_image_only.py b/train_image_only.py
index dda0d90..dcd3214 100644
--- a/train_image_only.py
+++ b/train_image_only.py
@@ -18,10 +18,6 @@ from transfusion_pytorch import Transfusion, print_modality_sample
 
 from accelerate import Accelerator
 
-rmtree('./results', ignore_errors = True)
-results_folder = Path('./results')
-results_folder.mkdir(exist_ok = True, parents = True)
-
 # functions
 
 def divisible_by(num, den):
@@ -39,29 +35,6 @@ class Decoder(Module):
         x = rearrange(x, '... h w (p1 p2) -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2, h = 14)
         return ((x + 1) * 0.5).clamp(min = 0., max = 1.)
 
-model = Transfusion(
-    num_text_tokens = 10,
-    dim_latent = 4,
-    channel_first_latent = False,
-    modality_default_shape = (14, 14),
-    modality_encoder = Encoder(),
-    modality_decoder = Decoder(),
-    add_pos_emb = True,
-    modality_num_dim = 2,
-    velocity_consistency_loss_weight = 0.1,
-    reconstruction_loss_weight = 0.1,
-    model_output_clean = True,
-    transformer = dict(
-        dim = 64,
-        depth = 4,
-        dim_head = 32,
-        heads = 8,
-        attn_laser = True
-    )
-)
-
-ema_model = model.create_ema()
-
 class MnistDataset(Dataset):
     def __init__(self):
         self.mnist = torchvision.datasets.MNIST(
@@ -82,42 +55,91 @@ def cycle(iter_dl):
         for batch in iter_dl:
             yield batch
 
-dataset = MnistDataset()
+def main():
+    rmtree('./results', ignore_errors = True)
+    results_folder = Path('./results')
+    results_folder.mkdir(exist_ok = True, parents = True)
+
+    model = Transfusion(
+        num_text_tokens = 10,
+        dim_latent = 4,
+        channel_first_latent = False,
+        modality_default_shape = (14, 14),
+        modality_encoder = Encoder(),
+        modality_decoder = Decoder(),
+        add_pos_emb = True,
+        modality_num_dim = 2,
+        velocity_consistency_loss_weight = 0.1,
+        reconstruction_loss_weight = 0.1,
+        model_output_clean = True,
+        transformer = dict(
+            dim = 64,
+            depth = 4,
+            dim_head = 32,
+            heads = 8,
+            attn_laser = True
+        )
+    )
+
+    ema_model = model.create_ema()
+    dataset = MnistDataset()
+
+    dataloader = DataLoader(
+        dataset,
+        batch_size = 32,
+        shuffle = True,
+        num_workers = 4,
+        pin_memory = torch.cuda.is_available(),
+        persistent_workers = True
+    )
+
+    optimizer = MuonAdamAtan2(model.muon_parameters(), model.parameters(), lr = 8e-4)
+    accelerator = Accelerator(mixed_precision = 'bf16')
 
-dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
-iter_dl = cycle(dataloader)
+    model, optimizer = accelerator.prepare(model, optimizer)
+    # Keep host batches pinned until the explicit asynchronous device transfer.
+    dataloader = accelerator.prepare_data_loader(dataloader, device_placement = False)
+    iter_dl = cycle(dataloader)
 
-optimizer = MuonAdamAtan2(model.muon_parameters(), model.parameters(), lr = 8e-4)
+    ema_model.to(accelerator.device)
 
-accelerator = Accelerator()
+    # Buffer detached scalars so logging synchronizes once per 50 steps,
+    # while retaining every original output line and flushing before sampling.
+    pending_losses = []
 
-model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
+    # train loop
 
-ema_model.to(accelerator.device)
+    for step in range(1, 100_000 + 1):
 
-# train loop
+        batch = next(iter_dl).to(accelerator.device, non_blocking = True)
+        loss = model(batch, velocity_consistency_ema_model = ema_model)
+        accelerator.backward(loss)
 
-for step in range(1, 100_000 + 1):
+        accelerator.clip_grad_norm_(model.parameters(), 0.5)
 
-    loss = model(next(iter_dl), velocity_consistency_ema_model = ema_model)
-    accelerator.backward(loss)
+        optimizer.step()
+        optimizer.zero_grad(set_to_none = True)
 
-    accelerator.clip_grad_norm_(model.parameters(), 0.5)
+        ema_model.update()
 
-    optimizer.step()
-    optimizer.zero_grad()
+        pending_losses.append(loss.detach())
+        if divisible_by(step, 50):
+            loss_values = torch.stack(pending_losses).cpu().tolist()
+            for logged_step, loss_value in zip(range(step - len(loss_values) + 1, step + 1), loss_values):
+                accelerator.print(f'{logged_step}: {loss_value:.3f}')
+            pending_losses.clear()
 
-    ema_model.update()
+        if divisible_by(step, 500):
+            accelerator.wait_for_everyone()
 
-    accelerator.print(f'{step}: {loss.item():.3f}')
+            if accelerator.is_main_process:
+                image = ema_model.generate_modality_only(batch_size = 64)
 
-    if divisible_by(step, 500):
-        accelerator.wait_for_everyone()
+                save_image(
+                    rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
+                    str(results_folder / f'{step}.png')
+                )
 
-        if accelerator.is_main_process:
-            image = ema_model.generate_modality_only(batch_size = 64)
 
-            save_image(
-                rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
-                str(results_folder / f'{step}.png')
-            )
+if __name__ == '__main__':
+    main()
```
