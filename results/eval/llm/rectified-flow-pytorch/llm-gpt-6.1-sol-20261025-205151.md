# AgenticTorch report: rectified-flow-pytorch

**Result: 3.17x faster, loss within noise** on an NVIDIA L4.

- Command: `HF_HOME=hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python train_oxford.py`
- Baseline: 446.4 ms/step over 2 runs
- Model: gpt-6.1-sol, 10 turns, 82052 input and 1758 output tokens

Optimized the Oxford training path with Accelerate bfloat16 mixed precision, CUDA fused Adam, compiled training-network forward (without changing checkpoint keys), and cuDNN autotuning. Added four DataLoader workers and CUDA pinned memory, keeping workers non-persistent so sampling cannot reset the active training iterator. Replaced explicit full-attention score materialization with equivalent scaled_dot_product_attention. Deferred unchanged per-step console output until backward, optimizer, and EMA work has been queued. No changes to model architecture, dataset, batch size, learning rate, step count, or output/checkpoint schedules. Changes were not run or benchmarked.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 446.4 ms/step | 141.0 ms/step | kept: 3.17x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/rectified_flow_pytorch/rectified_flow.py b/rectified_flow_pytorch/rectified_flow.py
index dee8fda..c65f679 100644
--- a/rectified_flow_pytorch/rectified_flow.py
+++ b/rectified_flow_pytorch/rectified_flow.py
@@ -744,11 +744,7 @@ class Attention(Module):
         mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
         k, v = map(partial(cat, dim = -2), ((mk, k), (mv, v)))
 
-        q = q * self.scale
-        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
-
-        attn = sim.softmax(dim = -1)
-        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
+        out = F.scaled_dot_product_attention(q, k, v, dropout_p = 0.)
 
         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
         return self.to_out(out)
@@ -1194,7 +1190,16 @@ class Trainer(Module):
         # optimizer, dataloader, and all that
 
         self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
-        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
+        # Keep workers non-persistent: sample() also opens an iterator on this
+        # loader and must not reset the training iterator.
+        self.dl = DataLoader(
+            dataset,
+            batch_size = batch_size,
+            shuffle = True,
+            drop_last = True,
+            num_workers = 4,
+            pin_memory = self.accelerator.device.type == 'cuda'
+        )
 
         self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)
 
@@ -1339,12 +1344,8 @@ class Trainer(Module):
                 if self.return_loss_breakdown:
                     loss, loss_breakdown = self.model(data, return_loss_breakdown = True)
                     self.log(loss_breakdown._asdict(), step = step)
-
-                    breakdown_str = ' | '.join(f'{k}: {v.item() if is_tensor(v) else v:.3f}' for k, v in loss_breakdown._asdict().items())
-                    self.accelerator.print(f'[{step}] {breakdown_str}')
                 else:
                     loss = self.model(data)
-                    self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
 
                 self.accelerator.backward(loss)
 
@@ -1365,6 +1366,14 @@ class Trainer(Module):
                 if self.accelerator.sync_gradients:
                     self.ema_model.update()
 
+            # Enqueue backward, optimizer and EMA work before synchronizing for
+            # the per-step console output.
+            if self.return_loss_breakdown:
+                breakdown_str = ' | '.join(f'{k}: {v.item() if is_tensor(v) else v:.3f}' for k, v in loss_breakdown._asdict().items())
+                self.accelerator.print(f'[{step}] {breakdown_str}')
+            else:
+                self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
+
             self.accelerator.wait_for_everyone()
 
             if self.is_main:
diff --git a/train_oxford.py b/train_oxford.py
index aa54174..71ef633 100644
--- a/train_oxford.py
+++ b/train_oxford.py
@@ -41,6 +41,9 @@ def train(
     results_folder = './results',
     clear_results_folder = True
 ):
+    if torch.cuda.is_available():
+        torch.backends.cudnn.benchmark = True
+
     flowers_dataset = OxfordFlowersDataset(
         image_size = image_size
     )
@@ -60,9 +63,17 @@ def train(
         sample_temperature = sample_temperature,
         save_results_every = save_results_every,
         results_folder = results_folder,
-        clear_results_folder = clear_results_folder
+        clear_results_folder = clear_results_folder,
+        accelerate_kwargs = dict(mixed_precision = 'bf16'),
+        adam_kwargs = dict(fused = torch.cuda.is_available())
     )
 
+    # Compile only the training network's forward, leaving checkpoint keys and
+    # the EMA sampling model unchanged.
+    if trainer.accelerator.device.type == 'cuda':
+        network = trainer.accelerator.unwrap_model(trainer.model).model
+        network.forward = torch.compile(network.forward)
+
     trainer()
 
 if __name__ == '__main__':
```
