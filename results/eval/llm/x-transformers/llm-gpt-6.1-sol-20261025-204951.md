# AgenticTorch report: x-transformers

**Result: 5.61x faster, loss within noise** on an NVIDIA L4.

- Command: `python train_enwik8.py`
- Baseline: 705.3 ms/step over 2 runs
- Model: gpt-6.1-sol, 14 turns, 156297 input and 1667 output tokens

Optimized train_enwik8.py with CUDA bfloat16 mixed precision when supported, SDPA attention, fused CUDA Adam, and a compiled training loss path while leaving generation eager. Dataset samples now stay on CPU for pinned batch collation and a single nonblocking device transfer per batch; sampling order and RNG behavior remain unchanged. Reused scalar loss values for printing/logging and made set-to-none gradient clearing explicit. Kept model dimensions, data, batch size, learning rate, step counts, and logging/generation cadence unchanged. No execution or benchmarking was available.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 705.3 ms/step | 125.8 ms/step | kept: 5.61x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/train_enwik8.py b/train_enwik8.py
index f607a4e..a479e4b 100644
--- a/train_enwik8.py
+++ b/train_enwik8.py
@@ -58,7 +58,8 @@ def train(
     read_gate_rank = 64,
     attn_aggregated_residuals = False
 ):
-    accelerator = Accelerator(cpu=cpu)
+    use_bf16 = not cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
+    accelerator = Accelerator(cpu=cpu, mixed_precision='bf16' if use_bf16 else 'no')
     device = accelerator.device
 
     generate_length = default(generate_length, seq_len)
@@ -72,6 +73,7 @@ def train(
             dim = 512,
             depth = 6,
             heads = 8,
+            attn_flash = True,
             rotary_pos_emb = False,
             polar_pos_emb = True,
             pre_and_post_norm = not (gated_multi_residual or attn_aggregated_residuals),
@@ -99,19 +101,20 @@ def train(
         def __getitem__(self, index):
             rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
             full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
-            return full_seq.to(device)
+            # Collate on the host so each batch needs only one device transfer.
+            return full_seq
 
         def __len__(self):
             return self.data.size(0) // self.seq_len
 
     train_dataset = TextSamplerDataset(data_train, seq_len)
     val_dataset   = TextSamplerDataset(data_val, seq_len)
-    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
-    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))
+    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True, pin_memory = device.type == 'cuda'))
+    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True, pin_memory = device.type == 'cuda'))
 
     # optimizer
 
-    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
+    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, fused=device.type == 'cuda')
 
     # experiment
 
@@ -122,31 +125,38 @@ def train(
         model, optim, train_loader, val_loader
     )
 
+    # Compile only the loss path; generation keeps its eager, variable-length cache.
+    training_model = torch.compile(model) if device.type == 'cuda' else model
+
     # training
 
     for i in tqdm.tqdm(range(num_batches), mininterval=10., desc='training'):
         model.train()
 
         for _ in range(gradient_accumulate_every):
-            loss = model(next(train_loader))
+            batch = next(train_loader).to(device, non_blocking=True)
+            loss = training_model(batch)
             accelerator.backward(loss / gradient_accumulate_every)
 
-        print(f'training loss: {loss.item()}')
+        loss_value = loss.item()
+        print(f'training loss: {loss_value}')
         if accelerator.is_main_process:
-            wandb.log(dict(loss = loss.item()))
+            wandb.log(dict(loss = loss_value))
 
         accelerator.clip_grad_norm_(model.parameters(), 0.5)
         optim.step()
-        optim.zero_grad()
+        optim.zero_grad(set_to_none=True)
 
         if i % validate_every == 0:
             model.eval()
             with torch.no_grad():
-                loss = model(next(val_loader))
+                batch = next(val_loader).to(device, non_blocking=True)
+                loss = model(batch)
 
-                print(f'validation loss: {loss.item()}')
+                loss_value = loss.item()
+                print(f'validation loss: {loss_value}')
                 if accelerator.is_main_process:
-                    wandb.log(dict(valid_loss = loss.item()))
+                    wandb.log(dict(valid_loss = loss_value))
 
         if i % generate_every == 0:
             model.eval()
@@ -155,7 +165,7 @@ def train(
             print(f'%s \n\n %s' % (prime, '*' * 100))
 
             sample = model.generate(
-                prompts = inp,
+                prompts = inp.to(device),
                 seq_len = generate_length,
                 cache_kv = True
             )
```
