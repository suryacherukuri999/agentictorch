# AgenticTorch report: jointbert

**Result: 3.05x faster, loss within noise** on an NVIDIA L4.

- Command: `TOKENIZERS_PARALLELISM=false HF_HOME=hf HF_HUB_OFFLINE=1 python main.py --task atis --model_type bert --model_dir atis_model --do_train`
- Baseline: 140.0 ms/step over 2 runs
- Model: gpt-6.1-sol, 11 turns, 68143 input and 1567 output tokens

Enabled CUDA bfloat16 autocast when supported and compiled the non-CRF BERT training path while preserving the original model for evaluation and checkpoint saving. Replaced variable-sized masked slot-logit gathers with ignore-index labels, eliminating a synchronization and compilation obstacle without changing loss semantics. Enabled fused CUDA AdamW, explicit set-to-none gradient clearing, and pinned-memory/nonblocking batch transfers. Deferred training-loss scalar synchronization until training finishes, using a detached float64 device accumulator. Kept batch sizes, schedule, step counts, logging/evaluation/checkpoint cadence, and checkpoint format unchanged. No execution or benchmarking was performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 140.0 ms/step | 45.9 ms/step | kept: 3.05x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/model/modeling_jointbert.py b/model/modeling_jointbert.py
index 2ecb31b..39ec372 100644
--- a/model/modeling_jointbert.py
+++ b/model/modeling_jointbert.py
@@ -51,9 +51,11 @@ class JointBERT(BertPreTrainedModel):
                 # Only keep active parts of the loss
                 if attention_mask is not None:
                     active_loss = attention_mask.view(-1) == 1
-                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
-                    active_labels = slot_labels_ids.view(-1)[active_loss]
-                    slot_loss = slot_loss_fct(active_logits, active_labels)
+                    # Ignoring masked labels gives the same mean over valid targets,
+                    # without a variable-sized gather (and its CUDA synchronization).
+                    active_labels = slot_labels_ids.view(-1).masked_fill(
+                        ~active_loss, self.args.ignore_index)
+                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), active_labels)
                 else:
                     slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
             total_loss += self.args.slot_loss_coef * slot_loss
diff --git a/trainer.py b/trainer.py
index 014a223..d1b633a 100644
--- a/trainer.py
+++ b/trainer.py
@@ -38,8 +38,16 @@ class Trainer(object):
         self.model.to(self.device)
 
     def train(self):
+        use_cuda = self.device == "cuda"
+        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
         train_sampler = RandomSampler(self.train_dataset)
-        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
+        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
+                                      batch_size=self.args.train_batch_size, pin_memory=use_cuda)
+        # Keep the original module for evaluation and save_pretrained, so checkpoint
+        # keys and the saved model format are unchanged.
+        train_model = self.model
+        if use_cuda and self.args.model_type == 'bert' and not self.args.use_crf:
+            train_model = torch.compile(self.model)
 
         if self.args.max_steps > 0:
             t_total = self.args.max_steps
@@ -54,7 +62,8 @@ class Trainer(object):
              'weight_decay': self.args.weight_decay},
             {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
         ]
-        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
+        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,
+                          eps=self.args.adam_epsilon, fused=use_cuda)
         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
 
         # Train!
@@ -68,8 +77,9 @@ class Trainer(object):
         logger.info("  Save steps = %d", self.args.save_steps)
 
         global_step = 0
-        tr_loss = 0.0
-        self.model.zero_grad()
+        # Accumulate detached losses on device; only synchronize at the end.
+        tr_loss = torch.zeros((), device=self.device, dtype=torch.float64)
+        self.model.zero_grad(set_to_none=True)
 
         train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
 
@@ -77,7 +87,7 @@ class Trainer(object):
             epoch_iterator = tqdm(train_dataloader, desc="Iteration")
             for step, batch in enumerate(epoch_iterator):
                 self.model.train()
-                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
+                batch = tuple(t.to(self.device, non_blocking=use_cuda) for t in batch)  # GPU or CPU
 
                 inputs = {'input_ids': batch[0],
                           'attention_mask': batch[1],
@@ -85,21 +95,22 @@ class Trainer(object):
                           'slot_labels_ids': batch[4]}
                 if self.args.model_type != 'distilbert':
                     inputs['token_type_ids'] = batch[2]
-                outputs = self.model(**inputs)
-                loss = outputs[0]
+                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=use_bf16):
+                    outputs = train_model(**inputs)
+                    loss = outputs[0]
 
                 if self.args.gradient_accumulation_steps > 1:
                     loss = loss / self.args.gradient_accumulation_steps
 
                 loss.backward()
 
-                tr_loss += loss.item()
+                tr_loss.add_(loss.detach())
                 if (step + 1) % self.args.gradient_accumulation_steps == 0:
                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
 
                     optimizer.step()
                     scheduler.step()  # Update learning rate schedule
-                    self.model.zero_grad()
+                    self.model.zero_grad(set_to_none=True)
                     global_step += 1
 
                     if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
@@ -116,7 +127,7 @@ class Trainer(object):
                 train_iterator.close()
                 break
 
-        return global_step, tr_loss / global_step
+        return global_step, tr_loss.item() / global_step
 
     def evaluate(self, mode):
         if mode == 'test':
@@ -127,7 +138,8 @@ class Trainer(object):
             raise Exception("Only dev and test dataset available")
 
         eval_sampler = SequentialSampler(dataset)
-        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
+        eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
+                                     batch_size=self.args.eval_batch_size, pin_memory=self.device == "cuda")
 
         # Eval!
         logger.info("***** Running evaluation on %s dataset *****", mode)
@@ -143,7 +155,7 @@ class Trainer(object):
         self.model.eval()
 
         for batch in tqdm(eval_dataloader, desc="Evaluating"):
-            batch = tuple(t.to(self.device) for t in batch)
+            batch = tuple(t.to(self.device, non_blocking=self.device == "cuda") for t in batch)
             with torch.no_grad():
                 inputs = {'input_ids': batch[0],
                           'attention_mask': batch[1],
```
