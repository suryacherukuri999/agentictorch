# AgenticTorch report: jointbert

**Result: 2.72x faster, loss within noise** on an NVIDIA L4.

- Command: `TOKENIZERS_PARALLELISM=false HF_HOME=hf HF_HUB_OFFLINE=1 python main.py --task atis --model_type bert --model_dir atis_model --do_train`
- Baseline: 139.8 ms/step over 2 runs
- Model: gpt-6.1-sol, 23 turns, 213247 input and 1288 output tokens

## Diagnosis

```
profiled 10 steps, 256.0 ms/step under the profiler
GPU busy 45% of the time
waiting on the DataLoader 4% of the time
host-device syncs per step: 10.0 (copies to the CPU: 5.0)
  5.0 per step at trainer.py:80: batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
  1.0 per step at model/modeling_jointbert.py:25: outputs = self.bert(input_ids, attention_mask=attention_mask,
  1.0 per step at model/modeling_jointbert.py:54: active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
  1.0 per step at model/modeling_jointbert.py:55: active_labels = slot_labels_ids.view(-1)[active_loss]
  1.0 per step at trainer.py:94: loss.backward()
  1.0 per step at trainer.py:96: tr_loss += loss.item()
top ops by GPU time:
  35.1%  aten::mm  backward of model/modeling_jointbert.py:25: outputs = self.bert(input_ids, attention_mask=attention_mask,
  18.9%  aten::addmm  at model/modeling_jointbert.py:25: outputs = self.bert(input_ids, attention_mask=attention_mask,
   6.1%  aten::_foreach_addcdiv_  at trainer.py:100: optimizer.step()
   4.8%  aten::_foreach_lerp_  at trainer.py:100: optimizer.step()
   4.8%  aten::_foreach_addcmul_  at trainer.py:100: optimizer.step()
   3.3%  aten::_foreach_add_  at trainer.py:100: optimizer.step()
   3.3%  aten::_foreach_mul_  at trainer.py:100: optimizer.step()
   3.3%  aten::_foreach_div_  at trainer.py:100: optimizer.step()
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Use CUDA bfloat16 autocast for training forward and loss to accelerate BERT matmuls. | 141.2 ms/step | 80.8 ms/step | kept: 1.75x faster, loss within noise |
| 2 | Enable fused CUDA AdamW to reduce optimizer kernel launches and memory traffic without changing hyperparameters. | 81.9 ms/step | 61.8 ms/step | kept: 1.33x faster, loss within noise |
| 3 | Replace dynamic boolean indexing in slot cross entropy with ignore-index targets to avoid CUDA synchronization. | 61.4 ms/step | 64.8 ms/step | reverted: 0.95x is within noise |
| 4 | Compile the BERT backbone in place to fuse operations and reduce Python/kernel-launch overhead, preserving checkpoint structure. | 60.4 ms/step | 51.6 ms/step | kept: 1.17x faster, loss within noise |
| 5 | Use pinned training batches and non-blocking device copies to avoid five blocking host-to-device transfers per step. | 50.8 ms/step | 49.6 ms/step | reverted: 1.02x is within noise |

## Diff of the kept changes

```diff
diff --git a/trainer.py b/trainer.py
index 014a223..7df03e0 100644
--- a/trainer.py
+++ b/trainer.py
@@ -38,6 +38,8 @@ class Trainer(object):
         self.model.to(self.device)
 
     def train(self):
+        if self.device == "cuda" and self.args.model_type == "bert":
+            self.model.bert.compile()
         train_sampler = RandomSampler(self.train_dataset)
         train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
 
@@ -54,7 +56,8 @@ class Trainer(object):
              'weight_decay': self.args.weight_decay},
             {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
         ]
-        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
+        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,
+                          eps=self.args.adam_epsilon, fused=self.device == "cuda")
         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
 
         # Train!
@@ -85,8 +88,10 @@ class Trainer(object):
                           'slot_labels_ids': batch[4]}
                 if self.args.model_type != 'distilbert':
                     inputs['token_type_ids'] = batch[2]
-                outputs = self.model(**inputs)
-                loss = outputs[0]
+                with torch.autocast(device_type=self.device, dtype=torch.bfloat16,
+                                    enabled=self.device == "cuda"):
+                    outputs = self.model(**inputs)
+                    loss = outputs[0]
 
                 if self.args.gradient_accumulation_steps > 1:
                     loss = loss / self.args.gradient_accumulation_steps
```

## Sources the agent read

- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP examples: Typical Mixed Precision Training](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#typical-mixed-precision-training)
- [PyTorch AMP examples: Gradient penalty](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-penalty)
- [PyTorch CUDA semantics: Usage with torch.cuda.amp](https://docs.pytorch.org/docs/2.14/notes/cuda.html#usage-with-torch-cuda-amp)
- [torch.amp: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html#automatic-mixed-precision-package-torch-amp)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [PyTorch AMP recipe: Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#automatic-mixed-precision)
- [torch.compile tutorial: Demonstrating Speedups](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups)
- [Scaled dot product attention: Using SDPA with torch.compile](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#using-sdpa-with-torch-compile)
- [PyTorch GitHub issues: [feature request] index_select is very slow on sparse tensors (and my proposed algorithm to fix it)](https://github.com/pytorch/pytorch/issues/61788)
- [PyTorch CUDA semantics: Data Dependent Control Flow](https://docs.pytorch.org/docs/2.14/notes/cuda.html#data-dependent-control-flow)
- [torch.compile tutorial: Basic Usage](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#basic-usage)
- [torch.compile tutorial: Graph Breaks](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#graph-breaks)
- [torch.compile tutorial: Benefits over TorchScript](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#benefits-over-torchscript)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [Pinned memory and non_blocking: non_blocking=True](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#non-blocking-true)
- [Pinned memory and non_blocking: Asynchronous vs. Synchronous Operations with non_blocking=True (CUDA cudaMemcpyAsync)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)
- [PyTorch CUDA semantics: Use pinned memory buffers](https://docs.pytorch.org/docs/2.14/notes/cuda.html#use-pinned-memory-buffers)
