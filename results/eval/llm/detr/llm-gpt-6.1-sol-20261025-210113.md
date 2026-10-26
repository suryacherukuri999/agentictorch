# AgenticTorch report: detr

**Result: no change was kept.**

- Command: `tar xf data/coco/subset.tar -C data/coco && python main.py --coco_path data/coco --output_dir out`
- Baseline: 211.2 ms/step over 2 runs
- Model: gpt-6.1-sol, 23 turns, 516142 input and 4178 output tokens

Enabled bfloat16 autocast for training forwards with float32 matching/losses, dynamic-shape torch.compile for the training model while retaining original checkpoint keys and full-precision evaluation, and fused CUDA AdamW. Disabled unused attention weights to enable efficient SDPA. Added pinned NestedTensor/data-loader memory and nonblocking device copies. Batched logging scalar transfers while retaining every-step finite-loss checks and all existing metrics, removed the box-count GPU round trip, and made CUDA box-validity assertions asynchronous. Training settings, data sampling, print/checkpoint formats and schedules are unchanged. No execution or benchmarking was performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 211.2 ms/step | failed (timed out) | reverted: run failed: timed out |

## Diff of the kept changes

```diff
diff --git a/engine.py b/engine.py
index ac5ea6f..74e24cf 100644
--- a/engine.py
+++ b/engine.py
@@ -26,10 +26,16 @@ def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
     print_freq = 10
 
     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
-        samples = samples.to(device)
-        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
-
-        outputs = model(samples)
+        samples = samples.to(device, non_blocking=True)
+        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
+
+        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
+                            enabled=device.type == 'cuda'):
+            outputs = model(samples)
+        # Matching (including cdist) and box losses need float32 precision.
+        outputs = {k: ([{name: value.float() for name, value in aux.items()}
+                        for aux in v] if k == 'aux_outputs' else v.float())
+                   for k, v in outputs.items()}
         loss_dict = criterion(outputs, targets)
         weight_dict = criterion.weight_dict
         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
@@ -42,21 +48,27 @@ def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
-        loss_value = losses_reduced_scaled.item()
+        # Fetch all logging scalars together, rather than synchronizing once
+        # per meter (and again after the optimizer). Keep the per-step check.
+        log_values = dict(loss=losses_reduced_scaled,
+                          **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
+        log_values['class_error'] = loss_dict_reduced['class_error']
+        log_values = dict(zip(log_values, torch.stack(
+            [v.detach() for v in log_values.values()]).cpu().tolist()))
+        loss_value = log_values['loss']
 
         if not math.isfinite(loss_value):
             print("Loss is {}, stopping training".format(loss_value))
             print(loss_dict_reduced)
             sys.exit(1)
 
-        optimizer.zero_grad()
+        optimizer.zero_grad(set_to_none=True)
         losses.backward()
         if max_norm > 0:
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
         optimizer.step()
 
-        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
-        metric_logger.update(class_error=loss_dict_reduced['class_error'])
+        metric_logger.update(**log_values)
         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
     # gather the stats from all processes
     metric_logger.synchronize_between_processes()
@@ -86,8 +98,8 @@ def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, out
         )
 
     for samples, targets in metric_logger.log_every(data_loader, 10, header):
-        samples = samples.to(device)
-        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
+        samples = samples.to(device, non_blocking=True)
+        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
 
         outputs = model(samples)
         loss_dict = criterion(outputs, targets)
@@ -99,10 +111,12 @@ def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, out
                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                       for k, v in loss_dict_reduced.items()}
-        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
-                             **loss_dict_reduced_scaled,
-                             **loss_dict_reduced_unscaled)
-        metric_logger.update(class_error=loss_dict_reduced['class_error'])
+        log_values = dict(loss=sum(loss_dict_reduced_scaled.values()),
+                          **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
+        log_values['class_error'] = loss_dict_reduced['class_error']
+        log_values = dict(zip(log_values, torch.stack(
+            list(log_values.values())).cpu().tolist()))
+        metric_logger.update(**log_values)
 
         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
         results = postprocessors['bbox'](outputs, orig_target_sizes)
diff --git a/main.py b/main.py
index e5f9eff..f460306 100644
--- a/main.py
+++ b/main.py
@@ -136,7 +136,8 @@ def main(args):
         },
     ]
     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
-                                  weight_decay=args.weight_decay)
+                                  weight_decay=args.weight_decay,
+                                  fused=device.type == 'cuda')
     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
 
     dataset_train = build_dataset(image_set='train', args=args)
@@ -153,9 +154,11 @@ def main(args):
         sampler_train, args.batch_size, drop_last=True)
 
     data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
-                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
+                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
+                                   pin_memory=device.type == 'cuda')
     data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
-                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
+                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
+                                 pin_memory=device.type == 'cuda')
 
     if args.dataset_file == "coco_panoptic":
         # We also evaluate AP during panoptic training, on original coco DS
@@ -188,13 +191,18 @@ def main(args):
             utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
         return
 
+    # Compile only the training forward; keep the original module for evaluation
+    # and state_dicts so checkpoint keys and evaluation precision stay unchanged.
+    # COCO augmentation produces different image sizes from batch to batch.
+    training_model = torch.compile(model, dynamic=True) if device.type == 'cuda' else model
+
     print("Start training")
     start_time = time.time()
     for epoch in range(args.start_epoch, args.epochs):
         if args.distributed:
             sampler_train.set_epoch(epoch)
         train_stats = train_one_epoch(
-            model, criterion, data_loader_train, optimizer, device, epoch,
+            training_model, criterion, data_loader_train, optimizer, device, epoch,
             args.clip_max_norm)
         lr_scheduler.step()
         if args.output_dir:
diff --git a/models/detr.py b/models/detr.py
index 23c2376..5444fe0 100644
--- a/models/detr.py
+++ b/models/detr.py
@@ -226,10 +226,14 @@ class SetCriterion(nn.Module):
 
         # Compute the average number of target boxes accross all nodes, for normalization purposes
         num_boxes = sum(len(t["labels"]) for t in targets)
-        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
         if is_dist_avail_and_initialized():
+            num_boxes = torch.as_tensor(num_boxes, dtype=torch.float,
+                                        device=next(iter(outputs.values())).device)
             torch.distributed.all_reduce(num_boxes)
-        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
+            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
+        else:
+            # Target lengths are already host metadata; no GPU round trip needed.
+            num_boxes = max(num_boxes, 1)
 
         # Compute all the requested losses
         losses = {}
diff --git a/models/transformer.py b/models/transformer.py
index dcd5367..717f05f 100644
--- a/models/transformer.py
+++ b/models/transformer.py
@@ -153,7 +153,7 @@ class TransformerEncoderLayer(nn.Module):
                      pos: Optional[Tensor] = None):
         q = k = self.with_pos_embed(src, pos)
         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
-                              key_padding_mask=src_key_padding_mask)[0]
+                              key_padding_mask=src_key_padding_mask, need_weights=False)[0]
         src = src + self.dropout1(src2)
         src = self.norm1(src)
         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
@@ -168,7 +168,7 @@ class TransformerEncoderLayer(nn.Module):
         src2 = self.norm1(src)
         q = k = self.with_pos_embed(src2, pos)
         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
-                              key_padding_mask=src_key_padding_mask)[0]
+                              key_padding_mask=src_key_padding_mask, need_weights=False)[0]
         src = src + self.dropout1(src2)
         src2 = self.norm2(src)
         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
@@ -218,13 +218,13 @@ class TransformerDecoderLayer(nn.Module):
                      query_pos: Optional[Tensor] = None):
         q = k = self.with_pos_embed(tgt, query_pos)
         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
-                              key_padding_mask=tgt_key_padding_mask)[0]
+                              key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
         tgt = tgt + self.dropout1(tgt2)
         tgt = self.norm1(tgt)
         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
-                                   key_padding_mask=memory_key_padding_mask)[0]
+                                   key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
         tgt = tgt + self.dropout2(tgt2)
         tgt = self.norm2(tgt)
         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
@@ -242,13 +242,13 @@ class TransformerDecoderLayer(nn.Module):
         tgt2 = self.norm1(tgt)
         q = k = self.with_pos_embed(tgt2, query_pos)
         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
-                              key_padding_mask=tgt_key_padding_mask)[0]
+                              key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
         tgt = tgt + self.dropout1(tgt2)
         tgt2 = self.norm2(tgt)
         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
-                                   key_padding_mask=memory_key_padding_mask)[0]
+                                   key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
         tgt = tgt + self.dropout2(tgt2)
         tgt2 = self.norm3(tgt)
         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
diff --git a/util/box_ops.py b/util/box_ops.py
index 9c088e5..8255afd 100644
--- a/util/box_ops.py
+++ b/util/box_ops.py
@@ -48,8 +48,14 @@ def generalized_box_iou(boxes1, boxes2):
     """
     # degenerate boxes gives inf / nan results
     # so do an early check
-    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
-    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
+    if boxes1.is_cuda:
+        torch._assert_async((boxes1[:, 2:] >= boxes1[:, :2]).all())
+    else:
+        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
+    if boxes2.is_cuda:
+        torch._assert_async((boxes2[:, 2:] >= boxes2[:, :2]).all())
+    else:
+        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
     iou, union = box_iou(boxes1, boxes2)
 
     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
diff --git a/util/misc.py b/util/misc.py
index dfa9fb5..43acf15 100644
--- a/util/misc.py
+++ b/util/misc.py
@@ -286,17 +286,19 @@ class NestedTensor(object):
         self.tensors = tensors
         self.mask = mask
 
-    def to(self, device):
-        # type: (Device) -> NestedTensor # noqa
-        cast_tensor = self.tensors.to(device)
+    def to(self, device, non_blocking=False):
+        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
         mask = self.mask
         if mask is not None:
-            assert mask is not None
-            cast_mask = mask.to(device)
+            cast_mask = mask.to(device, non_blocking=non_blocking)
         else:
             cast_mask = None
         return NestedTensor(cast_tensor, cast_mask)
 
+    def pin_memory(self):
+        return NestedTensor(self.tensors.pin_memory(),
+                            self.mask.pin_memory() if self.mask is not None else None)
+
     def decompose(self):
         return self.tensors, self.mask
```
