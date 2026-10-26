# AgenticTorch report: pytorch-yolov3

**Result: 2.10x faster, loss within noise** on an NVIDIA L4.

- Command: `python -m pytorchyolo.train --model config/yolov3.cfg --data data/coco128/coco128.data --evaluation_interval 1000 --checkpoint_interval 1000`
- Baseline: 560.1 ms/step over 2 runs
- Model: gpt-6.1-sol, 25 turns, 368770 input and 2014 output tokens

## Diagnosis

```
profiled 10 steps, 543.5 ms/step under the profiler
GPU busy 63% of the time
waiting on the DataLoader 15% of the time
host-device syncs per step: 23.0 (copies to the CPU: 11.0)
  6.0 per step at pytorchyolo/utils/loss.py:174: indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
  6.0 per step at pytorchyolo/utils/loss.py:110: t[range(num_targets), tcls[layer_index]] = 1
  3.0 per step at pytorchyolo/utils/loss.py:143: gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
  3.0 per step at pytorchyolo/utils/loss.py:155: t = t[j]
  2.0 per step at pytorchyolo/utils/utils.py:43: return tensor.detach().cpu()
  1.0 per step at pytorchyolo/train.py:161: targets = targets.to(device)
top ops by GPU time:
  40.4%  aten::convolution_backward  backward of pytorchyolo/models.py:210: x = module(x)
  13.9%  aten::cudnn_convolution  at pytorchyolo/models.py:210: x = module(x)
  12.2%  aten::cudnn_batch_norm_backward  backward of pytorchyolo/models.py:210: x = module(x)
   8.0%  aten::leaky_relu_backward  backward of pytorchyolo/models.py:210: x = module(x)
   6.1%  aten::cudnn_batch_norm  at pytorchyolo/models.py:210: x = module(x)
   4.3%  aten::leaky_relu  at pytorchyolo/models.py:210: x = module(x)
   3.2%  aten::add  at pytorchyolo/models.py:218: x = layer_outputs[-1] + layer_outputs[layer_i]
   2.9%  aten::add_  backward of pytorchyolo/models.py:210: x = module(x)
```

Reduced training from 560.1 to 252.3 ms/step (~2.22x faster, 55% less time), with benchmark loss within noise. Kept bfloat16 autocast for model execution while computing geometry/loss in FP32, channels-last convolution weights and inputs (using convolution-only conversion to preserve YOLO anchor buffers), and persistent DataLoader workers for fixed-resolution training to eliminate worker restarts on short COCO128 epochs. Loss synchronization cleanup was tested but reverted because its improvement was within noise. Model, batch size, optimizer schedule, step count, logging, checkpointing, and evaluation are unchanged.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Remove loss indexing/clamping CPU scalar synchronizations using GPU arange and grid dimensions from tensor shapes. | 562.0 ms/step | 556.1 ms/step | reverted: 1.01x is within noise |
| 2 | Use bfloat16 autocast for model forward and backward kernels while preserving FP32 box geometry and loss evaluation. | 564.1 ms/step | 419.8 ms/step | kept: 1.34x faster, loss within noise |
| 3 | Use channels-last convolution weights and image batches to accelerate bfloat16 CNN kernels; preserve YOLO's non-4D anchor buffers. | 392.4 ms/step | 369.6 ms/step | kept: 1.06x faster, loss within noise |
| 4 | Keep fixed-resolution training DataLoader workers alive across short COCO128 epochs, avoiding repeated process startup. | 377.4 ms/step | 252.3 ms/step | kept: 1.50x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/pytorchyolo/train.py b/pytorchyolo/train.py
index 2a1ae02..49b7243 100755
--- a/pytorchyolo/train.py
+++ b/pytorchyolo/train.py
@@ -51,6 +51,7 @@ def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_traini
         batch_size=batch_size,
         shuffle=True,
         num_workers=n_cpu,
+        persistent_workers=n_cpu > 0 and not multiscale_training,
         pin_memory=True,
         collate_fn=dataset.collate_fn,
         worker_init_fn=worker_seed_set)
@@ -98,6 +99,8 @@ def run():
     # ############
 
     model = load_model(args.model, args.pretrained_weights)
+    if device.type == "cuda":
+        model = torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
 
     # Print model
     if args.verbose:
@@ -157,11 +160,16 @@ def run():
         for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
             batches_done = len(dataloader) * epoch + batch_i
 
-            imgs = imgs.to(device, non_blocking=True)
+            imgs = imgs.to(device, non_blocking=True,
+                           memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
             targets = targets.to(device)
 
-            outputs = model(imgs)
+            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
+                                enabled=device.type == "cuda"):
+                outputs = model(imgs)
 
+            # Keep box geometry and loss evaluation in full precision.
+            outputs = [output.float() for output in outputs]
             loss, loss_components = compute_loss(outputs, targets, model)
 
             loss.backward()
```

## Sources the agent read

- [Pinned memory and non_blocking: Other copy directions (GPU -> CPU, CPU -> MPS)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps)
- [Pinned memory and non_blocking: non_blocking=True](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#non-blocking-true)
- [Pinned memory and non_blocking: Asynchronous vs. Synchronous Operations with non_blocking=True (CUDA cudaMemcpyAsync)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [PyTorch AMP recipe: Adding torch.autocast](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP recipe: All together: “Automatic Mixed Precision”](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision)
- [PyTorch AMP examples: Typical Mixed Precision Training](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#typical-mixed-precision-training)
- [Channels last memory format: Memory Format API](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api)
- [PyTorch performance tuning guide: Enable channels_last memory format for computer vision models](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models)
- [Channels last memory format: Performance Gains](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains)
- [PyTorch performance tuning guide: Enable cuDNN auto-tuner](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner)
- [Channels last memory format: Converting existing models](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models)
- [torch.utils.data: Memory Pinning](https://docs.pytorch.org/docs/2.14/data.html#memory-pinning)
- [torch.utils.data: torch.utils.data](https://docs.pytorch.org/docs/2.14/data.html#module-torch.utils.data)
- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [torch.utils.data: Single- and Multi-process Data Loading](https://docs.pytorch.org/docs/2.14/data.html#single-and-multi-process-data-loading)
