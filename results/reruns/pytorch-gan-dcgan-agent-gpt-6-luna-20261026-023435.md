# AgenticTorch report: pytorch-gan-dcgan

**Result: 2.04x faster, loss within noise** on an NVIDIA L4.

- Command: `cd implementations/dcgan && python dcgan.py`
- Baseline: 29.0 ms/step over 2 runs
- Model: gpt-6-luna, 22 turns, 222907 input and 1584 output tokens

## Diagnosis

```
profiled 10 steps, 164.6 ms/step under the profiler
GPU busy 3% of the time
waiting on the DataLoader 70% of the time
host-device syncs per step: 4.0 (copies to the CPU: 2.0)
  2.0 per step at implementations/dcgan/dcgan.py:187: % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
  1.0 per step at implementations/dcgan/dcgan.py:151: real_imgs = Variable(imgs.type(Tensor))
  1.0 per step at implementations/dcgan/dcgan.py:160: z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
top ops by GPU time:
  35.9%  aten::convolution_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
  13.8%  aten::cudnn_convolution  at implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
  12.7%  aten::convolution_backward  backward of implementations/dcgan/dcgan.py:95: out = self.model(img)
   3.6%  aten::cudnn_convolution  at implementations/dcgan/dcgan.py:95: out = self.model(img)
   3.4%  aten::cudnn_batch_norm_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
   3.3%  aten::upsample_nearest2d_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
   2.4%  aten::convolution_backward  at implementations/dcgan/dcgan.py:95: out = self.model(img)
   1.6%  aten::upsample_nearest2d  at implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
```

The baseline was spending most of its time waiting for MNIST batches. Configuring the DataLoader with 8 workers, pinned memory, and persistent workers, and using non-blocking CUDA transfers reduced the measured step time from 29.0 ms to 14.8 ms (about 2.15× faster). Direct CUDA noise generation, bfloat16 autocast, and fused Adam did not show reliable additional gains and were reverted.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Use 8 DataLoader workers with pinned memory and persistent workers, and asynchronously transfer image batches to CUDA to reduce the measured input wait. | 31.7 ms/step | 14.8 ms/step | kept: 2.15x faster, loss within noise |
| 2 | Generate latent noise directly on the target device instead of generating it with NumPy on the CPU and copying it to CUDA. | 16.6 ms/step | 14.7 ms/step | reverted: 1.13x is within noise |
| 3 | Use bfloat16 autocast for generator and discriminator convolution/linear operations while retaining float32 BCELoss for safe, numerically stable loss computation. | 15.5 ms/step | 16.3 ms/step | reverted: 0.95x is within noise |
| 4 | Enable fused CUDA Adam optimizers to reduce optimizer kernel-launch overhead without changing optimizer settings. | 13.8 ms/step | 13.4 ms/step | reverted: 1.03x is within noise |

## Diff of the kept changes

```diff
diff --git a/implementations/dcgan/dcgan.py b/implementations/dcgan/dcgan.py
index 46e64c0..b31ae5b 100644
--- a/implementations/dcgan/dcgan.py
+++ b/implementations/dcgan/dcgan.py
@@ -128,6 +128,9 @@ dataloader = torch.utils.data.DataLoader(
     ),
     batch_size=opt.batch_size,
     shuffle=True,
+    num_workers=opt.n_cpu,
+    pin_memory=cuda,
+    persistent_workers=opt.n_cpu > 0,
 )
 
 # Optimizers
@@ -148,7 +151,7 @@ for epoch in range(opt.n_epochs):
         fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
 
         # Configure input
-        real_imgs = Variable(imgs.type(Tensor))
+        real_imgs = Variable(imgs.to(device="cuda", non_blocking=True) if cuda else imgs)
 
         # -----------------
         #  Train Generator
```

## Sources the agent read

- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [torch.utils.data: Memory Pinning](https://docs.pytorch.org/docs/2.14/data.html#memory-pinning)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [PyTorch CUDA semantics: Mixing different CUDA system allocators in the same program](https://docs.pytorch.org/docs/2.14/notes/cuda.html#mixing-different-cuda-system-allocators-in-the-same-program)
- [PyTorch CUDA semantics: CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-semantics)
- [PyTorch performance tuning guide: Create tensors directly on the target device](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device)
- [PyTorch performance tuning guide: Avoid unnecessary CPU-GPU synchronization](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)
- [PyTorch GitHub issues: einsum running 100x slower than expected when controlling for number of ops](https://github.com/pytorch/pytorch/issues/41467)
- [torch.amp: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html#automatic-mixed-precision-package-torch-amp)
- [PyTorch AMP recipe: All together: “Automatic Mixed Precision”](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP examples: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#automatic-mixed-precision-examples)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [PyTorch performance tuning guide: Conclusion](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#conclusion)
- [PyTorch CUDA semantics: Usage with torch.cuda.amp](https://docs.pytorch.org/docs/2.14/notes/cuda.html#usage-with-torch-cuda-amp)
- [PyTorch GitHub issues: Training slowdown from 1.6 to 1.7.1](https://github.com/pytorch/pytorch/issues/50636)
