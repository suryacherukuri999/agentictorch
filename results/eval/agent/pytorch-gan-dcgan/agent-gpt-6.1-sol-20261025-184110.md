# AgenticTorch report: pytorch-gan-dcgan

**Result: 2.15x faster, loss within noise** on an NVIDIA L4.

- Command: `cd implementations/dcgan && python dcgan.py`
- Baseline: 18.4 ms/step over 2 runs
- Model: gpt-6.1-sol, 26 turns, 239905 input and 1721 output tokens

## Diagnosis

```
profiled 10 steps, 174.9 ms/step under the profiler
GPU busy 3% of the time
waiting on the DataLoader 71% of the time
host-device syncs per step: 4.0 (copies to the CPU: 2.0)
  2.0 per step at implementations/dcgan/dcgan.py:187: % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
  1.0 per step at implementations/dcgan/dcgan.py:151: real_imgs = Variable(imgs.type(Tensor))
  1.0 per step at implementations/dcgan/dcgan.py:160: z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
top ops by GPU time:
  36.0%  aten::convolution_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
  13.8%  aten::cudnn_convolution  at implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
  12.7%  aten::convolution_backward  backward of implementations/dcgan/dcgan.py:95: out = self.model(img)
   3.6%  aten::cudnn_convolution  at implementations/dcgan/dcgan.py:95: out = self.model(img)
   3.4%  aten::cudnn_batch_norm_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
   3.3%  aten::upsample_nearest2d_backward  backward of implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
   2.4%  aten::convolution_backward  at implementations/dcgan/dcgan.py:95: out = self.model(img)
   1.6%  aten::upsample_nearest2d  at implementations/dcgan/dcgan.py:69: img = self.conv_blocks(out)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Enable asynchronous MNIST loading with existing n_cpu workers and keep workers alive across epochs. | 28.2 ms/step | 18.7 ms/step | kept: 1.56x faster, loss within noise |
| 2 | Use DataLoader pinned memory and nonblocking image transfers to avoid a synchronous host-to-device copy. | 14.3 ms/step | 13.5 ms/step | kept: 1.06x faster, loss within noise |
| 3 | Use channels-last convolution weights and activations in FP32, with layout-safe discriminator flattening. | 14.0 ms/step | 14.6 ms/step | reverted: 0.96x is within noise |
| 4 | Use fused CUDA Adam to reduce optimizer kernel launches while preserving Adam hyperparameters. | 15.8 ms/step | 14.3 ms/step | kept: 1.10x faster, loss within noise |
| 5 | Skip discarded discriminator parameter gradients in the generator phase, preserving input gradients and all forward passes. | 12.7 ms/step | 12.7 ms/step | reverted: 1.00x is within noise |

## Diff of the kept changes

```diff
diff --git a/implementations/dcgan/dcgan.py b/implementations/dcgan/dcgan.py
index 46e64c0..956f69b 100644
--- a/implementations/dcgan/dcgan.py
+++ b/implementations/dcgan/dcgan.py
@@ -128,11 +128,14 @@ dataloader = torch.utils.data.DataLoader(
     ),
     batch_size=opt.batch_size,
     shuffle=True,
+    num_workers=opt.n_cpu,
+    persistent_workers=opt.n_cpu > 0,
+    pin_memory=cuda,
 )
 
 # Optimizers
-optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
-optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
+optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), fused=cuda)
+optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), fused=cuda)
 
 Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
 
@@ -148,7 +151,7 @@ for epoch in range(opt.n_epochs):
         fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
 
         # Configure input
-        real_imgs = Variable(imgs.type(Tensor))
+        real_imgs = imgs.to(device="cuda" if cuda else "cpu", dtype=torch.float32, non_blocking=True)
 
         # -----------------
         #  Train Generator
```

## Sources the agent read

- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [Pinned memory and non_blocking: pin_memory()](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#pin-memory)
- [Pinned memory and non_blocking: non_blocking=True](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#non-blocking-true)
- [Channels last memory format: Memory Format API](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api)
- [Channels last memory format: Converting existing models](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models)
- [Channels last memory format: Channels Last Memory Format in PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html#channels-last-memory-format-in-pytorch)
- [PyTorch performance tuning guide: Enable channels_last memory format for computer vision models](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models)
- [PyTorch CUDA semantics: TensorFloat-32 (TF32) on Ampere (and later) devices](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch AMP recipe: Default Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#default-precision)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
