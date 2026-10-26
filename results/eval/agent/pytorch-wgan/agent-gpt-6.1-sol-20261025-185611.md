# AgenticTorch report: pytorch-wgan

**Result: 1.41x faster, loss within noise** on an NVIDIA L4.

- Command: `python main.py --model WGAN-GP --is_train True --download True --dataroot datasets/cifar --dataset cifar --generator_iters 40000 --cuda True --batch_size 64`
- Baseline: 78.5 ms/step over 2 runs
- Model: gpt-6.1-sol, 23 turns, 230007 input and 1788 output tokens

## Diagnosis

```
profiled 10 steps, 174.3 ms/step under the profiler
GPU busy 33% of the time
waiting on the DataLoader 59% of the time
host-device syncs per step: 7.4 (copies to the CPU: 2.2)
  3.2 per step at models/wgan_gradient_penalty.py:120: return Variable(arg).cuda(self.cuda_index)
  2.0 per step at models/wgan_gradient_penalty.py:193: print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real:
  1.0 per step at models/wgan_gradient_penalty.py:301: eta = eta.cuda(self.cuda_index)
  1.0 per step at models/wgan_gradient_penalty.py:321: prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
  0.2 per step at models/wgan_gradient_penalty.py:209: print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
top ops by GPU time:
  21.2%  aten::convolution_backward  backward of models/wgan_gradient_penalty.py:80: x = self.main_module(x)
  14.6%  aten::cudnn_batch_norm_backward  backward of models/wgan_gradient_penalty.py:80: x = self.main_module(x)
   8.0%  aten::cudnn_convolution  at models/wgan_gradient_penalty.py:80: x = self.main_module(x)
   7.4%  aten::cudnn_batch_norm  at models/wgan_gradient_penalty.py:80: x = self.main_module(x)
   7.1%  aten::convolution_backward  backward of models/wgan_gradient_penalty.py:44: x = self.main_module(x)
   4.2%  aten::cudnn_convolution_transpose  at models/wgan_gradient_penalty.py:44: x = self.main_module(x)
   2.4%  aten::cudnn_convolution  at models/wgan_gradient_penalty.py:365: for i, (images, _) in enumerate(data_loader):
   2.4%  aten::mul  (no repo line: library code)
```

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | Overlap data preparation using four persistent DataLoader workers and pinned memory; use non-blocking CUDA transfers. | 74.9 ms/step | 64.7 ms/step | kept: 1.16x faster, loss within noise |
| 2 | Disable autograd for generator forwards in critic updates, eliminating generator gradients that were discarded before its optimizer step. | 65.7 ms/step | 59.9 ms/step | kept: 1.10x faster, loss within noise |
| 3 | Enable cuDNN autotuning to select faster convolution algorithms for the fixed training shapes. | 60.9 ms/step | 57.9 ms/step | kept: 1.05x faster, loss within noise |
| 4 | Use fused CUDA Adam for both optimizers with unchanged learning rate, betas, and update semantics. | 58.1 ms/step | 55.7 ms/step | kept: 1.04x faster, loss within noise |
| 5 | Reduce gradient-penalty transfer overhead: asynchronously copy unexpanded CPU-sampled eta and allocate gradient seeds directly on the output device. | 54.2 ms/step | 53.8 ms/step | reverted: 1.01x is within noise |

## Diff of the kept changes

```diff
diff --git a/models/wgan_gradient_penalty.py b/models/wgan_gradient_penalty.py
index fee0c5b..a162c5c 100755
--- a/models/wgan_gradient_penalty.py
+++ b/models/wgan_gradient_penalty.py
@@ -103,8 +103,8 @@ class WGAN_GP(object):
         self.batch_size = 64
 
         # WGAN_gradient penalty uses ADAM
-        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
-        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
+        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2), fused=self.cuda)
+        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2), fused=self.cuda)
 
         # Set the logger
         self.logger = Logger('./logs')
@@ -117,7 +117,7 @@ class WGAN_GP(object):
 
     def get_torch_variable(self, arg):
         if self.cuda:
-            return Variable(arg).cuda(self.cuda_index)
+            return Variable(arg).cuda(self.cuda_index, non_blocking=True)
         else:
             return Variable(arg)
 
@@ -126,6 +126,8 @@ class WGAN_GP(object):
         if cuda_flag:
             self.cuda_index = 0
             self.cuda = True
+            # Training uses fixed convolution shapes; tune kernels once.
+            torch.backends.cudnn.benchmark = True
             self.D.cuda(self.cuda_index)
             self.G.cuda(self.cuda_index)
             print("Cuda enabled flag: {}".format(self.cuda))
@@ -177,7 +179,10 @@ class WGAN_GP(object):
                 # Train with fake images
                 z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
 
-                fake_images = self.G(z)
+                # The generator is not updated during critic training. Its
+                # forward still runs in training mode to preserve BatchNorm updates.
+                with torch.no_grad():
+                    fake_images = self.G(z)
                 d_loss_fake = self.D(fake_images)
                 d_loss_fake = d_loss_fake.mean()
                 d_loss_fake.backward(one)
diff --git a/utils/data_loader.py b/utils/data_loader.py
index cae393b..cb511df 100755
--- a/utils/data_loader.py
+++ b/utils/data_loader.py
@@ -46,7 +46,9 @@ def get_data_loader(args):
     assert train_dataset
     assert test_dataset
 
-    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
+    train_dataloader = data_utils.DataLoader(
+        train_dataset, batch_size=args.batch_size, shuffle=True,
+        num_workers=4, persistent_workers=True, pin_memory=bool(args.cuda))
     test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)
 
     return train_dataloader, test_dataloader
```

## Sources the agent read

- [PyTorch performance tuning guide: Enable asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [Pinned memory and non_blocking: Additional considerations](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#additional-considerations)
- [Pinned memory and non_blocking: pin_memory()](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#pin-memory)
- [Pinned memory and non_blocking: Asynchronous vs. Synchronous Operations with non_blocking=True (CUDA cudaMemcpyAsync)](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync)
- [torch.utils.checkpoint: torch.utils.checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html#torch-utils-checkpoint)
- [PyTorch GitHub issues: MPS device appears much slower than CPU on M1 Mac Pro](https://github.com/pytorch/pytorch/issues/77799)
- [PyTorch GitHub issues: 4x performance regression for 3D convs with AMP on torch 2.9.0](https://github.com/pytorch/pytorch/issues/166122)
- [PyTorch AMP recipe: Inspecting/modifying gradients (e.g., clipping)](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#inspecting-modifying-gradients-e-g-clipping)
- [PyTorch performance tuning guide: Enable cuDNN auto-tuner](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner)
- [NVIDIA convolutional layers: 1. Quick Start Checklist](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#checklist)
- [PyTorch performance tuning guide: Disable bias for convolutions directly followed by a batch norm](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)
- [torch.optim: Algorithms](https://docs.pytorch.org/docs/2.14/optim.html#algorithms)
- [torch.amp: Autocasting](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)
- [PyTorch CUDA semantics: TensorFloat-32 (TF32) on Ampere (and later) devices](https://docs.pytorch.org/docs/2.14/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
- [torch.amp: CUDA Ops that promote to the widest input type](https://docs.pytorch.org/docs/2.14/amp.html#cuda-ops-that-promote-to-the-widest-input-type)
