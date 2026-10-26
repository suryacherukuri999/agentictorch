# AgenticTorch report: pytorch-wgan

**Result: 1.59x faster, loss within noise** on an NVIDIA L4.

- Command: `python main.py --model WGAN-GP --is_train True --download True --dataroot datasets/cifar --dataset cifar --generator_iters 40000 --cuda True --batch_size 64`
- Baseline: 75.3 ms/step over 2 runs
- Model: gpt-6.1-sol, 7 turns, 40125 input and 1905 output tokens

Optimized WGAN-GP with BF16 autocast for ordinary generator/critic forwards and FP32 loss reductions, keeping gradient-penalty double backward in FP32. Removed unused generator autograd work during critic updates while retaining BatchNorm updates; disabled autograd for saved-image generation without changing sampling batch sizes. Enabled fused CUDA Adam, cuDNN benchmarking, set-to-none gradient clearing, persistent data-loader workers, pinned memory, and nonblocking transfers. Reduced saved-image CPU transfers and allocated gradient seeds directly on-device. Left critic compilation disabled because its input-gradient penalty requires double backward. No runs or measurements were performed.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 75.3 ms/step | 47.2 ms/step | kept: 1.59x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/models/wgan_gradient_penalty.py b/models/wgan_gradient_penalty.py
index fee0c5b..a1efaa7 100755
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
 
@@ -126,6 +126,7 @@ class WGAN_GP(object):
         if cuda_flag:
             self.cuda_index = 0
             self.cuda = True
+            torch.backends.cudnn.benchmark = True
             self.D.cuda(self.cuda_index)
             self.G.cuda(self.cuda_index)
             print("Cuda enabled flag: {}".format(self.cuda))
@@ -156,7 +157,7 @@ class WGAN_GP(object):
             Wasserstein_D = 0
             # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
             for d_iter in range(self.critic_iter):
-                self.D.zero_grad()
+                self.D.zero_grad(set_to_none=True)
 
                 images = self.data.__next__()
                 # Check for batch to have full batch_size
@@ -170,20 +171,25 @@ class WGAN_GP(object):
                 # Train discriminator
                 # WGAN - Training discriminator more iterations than generator
                 # Train with real images
-                d_loss_real = self.D(images)
-                d_loss_real = d_loss_real.mean()
+                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.cuda):
+                    d_loss_real = self.D(images)
+                d_loss_real = d_loss_real.float().mean()
                 d_loss_real.backward(mone)
 
                 # Train with fake images
                 z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
 
-                fake_images = self.G(z)
-                d_loss_fake = self.D(fake_images)
-                d_loss_fake = d_loss_fake.mean()
+                # Keep BatchNorm updates, but do not build an unused generator graph.
+                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.cuda):
+                    fake_images = self.G(z)
+                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.cuda):
+                    d_loss_fake = self.D(fake_images)
+                d_loss_fake = d_loss_fake.float().mean()
                 d_loss_fake.backward(one)
 
                 # Train with gradient penalty
-                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
+                # The input-gradient norm and its double backward stay in FP32.
+                gradient_penalty = self.calculate_gradient_penalty(images.detach(), fake_images.detach().float())
                 gradient_penalty.backward()
 
 
@@ -196,13 +202,14 @@ class WGAN_GP(object):
             for p in self.D.parameters():
                 p.requires_grad = False  # to avoid computation
 
-            self.G.zero_grad()
+            self.G.zero_grad(set_to_none=True)
             # train generator
             # compute loss with fake images
             z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
-            fake_images = self.G(z)
-            g_loss = self.D(fake_images)
-            g_loss = g_loss.mean()
+            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.cuda):
+                fake_images = self.G(z)
+                g_loss = self.D(fake_images)
+            g_loss = g_loss.float().mean()
             g_loss.backward(mone)
             g_cost = -g_loss
             self.g_optimizer.step()
@@ -232,9 +239,10 @@ class WGAN_GP(object):
 
                 # Denormalize images and save them in grid 8x8
                 z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
-                samples = self.G(z)
-                samples = samples.mul(0.5).add(0.5)
-                samples = samples.data.cpu()[:64]
+                with torch.no_grad():
+                    samples = self.G(z)
+                    samples = samples.mul(0.5).add(0.5)
+                samples = samples[:64].cpu()
                 grid = utils.make_grid(samples)
                 utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
 
@@ -317,9 +325,7 @@ class WGAN_GP(object):
 
         # calculate gradients of probabilities with respect to examples
         gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
-                               grad_outputs=torch.ones(
-                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
-                                   prob_interpolated.size()),
+                               grad_outputs=torch.ones_like(prob_interpolated),
                                create_graph=True, retain_graph=True)[0]
 
         # flatten the gradients to it calculates norm batchwise
@@ -334,6 +340,7 @@ class WGAN_GP(object):
         else:
             return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])
 
+    @torch.no_grad()
     def generate_img(self, z, number_of_images):
         samples = self.G(z).data.cpu().numpy()[:number_of_images]
         generated_images = []
diff --git a/utils/data_loader.py b/utils/data_loader.py
index cae393b..533e065 100755
--- a/utils/data_loader.py
+++ b/utils/data_loader.py
@@ -1,3 +1,4 @@
+import os
 import torchvision.datasets as dset
 import torchvision.transforms as transforms
 import torch.utils.data as data_utils
@@ -46,7 +47,10 @@ def get_data_loader(args):
     assert train_dataset
     assert test_dataset
 
-    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
+    num_workers = min(4, os.cpu_count() or 1)
+    train_dataloader = data_utils.DataLoader(
+        train_dataset, batch_size=args.batch_size, shuffle=True,
+        num_workers=num_workers, pin_memory=bool(args.cuda), persistent_workers=True)
     test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)
 
     return train_dataloader, test_dataloader
```
