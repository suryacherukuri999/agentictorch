# AgenticTorch report: nerf-pytorch

**Result: 2.71x faster, loss within noise** on an NVIDIA L4.

- Command: `python run_nerf.py --config configs/lego.txt`
- Baseline: 184.8 ms/step over 2 runs
- Model: gpt-6.1-sol, 10 turns, 132402 input and 3161 output tokens

Added CUDA bfloat16 autocast for the MLPs while retaining float32 encoding, rendering, and loss; compiled model forwards and positional encoders without changing checkpoint keys; enabled fused CUDA Adam and set-to-none gradient clearing. Removed unconditional GPU synchronization from disabled debug checks, encoded view directions once per ray, detached already nondifferentiable PDF sampling inputs, kept images on GPU, cached coordinate pools, and generated only sampled training rays. Model architecture, sampling order, batch size, learning-rate schedule, step count, and logging/checkpoint structure are unchanged. No execution or benchmarking was available.

## Changes

| # | Change | Before | After | Outcome |
|---|---|---|---|---|
| 1 | llm patch, made without measuring | 184.8 ms/step | 68.0 ms/step | kept: 2.71x faster, loss within noise |

## Diff of the kept changes

```diff
diff --git a/run_nerf.py b/run_nerf.py
index bc270be..11c08d7 100644
--- a/run_nerf.py
+++ b/run_nerf.py
@@ -41,12 +41,17 @@ def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
     embedded = embed_fn(inputs_flat)
 
     if viewdirs is not None:
-        input_dirs = viewdirs[:,None].expand(inputs.shape)
-        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
-        embedded_dirs = embeddirs_fn(input_dirs_flat)
+        # Each sample on a ray has the same direction; encode it only once.
+        encoded_dirs = embeddirs_fn(viewdirs)
+        embedded_dirs = encoded_dirs[:, None, :].expand(
+            *inputs.shape[:-1], encoded_dirs.shape[-1]).reshape(-1, encoded_dirs.shape[-1])
         embedded = torch.cat([embedded, embedded_dirs], -1)
 
-    outputs_flat = batchify(fn, netchunk)(embedded)
+    # Keep positional encoding and volume integration in float32.
+    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
+                        enabled=device.type == 'cuda'):
+        outputs_flat = batchify(fn, netchunk)(embedded)
+    outputs_flat = outputs_flat.float()
     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
     return outputs
 
@@ -203,8 +208,9 @@ def create_nerf(args):
                                                                 embeddirs_fn=embeddirs_fn,
                                                                 netchunk=args.netchunk)
 
-    # Create optimizer
-    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
+    # Fuse the per-parameter Adam updates on CUDA.
+    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999),
+                                 fused=device.type == 'cuda')
 
     start = 0
     basedir = args.basedir
@@ -234,6 +240,15 @@ def create_nerf(args):
 
     ##########################
 
+    if device.type == 'cuda':
+        # Compile the callables, not the modules, to preserve checkpoint keys.
+        model.forward = torch.compile(model.forward)
+        if model_fine is not None:
+            model_fine.forward = torch.compile(model_fine.forward)
+        embed_fn = torch.compile(embed_fn)
+        if embeddirs_fn is not None:
+            embeddirs_fn = torch.compile(embeddirs_fn)
+
     render_kwargs_train = {
         'network_query_fn' : network_query_fn,
         'perturb' : args.perturb,
@@ -390,7 +405,7 @@ def render_rays(ray_batch,
         rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
 
         z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
-        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
+        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1].detach(), N_importance, det=(perturb==0.), pytest=pytest)
         z_samples = z_samples.detach()
 
         z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
@@ -411,9 +426,10 @@ def render_rays(ray_batch,
         ret['acc0'] = acc_map_0
         ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
 
-    for k in ret:
-        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
-            print(f"! [Numerical Error] {k} contains nan or inf.")
+    if DEBUG:
+        for k in ret:
+            if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
+                print(f"! [Numerical Error] {k} contains nan or inf.")
 
     return ret
 
@@ -691,8 +707,7 @@ def train():
         i_batch = 0
 
     # Move training data to GPU
-    if use_batching:
-        images = torch.Tensor(images).to(device)
+    images = torch.Tensor(images).to(device)
     poses = torch.Tensor(poses).to(device)
     if use_batching:
         rays_rgb = torch.Tensor(rays_rgb).to(device)
@@ -707,6 +722,19 @@ def train():
     # Summary writers
     # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
     
+    # Pixel pools are fixed throughout training. Keep their original row-major
+    # order so NumPy's sampling (and its random-number stream) is unchanged.
+    if not use_batching and N_rand is not None:
+        full_coords = torch.stack(torch.meshgrid(
+            torch.arange(H, device=device), torch.arange(W, device=device),
+            indexing='ij'), -1).reshape(-1, 2)
+        dH = int(H//2 * args.precrop_frac)
+        dW = int(W//2 * args.precrop_frac)
+        crop_coords = torch.stack(torch.meshgrid(
+            torch.arange(H//2 - dH, H//2 + dH, device=device),
+            torch.arange(W//2 - dW, W//2 + dW, device=device),
+            indexing='ij'), -1).reshape(-1, 2)
+
     start = start + 1
     for i in trange(start, N_iters):
         time0 = time.time()
@@ -729,39 +757,28 @@ def train():
             # Random from one image
             img_i = np.random.choice(i_train)
             target = images[img_i]
-            target = torch.Tensor(target).to(device)
             pose = poses[img_i, :3,:4]
 
             if N_rand is not None:
-                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
-
                 if i < args.precrop_iters:
-                    dH = int(H//2 * args.precrop_frac)
-                    dW = int(W//2 * args.precrop_frac)
-                    coords = torch.stack(
-                        torch.meshgrid(
-                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
-                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
-                        ), -1)
+                    coords = crop_coords
                     if i == start:
-                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
+                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                 else:
-                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
+                    coords = full_coords
 
-                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                 select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
-                select_coords = coords[select_inds].long()  # (N_rand, 2)
-                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
-                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
+                select_coords = coords[select_inds]  # (N_rand, 2)
+                rays_o, rays_d = get_rays(H, W, K, pose, coords=select_coords)
                 batch_rays = torch.stack([rays_o, rays_d], 0)
                 target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
 
         #####  Core optimization loop  #####
+        optimizer.zero_grad(set_to_none=True)
         rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                 verbose=i < 10, retraw=True,
                                                 **render_kwargs_train)
 
-        optimizer.zero_grad()
         img_loss = img2mse(rgb, target_s)
         trans = extras['raw'][...,-1]
         loss = img_loss
diff --git a/run_nerf_helpers.py b/run_nerf_helpers.py
index bc6ee77..01e6c38 100644
--- a/run_nerf_helpers.py
+++ b/run_nerf_helpers.py
@@ -150,10 +150,15 @@ class NeRF(nn.Module):
 
 
 # Ray helpers
-def get_rays(H, W, K, c2w):
-    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
-    i = i.t()
-    j = j.t()
+def get_rays(H, W, K, c2w, coords=None):
+    if coords is None:
+        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
+        i = i.t()
+        j = j.t()
+    else:
+        # Training only needs the sampled pixels, not the full image's rays.
+        i = coords[:, 1].to(dtype=c2w.dtype)
+        j = coords[:, 0].to(dtype=c2w.dtype)
     dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
     # Rotate ray directions from camera frame to the world frame
     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
```
