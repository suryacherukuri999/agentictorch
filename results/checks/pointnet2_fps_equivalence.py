"""Does the Triton farthest point sampling the agent wrote (gpt-6.1-sol, on pointnet2)
pick exactly the points the original loop picks? Run on a Modal L4:

modal run results/checks/pointnet2_fps_equivalence.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "numpy")
)
app = modal.App("hotloop-fpscheck")

CHECK = r"""
import torch, triton, triton.language as tl

@triton.jit
def _fps_kernel(xyz, initial, output, N: tl.constexpr, S: tl.constexpr,
                STRIDE_B: tl.constexpr, STRIDE_N: tl.constexpr,
                STRIDE_C: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < N
    base = xyz + batch * STRIDE_B
    x = tl.load(base + offsets * STRIDE_N, valid, other=0)
    y = tl.load(base + offsets * STRIDE_N + STRIDE_C, valid, other=0)
    z = tl.load(base + offsets * STRIDE_N + 2 * STRIDE_C, valid, other=0)
    distance = tl.full((BLOCK,), 1.0e10, tl.float32)
    farthest = tl.load(initial + batch)
    for i in range(S):
        tl.store(output + batch * S + i, farthest)
        cx = tl.load(base + farthest * STRIDE_N)
        cy = tl.load(base + farthest * STRIDE_N + STRIDE_C)
        cz = tl.load(base + farthest * STRIDE_N + 2 * STRIDE_C)
        dx = x - cx
        dy = y - cy
        dz = z - cz
        dist = (dx * dx + dy * dy) + dz * dz
        distance = tl.where(dist < distance, dist, distance)
        farthest = tl.argmax(tl.where(valid, distance, -float('inf')), 0,
                             tie_break_left=True).to(tl.int64)

def original(xyz, npoint, farthest):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

torch.manual_seed(0)
total = same = 0
for trial in range(20):
    for (N, S) in ((1024, 512), (512, 128)):
        points = torch.randn(24, 3, N, device="cuda")
        if trial % 5 == 0:
            points = points.round(decimals=1)  # many exact ties in distance
        xyz = points.permute(0, 2, 1)  # non-contiguous, as in the model
        start = torch.randint(0, N, (24,), device="cuda")
        want = original(xyz, S, start.clone())
        got = torch.empty(24, S, dtype=torch.long, device="cuda")
        block = triton.next_power_of_2(N)
        _fps_kernel[(24,)](xyz, start, got, N, S, *xyz.stride(), block, enable_fp_fusion=False)
        total += 1
        same += bool(torch.equal(want, got))
print(f"identical picks in {same} of {total} runs", torch.cuda.get_device_name())
"""


@app.function(image=image, gpu="L4", timeout=600)
def check() -> str:
    import subprocess

    with open("/tmp/check.py", "w") as f:
        f.write(CHECK)
    out = subprocess.run(["python", "/tmp/check.py"], capture_output=True, text=True)
    return out.stdout + out.stderr[-1500:]


@app.local_entrypoint()
def main():
    print(check.remote())
