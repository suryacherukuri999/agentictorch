# AgenticTorch

An agent that finds and fixes slow PyTorch training code in real open-source repos.
It proves every fix by rerunning the training on a GPU: a change is kept only if
it is faster beyond run-to-run noise and the loss stays the same.

## Results

Every number below comes from a run saved in `results/`, and
`python -m agentictorch.scoreboard` rebuilds these tables. Each one compares a repo's
original code with the changed code in pairs: both run in the same container on the
same GPU, once per seed (two seeds), for 50 measured steps after 10 warm-up steps.

### The agent on 15 real repos

The agent ran with gpt-6-luna, a small OpenAI model; every gpt-6-luna call in the
project, all setups included, cost $0.08 of API credit. For each repo where it
kept a change, the original and the changed code were measured again on an A100:

| Repo | Stars | GPU | Before (ms/step) | After (ms/step) | Speedup | Loss | Fixes | Pull request |
|---|---|---|---|---|---|---|---|---|
| [monologg/JointBERT](https://github.com/monologg/JointBERT) | 749 | A100-SXM4-40GB | 67.4 | 17.3 | 3.90x | within noise | bf16 autocast, fused AdamW, torch.compile with CUDA graphs | not sent: no outside pull request merged in 12 months |
| [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) | 4,660 | A100-SXM4-40GB | 13.7 | 4.0 | 3.42x | within noise | DataLoader workers with pinned memory, torch.compile | not sent: no outside pull request merged in 12 months |
| [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | 4,967 | A100-SXM4-40GB | 235.4 | 108.3 | 2.17x | within noise | no sync per point in farthest point sampling | not sent: no outside pull request merged in 12 months |
| [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) | 6,058 | A100-SXM4-80GB | 80.8 | 39.4 | 2.05x | within noise | TF32 matmuls | not sent: no outside pull request merged in 12 months |
| [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) | 17,457 | A100-SXM4-40GB | 24.5 | 12.0 | 2.04x | within noise | DataLoader workers with pinned memory | not sent: no outside pull request merged in 12 months |
| [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) | 7,436 | A100-SXM4-40GB | 265.4 | 146.4 | 1.81x | within noise | bf16 autocast, persistent DataLoader workers, fused Adam | not sent: no outside pull request merged in 12 months |
| [Zeleni9/pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan) | 809 | A100-SXM4-40GB | 39.5 | 28.7 | 1.38x | within noise | DataLoader workers with pinned memory | not sent: no outside pull request merged in 12 months |
| [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) | 25,553 | A100-SXM4-80GB | 19.8 | 15.8 | 1.22x | within noise | DataLoader workers with pinned memory, fused Adam | drafted, not sent |
| [w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) | 646 | A100-SXM4-40GB | 144.7 | 118.3 | 1.22x | within noise | bf16 autocast for the forward pass and loss | not sent: no outside pull request merged in 12 months |
| [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | 489 | A100 80GB PCIe | 245.9 | 202.3 | 1.22x | within noise | DataLoader workers with pinned memory | issue drafted, not sent |
| [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR) | 825 | A100-SXM4-40GB | 178.1 | 175.8 | 1.01x (1.01x is within noise) | within noise | bf16 autocast, channels_last | not sent: no outside pull request merged in 12 months |
| [facebookresearch/detr](https://github.com/facebookresearch/detr) | 15,352 | L4 | 205.0 |  | nothing kept |  |  | not sent: no outside pull request merged in 12 months |
| [giddyyupp/ganilla](https://github.com/giddyyupp/ganilla) | 482 | L4 | 130.5 |  | nothing kept |  |  | not sent: no outside pull request merged in 12 months |
| [lucidrains/transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) | 1,408 | L4 | 85.3 |  | nothing kept |  |  | not sent |
| [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers) | 5,961 | L4 | 717.8 |  | nothing kept |  |  | drafted from the gpt-6.1-sol run, not sent |

- **GPU** is the card Modal assigned to an A100-40GB request; every request was
  billed as an A100-40GB. Both sides of a pair always ran on the same card.
- **SimCLR:** the same changes were 2.06x faster on an L4. On the A100 the GPU work
  shrank until image augmentation on 8 CPU cores set the pace (data wait rose from
  4% to 36% of each step). PyTorch-YOLOv3 shows the same effect more mildly: 2.07x
  on an L4, 1.81x on the A100, with 39% of each A100 step waiting on data.
- **Nothing kept** means every change the agent tried crashed, was slower, or was
  within noise, so the original code stands.
- **Pull requests:** none sent yet. 11 of the 15 repos haven't merged a pull request
  from outside their team in the last 12 months.

### Against the reference model working directly

The same repos fixed by the reference model at max effort, editing each repo itself: it
read the code and the profiles and measured its attempts with the same paired
benchmark (up to three per repo, each bundling several changes), but outside the
agent loop. On an L4, as a reference for how much the agent with gpt-6-luna leaves on
the table:

| Repo | GPU | Original (ms/step) | Agent | Agent's changes | Reference model, direct | Its changes |
|---|---|---|---|---|---|---|
| efficient-kan | L4 | 17.3 | 3.40x | DataLoader workers with pinned memory, torch.compile | 4.30x | loader workers with pinned memory, no per-step syncs, CUDA graphs through torch.compile, fused AdamW |
| x-transformers | L4 | 707.7 | nothing kept |  | 4.13x | batches built on the CPU and copied once, flash attention, bf16 |
| rectified-flow-pytorch | L4 | 439.4 | 1.17x | DataLoader workers with pinned memory | 3.12x | bf16, torch.compile through accelerate, loader workers |
| pointnet2 | L4 | 274.0 | 1.65x | no sync per point in farthest point sampling | 2.61x | no syncs in farthest point sampling and grouping, sampling replayed from a CUDA graph, topk instead of a full sort, accuracy kept on the GPU |
| simclr | L4 | 416.6 | 2.06x | bf16 autocast, channels_last | 2.44x | bf16 autocast, channels_last, cudnn autotuning, pinned copies, no per-step epoch-loss sync |
| pytorch-ddpm | L4 | 549.2 | 1.29x | bf16 autocast for the forward pass and loss | 2.28x | bf16 autocast, compiled trainer, fused EMA update, one loss sync, pinned copies |
| pytorch-yolov3 | L4 | 529.2 | 2.07x | bf16 autocast, persistent DataLoader workers, fused Adam | 2.28x | bf16 forward with a float32 loss, channels_last convolutions, persistent workers, fewer syncs in the loss |
| jointbert | L4 | 137.9 | 2.65x | bf16 autocast, fused AdamW, torch.compile with CUDA graphs | 2.20x | bf16 autocast, fused AdamW, no per-step loss sync |
| pytorch-wgan | L4 | 77.9 | 1.18x | DataLoader workers with pinned memory | 2.17x | loader workers with pinned memory, noise sampled on the GPU, no generator gradients in critic steps, one sync per printed line, fused Adam, cudnn autotuning, bf16 networks with float32 outputs |
| pytorch-gan-dcgan | L4 | 28.6 | 2.03x | DataLoader workers with pinned memory | 2.10x | loader workers with pinned memory, noise sampled on the GPU, one sync for the printed losses, fused Adam, cudnn autotuning |
| vit-pytorch | L4 | 26.7 | 1.37x | DataLoader workers with pinned memory, fused Adam | 1.74x | loader workers, torch.compile, one sync for the printed losses |
| transfusion-pytorch | L4 | 92.5 | nothing kept |  | 1.55x | loader workers, compiled transformer forward |
| nerf-pytorch | L4 | 191.7 | 1.18x | TF32 matmuls | 1.30x | TF32, no per-step host copies or syncs, ray sampling on the GPU |
| ganilla | L4 | 123.2 | nothing kept |  | 1.13x | compiled networks, cudnn autotuning, pinned non-blocking input copies |
| detr | L4 | 195.4 | nothing kept |  | 1.06x | one sync for logging, box checks on the GPU, matcher indices copied once, pinned copies, 8 workers, bf16 forward |

With the standard fixes (mixed precision, DataLoader workers, fused optimizers),
the agent came within 10% of the direct fix or beat it on JointBERT, PyTorch-GAN,
PyTorch-YOLOv3 and nerf-pytorch. It fell far short where the win needed a bigger or
riskier change, such as flash attention behind a library flag (x-transformers),
compiling library code (rectified-flow, pytorch-ddpm, transfusion), or bf16 in a GAN
whose gradient penalty takes a second backward pass (pytorch-wgan).

### Against simpler setups

On the five GPU-bound repos, all with gpt-6-luna on an L4. **rules** applies a few
AST rules (DataLoader workers, explicit syncs in the loop); **llm** has the model
read the code and write one patch without running anything; **llm-kb** does the same
with the documentation search; **agent** is the full loop. A blind patch ships
whatever it does, so a crash or a slowdown counts as harmful.

| Target | rules | llm | llm-kb | agent |
|---|---|---|---|---|
| ganilla | 1.00x | 0.95x harmful | 1.46x | 1.00x |
| jointbert | 1.00x | 1.66x | 1.70x | 2.65x |
| nerf-pytorch | 1.00x | 2.49x | 2.55x | 1.18x |
| pytorch-ddpm | 1.00x | 1.37x | 1.00x harmful | 1.29x |
| simclr | 1.00x | 1.62x | 1.11x | 2.06x |

| Setup | Targets | Speedup (geometric mean) | Harmful changes shipped | Harmful attempts caught |
|---|---|---|---|---|
| rules | 5 | 1.00x | 0 |  |
| llm | 5 | 1.54x | 1 |  |
| llm-kb | 5 | 1.48x | 1 |  |
| agent | 5 | 1.53x | 0 | 3 |

Measuring didn't raise the typical speedup here; it removed the harm. The blind
patches were bolder: on nerf-pytorch, compiling the model and running the whole
render in bf16 gave about 2.5x, where the agent kept only TF32. But blind patching
also shipped a slowdown on ganilla and a crash on pytorch-ddpm. The agent shipped
neither, and its keep rule caught three harmful attempts of its own.

### The model, the loop and a direct fix

Four ways to fix the same 15 repos, each measured against the original code on an L4;
1.00x means nothing was kept:
- **Agent, gpt-6-luna** and **Agent, gpt-6.1-sol:** the full loop, with the small
  model and with a stronger one at 20 times the price per token.
- **One patch, gpt-6.1-sol:** the `llm` setup. The stronger model reads the code and
  writes one patch without running anything, and the patch ships whatever it does.
- **Reference model, direct:** the reference fixes from the section above.

| Repo | Agent, gpt-6-luna | Agent, gpt-6.1-sol | One patch, gpt-6.1-sol | Reference model, direct |
|---|---|---|---|---|
| detr | 1.00x | 1.00x | 1.00x harmful | 1.06x |
| efficient-kan | 3.40x | 3.58x | 2.17x | 4.30x |
| ganilla | 1.00x | 1.34x | 1.54x | 1.13x |
| jointbert | 2.65x | 2.72x | 3.05x | 2.20x |
| nerf-pytorch | 1.18x | 2.43x | 2.71x | 1.30x |
| pointnet2 | 1.65x | 4.89x | 1.93x | 2.61x |
| pytorch-ddpm | 1.29x | 2.54x | 2.52x | 2.28x |
| pytorch-gan-dcgan | 2.03x | 2.15x | 1.93x | 2.10x |
| pytorch-wgan | 1.18x | 1.41x | 1.59x | 2.17x |
| pytorch-yolov3 | 2.07x | 2.10x | 0.60x harmful | 2.28x |
| rectified-flow-pytorch | 1.17x | 3.13x | 3.17x | 3.12x |
| simclr | 2.06x | 2.39x | 2.50x | 2.44x |
| transfusion-pytorch | 1.00x | 1.58x | 0.86x harmful | 1.55x |
| vit-pytorch | 1.37x | 2.43x | 1.80x | 1.74x |
| x-transformers | 1.00x | 5.52x | 5.61x | 4.13x |
| **Geometric mean** | **1.48x** | **2.36x** | **1.92x** | **2.12x** |
| Harmful changes shipped | 0 | 0 | 3 | 0 |

**The loop beat both direct approaches.** With the same model, the agent's geometric
mean was 2.36x against 1.92x for one blind patch, and it shipped no harm where the
blind patch shipped three: PyTorch-YOLOv3 at 0.60x (cuDNN autotuning and
torch.compile, on batch shapes that keep changing), transfusion-pytorch at 0.86x
(bf16 on a transformer too small to gain from it), and DETR, whose compiled patch
timed out. Against the reference model working directly, the agent with gpt-6.1-sol was
ahead by more than 3% on 7 repos, level on 4 and behind on 4 (2.36x against 2.12x),
though that comparison mixes the method with the model: with gpt-6-luna the agent
was behind (1.48x).

**Measuring one change at a time leaves gains behind.** The blind patch was ahead of
the agent on 5 repos, level on 4 and behind on 6. It bundles every change the model
is confident about, while the agent keeps a change only if it is at least 3% faster
in every pair on its own, so small gains that only add up together get dropped. On
nerf-pytorch the agent tried fused Adam and removing the debug-check syncs, measured
each within noise (1.00x to 1.04x) and reverted them; the blind patch kept them with
the rest and measured 2.71x against the agent's 2.43x.

**The model sets the ceiling.** gpt-6.1-sol was faster than gpt-6-luna on 14 of the
15 repos, by more than 10% on 10 of them, and tied on DETR. Most of the gap is
torch.compile: gpt-6.1-sol got it working on 9 repos, gpt-6-luna on 2, and gpt-6-luna
never tried it on five others. gpt-6.1-sol also kept bf16 on 8 repos to gpt-6-luna's
4, used 62 of 65 benchmark rounds in its runs with reports where gpt-6-luna often
stopped early (53 of 75), and made three changes gpt-6-luna never tried: switching
on x-transformers' existing flash attention option, replacing vit-pytorch's
hand-written attention with PyTorch's fused `scaled_dot_product_attention`, and a
Triton kernel for PointNet++.

#### What gpt-6.1-sol and the reference model each changed

The agent with gpt-6.1-sol against the reference model's direct fix, with gpt-6-luna for
reference. A number in brackets is that change's own measured gain in its round.

Where the agent with gpt-6.1-sol was faster:

| Repo | gpt-6-luna | gpt-6.1-sol | Reference model | Only gpt-6.1-sol did | Why it won |
|---|---|---|---|---|---|
| x-transformers | 1.00x | 5.52x | 4.13x | torch.compile (1.37x), fused Adam (1.03x) | Both had flash attention and bf16; compiling fused the rest |
| pointnet2 | 1.65x | 4.89x | 2.61x | A Triton kernel for the sampling loop (1.95x), bf16 convolutions (1.66x) | One kernel launch instead of about 4,000 small ones; the direct fix's CUDA graph only removed the CPU's launch cost |
| vit-pytorch | 1.37x | 2.43x | 1.74x | PyTorch's fused attention in place of the hand-written one (1.27x), compiling with CUDA graphs (1.51x) | The direct fix compiled without CUDA graphs and kept the slow attention |
| nerf-pytorch | 1.18x | 2.43x | 1.30x | bf16 for the MLP only, rendering kept in float32 (2.06x), torch.compile (1.23x) | The MLP is most of each step; the direct fix used TF32 |
| pytorch-ddpm | 1.29x | 2.54x | 2.28x | channels_last (1.13x) | Both had bf16 and torch.compile; channels_last speeds up bf16 convolutions |
| jointbert | 2.65x | 2.72x | 2.20x | Nothing gpt-6-luna lacked | The direct fix skipped torch.compile (1.17x for gpt-6.1-sol) |
| ganilla | 1.00x | 1.34x | 1.13x | Nothing new: the same compile and pinned copies | The direct fix also turned on cuDNN autotuning and compiled each network's forward instead of the whole module; which detail mattered is untested |

Where the reference model's direct fix was faster:

| Repo | gpt-6-luna | gpt-6.1-sol | Reference model | Only the reference model did | Why it won |
|---|---|---|---|---|---|
| pytorch-wgan | 1.18x | 1.41x | 2.17x | bf16 for both networks, outputs cast back to float32 | 1.38x without it; neither agent run tried bf16 |
| efficient-kan | 3.40x | 3.58x | 4.30x | Removed the progress bar's two syncs per step | Likely because at 4 ms per step each wait shows; untested on its own |
| pytorch-yolov3 | 2.07x | 2.10x | 2.28x | Fewer syncs in the loss code | Small gap; that change alone measured 1.01x |
| detr | 1.00x | 1.00x | 1.06x | Six small changes (bf16, pinned copies, fewer syncs and others) | Nothing helps much at batch size 2 |

About equal (within 3%): rectified-flow-pytorch (3.13x against 3.12x),
transfusion-pytorch (1.58x against 1.55x), SimCLR (2.39x against 2.44x) and
PyTorch-GAN (2.15x against 2.10x), with the same main changes on both sides.

PointNet++ was the hardest. Its farthest point sampling is a Python loop of 640
steps per forward pass, and a masked update inside it (`distance[mask] = dist[mask]`)
made the CPU wait for the GPU 1,280 times per step; the GPU was busy 11% of the
time. Both models found that line in their first round, since the profiler names
it. Only gpt-6.1-sol got past the thousands of tiny kernel launches the loop still
made: it wrote a Triton kernel that runs the whole sampling loop at once, then added
bf16 for the convolutions. The kernel picks exactly the points the original loop
picks (identical in 40 of 40 test runs on a GPU, including inputs with many ties).
The reference model's direct fix replayed the loop from a CUDA graph instead and reached
2.61x; one blind patch from gpt-6.1-sol reached 1.93x.

DETR resisted everyone. At batch size 2, Python overhead per op sets the pace;
neither agent run kept a change, the blind patch's dynamic-shape compile timed out,
and the best of three direct attempts measured 1.06x.

- **Lost reports:** on pytorch-ddpm and rectified-flow-pytorch, the monthly GPU
  budget guard stopped the gpt-6.1-sol agent run's final measurement before it wrote
  a report. Their numbers come from measuring the kept changes against the original
  again, in pairs (`results/final-pairs/`).
- **Late direct fixes:** for efficient-kan, PyTorch-GAN and pytorch-wgan the direct
  fix was at first only the one-line DataLoader fix from the go/no-go check. The full
  direct fixes in the table came after the agent runs, with their reports at hand;
  the pytorch-wgan one borrows gpt-6.1-sol's idea of skipping the generator's
  gradients in critic steps.
- **Cost:** an agent run with gpt-6.1-sol cost $0.05 to $0.09 of API credit and a
  one-patch run $0.026 to $0.15; every gpt-6-luna call in the project cost $0.08 in
  total. All are priced from the token usage the API reported, cache discount
  included (`results/checks/llm_spend.txt`). The first six gpt-6.1-sol agent runs
  saved no cached-token counts, so the four of them with reports price every prompt
  token at the full rate ($0.44 to $0.76 each).

### Planted test suite

Small scripts of our own (ResNet-18 on CIFAR-10, a character-level GPT, DistilBERT),
each with one planted mistake; the planted cost is how much faster the clean script
runs. 6 of the 14 mistakes cost nothing measurable on an L4, in speed or memory.
With a local 9B model (ornith-1.5 through Ollama), the agent won back all of a 2.48x
DataLoader slowdown (2.56x) and 92% of a 2.23x fp32 slowdown (2.03x). The local
model was too slow for the real repos: Ollama serves it one request at a time, so
three agents queued 4 to 12 minutes for each turn.

## How it works

```
  Mac                                         Modal (one container image per repo)
  ---------------------------------------     ------------------------------------
  agent loop: an LLM calling tools
    read_file, search_code, apply_edit        the repo's own training command,
    search_docs (local index, cited URLs)       run for N steps with a small
    profile, quick_check, run_benchmark  -->    module injected at startup
                                         <--  step times, peak memory, losses,
  keep/revert rule (code, not the model)        profiler summary
  report: changes, measured effect, diff
```

1. **Baseline.** Run the repo's training command for a few dozen steps, once per
   seed, on a cloud GPU. Record step time, peak memory and loss.
2. **Profile.** A separate short run under the PyTorch profiler, summarized as text:
   GPU busy share, data-loading wait, host-device syncs per step, and the top ops
   by GPU time, each with the repo line that caused it.
3. **Diagnose.** The model reads the summary and the code, and searches a local
   index of PyTorch and NVIDIA performance docs.
4. **Try a fix.** It edits a git working copy of the repo, never the original. A
   five-step quick check on the GPU catches crashes before a full benchmark.
5. **Keep or revert.** The best version so far and the edited one run as a pair.
   Code decides whether the edit stays.
6. **Report.** After up to five benchmark rounds, one more pair measures the
   original against everything kept, and a markdown report lists each attempt,
   its measured effect, the diff and the sources read.

### Measuring someone else's code without editing it

The harness is a `sitecustomize.py` put on `PYTHONPATH`, so it loads before the
repo's code and the repo stays unchanged. It:
- uses a global optimizer step hook as the step clock (GANs that step two
  optimizers per iteration count once);
- records the loss without a per-step sync by wrapping `Tensor.backward` and
  `GradScaler.scale`, and reads the tensors at the end;
- syncs CUDA only at the edges of the measured window, after the warm-up steps;
- sets seeds so the data order repeats, and stops the run cleanly after N steps.

### Paired runs and the keep rule

Containers land on different CPUs and GPUs. For CPU-bound training that alone
moved step time by 40% between two runs of the same code. So the baseline and the
candidate run in the same container, one pair per seed, with the order
alternating, and the speedup is the geometric mean of the per-pair ratios.

A change is kept only if:
- it is at least 3% faster in every pair (or a run that ran out of memory now fits), and
- its final loss passes a two-sample t-test at 95% against the run-to-run spread
  of the loss, with a 2% floor for repos whose loss barely varies.

Crashes, NaN losses, slowdowns and changes within noise are reverted.

### Profiles that point at lines

In the profile run only, a `TorchFunctionMode` wraps each torch call made from
repo code in a profiler range named after its source line, so every op and kernel
is attributed to a `file:line`. Backward ops are mapped to the forward line that
created them through autograd sequence numbers. CUDA's sync debug mode attributes
each host-device sync to the repo line that triggered it, including implicit ones
such as formatting a CUDA tensor into a progress bar.

### Knowledge base

The PyTorch performance tuning guide, the AMP, DataLoader, CUDA semantics,
checkpointing and `torch.compile` docs, NVIDIA's deep learning performance guides,
and resolved PyTorch GitHub issues about performance, split into sections with
their URLs. Search is BM25 plus `bge-small-en-v1.5` embeddings, fused with
reciprocal rank fusion and reranked with `bge-reranker-base`, all local. On a
15-query check (symptom to the section that explains the fix), 14 find it.

## How to run

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), a [Modal](https://modal.com)
account for the GPU runs, and a model: a local one through
[Ollama](https://ollama.com), or an API key for Gemini or OpenAI in the
environment.

```
uv sync
python -m modal setup                         # once
python -m agentictorch.kb.fetch               # download the docs
python -m agentictorch.kb.search build        # build the search index

agentictorch download pytorch-ddpm            # dataset into a Modal volume, once
agentictorch measure pytorch-ddpm             # baseline on an L4, two seeds
agentictorch profile pytorch-ddpm             # profiler summary
agentictorch run pytorch-ddpm --setup agent --model gpt-6-luna
```

Models are listed in `agentictorch/agent/models.py`; keys are read from the environment
or `~/agentictorch-work/.env`.

Each target repo is a small config in `targets/<name>/target.toml`: the repo, a
pinned commit, the training command, extra pip packages and the dataset. `--setup`
picks what runs: `rules` (a few AST rules), `llm` (the model reads the code and
patches it once without measuring), `llm-kb` (the same with the docs), or `agent`.
`python -m agentictorch.scoreboard` compares the setups over everything in `results/`.

`python -m agentictorch.survey` rebuilds the static scan of popular PyTorch repos that
motivated the project (`results/survey/`).

## Limitations

- Runs are short: a few dozen measured steps. The loss check covers that window
  only, so a change that hurts convergence later would pass. Pull requests put
  precision changes behind an opt-in flag for that reason.
- Speedups depend on the GPU. A result on an L4 says nothing about another GPU,
  and results are never compared across GPUs.
- Some static warning signs cost nothing in practice. Of 14 planted mistakes in
  the test suite, 6 had no measurable cost on an L4, in speed or memory, because
  they were off the bottleneck (for example, `loss.item()` every step in a loop
  that waits on the GPU anyway).
- With two seeds, the loss check is only as tight as the loss is repeatable. On
  repos whose loss varies a lot between seeds, it can only catch large changes.
  nerf-pytorch's loss varies about 2x between seeds, so its bolder blind patches
  (the whole render in bf16) passed a check that can't say much.
- The agent tries one change per round and keeps it only if it is at least 3%
  faster in every pair, so small gains that only add up together get dropped: one
  blind patch from the same model beat it that way on five repos. With gpt-6-luna it
  also rarely attempted the bigger changes, such as switching on flash attention or
  compiling a library's code; gpt-6.1-sol did.
- Python only. The tools search and edit the repo's Python code, and the profiler
  maps time to Python lines. A compiled C++/CUDA extension shows up only as GPU
  kernels under the Python line that calls it, and there's no step to rebuild one
  after an edit. None of the 15 repos had one in its training path.

## What didn't work

- **Unpaired runs.** Measuring the baseline and the candidate in separate
  containers gave up to 40% noise on CPU-bound repos. Pairing fixed it.
- **Comparing each run with the baseline run of the same seed** for the loss
  check. On chaotic training (cuDNN autotuning, a high learning rate), two runs
  with the same seed and code ended 0.1 apart in loss. A pooled t-test replaced it.
- **The profiler's `with_stack=True`** for line numbers. On PyTorch 2.14 it named
  frames by their function's `def` line, which is worse than no line for an agent
  deciding where to edit.
- **Trusting the data-wait share.** On x-transformers, 61% of each step showed as
  waiting on the DataLoader. The dataset copied each sample to the GPU, which made
  the CPU wait for the GPU inside the loader. Copying whole batches instead cut the
  wait to 0% and left the step time unchanged (1.01x). The profile's GPU-busy
  share (88%) was the real signal.
- **Trusting the loss check with float16.** The agent kept float16 autocast without
  a GradScaler on two repos. The loss matched over the measured steps, but without
  loss scaling small gradients can underflow over a long run. The agent now refuses
  float16 unless the repo uses a GradScaler.

## Credits

PyTorch's and NVIDIA's performance documentation, which the knowledge base
indexes and the reports cite. BAAI's bge embedding and reranker models through
sentence-transformers. The authors of the repos used as targets.

MIT license.
