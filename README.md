# Hotloop

An agent that finds and fixes slow PyTorch training code in real open-source repos.
It proves every fix by rerunning the training on a GPU: a change is kept only if
it is faster beyond run-to-run noise and the loss stays the same.

## Results

Every number below comes from a run saved in `results/`, and
`python -m hotloop.scoreboard` rebuilds these tables. Each one compares a repo's
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

### Against a hand fix

The same repos with the usual fixes applied by hand and measured the same way, on an
L4, as a reference for how much the agent with gpt-6-luna leaves on the table:

| Repo | GPU | Original (ms/step) | Agent | Agent's changes | Hand fix | Hand fix's changes |
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
the agent came within 10% of the hand fix or beat it on JointBERT, PyTorch-GAN,
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

### Does the model matter?

Yes. The same agent with gpt-6.1-sol, a stronger model at 20 times the price per
token, on all 15 repos, next to gpt-6-luna and the hand fix. All on an L4, each
against the original code, loss within noise; 1.00x means nothing was kept:

| Repo | gpt-6-luna | gpt-6.1-sol | Hand fix |
|---|---|---|---|
| detr | 1.00x | 1.00x | 1.06x |
| efficient-kan | 3.40x | 3.58x | 4.30x |
| ganilla | 1.00x | 1.34x | 1.13x |
| jointbert | 2.65x | 2.72x | 2.20x |
| nerf-pytorch | 1.18x | 2.43x | 1.30x |
| pointnet2 | 1.65x | 4.89x | 2.61x |
| pytorch-ddpm | 1.29x | 2.54x | 2.28x |
| pytorch-gan-dcgan | 2.03x | 2.15x | 2.10x |
| pytorch-wgan | 1.18x | 1.41x | 2.17x |
| pytorch-yolov3 | 2.07x | 2.10x | 2.28x |
| rectified-flow-pytorch | 1.17x | 3.13x | 3.12x |
| simclr | 2.06x | 2.39x | 2.44x |
| transfusion-pytorch | 1.00x | 1.58x | 1.55x |
| vit-pytorch | 1.37x | 2.43x | 1.74x |
| x-transformers | 1.00x | 5.52x | 4.13x |
| **Geometric mean** | **1.48x** | **2.36x** | **2.12x** |

gpt-6.1-sol was faster than gpt-6-luna on 14 of the 15 repos, by more than 10% on 10
of them, and tied on DETR. Most of the gap is torch.compile: gpt-6.1-sol got it
working on 9 repos, gpt-6-luna on 2. gpt-6.1-sol also kept bf16 on 8 repos to
gpt-6-luna's 4, and made three changes gpt-6-luna never tried: switching on
x-transformers' existing flash attention option, replacing vit-pytorch's hand-written
attention with PyTorch's fused `scaled_dot_product_attention`, and a Triton kernel
for PointNet++ (below).

Against the hand fix, gpt-6.1-sol was ahead by more than 3% on 7 repos, within 3% on
4 and behind on 4. The biggest hand-fix win was a change neither model tried: bf16
for both networks of pytorch-wgan, whose gradient penalty takes a second backward
pass. With it the hand fix measured 2.17x; the same hand fix without bf16 measured
1.38x, close to gpt-6.1-sol's 1.41x. On PyTorch-YOLOv3 both models found the winning
fixes (bf16, and keeping DataLoader workers alive across the 8-step epochs). The 23
host-device syncs per step in its loss code, which the profiler traced to their
lines, cost nothing measurable: removing them on their own measured 1.01x.

PointNet++ was the hardest. Its farthest point sampling is a Python loop of 640
steps per forward pass, and a masked update inside it (`distance[mask] = dist[mask]`)
made the CPU wait for the GPU 1,280 times per step; the GPU was busy 11% of the
time. Both models found that line in their first round, since the profiler names
it. Only gpt-6.1-sol got past the thousands of tiny kernel launches the loop still
made: it wrote a Triton kernel that runs the whole sampling loop at once, then added
bf16 for the convolutions. The kernel picks exactly the points the original loop
picks (identical in 40 of 40 test runs on a GPU, including inputs with many ties).
The hand fix replayed the loop from a CUDA graph instead and reached 2.61x.

DETR resisted everyone. At batch size 2, Python overhead per op sets the pace;
neither model kept a change, and the best of three hand attempts measured 1.06x.

- **Lost reports:** on pytorch-ddpm and rectified-flow-pytorch, the monthly GPU
  budget guard stopped the gpt-6.1-sol run's final measurement before it wrote a
  report. Their numbers come from measuring the kept changes against the original
  again, in pairs (`results/final-pairs/`).
- **Late hand fixes:** for efficient-kan, PyTorch-GAN and pytorch-wgan the hand fix
  was at first only the one-line DataLoader fix from the go/no-go check. The full
  hand fixes in the table came after the agent runs, with their reports at hand; the
  pytorch-wgan one borrows gpt-6.1-sol's idea of skipping the generator's gradients
  in critic steps.
- **Cost:** a gpt-6.1-sol run cost $0.05 to $0.09 of API credit, priced from the
  token usage the API reported, cache discount included
  (`results/checks/llm_spend.txt`). The first six gpt-6.1-sol runs saved no
  cached-token counts, so the four of them with reports price every prompt token
  at the full rate ($0.44 to $0.76 each).

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
python -m modal setup                    # once
python -m hotloop.kb.fetch               # download the docs
python -m hotloop.kb.search build        # build the search index

hotloop download pytorch-ddpm            # dataset into a Modal volume, once
hotloop measure pytorch-ddpm             # baseline on an L4, two seeds
hotloop profile pytorch-ddpm             # profiler summary
hotloop run pytorch-ddpm --setup agent --model gpt-6-luna
```

Models are listed in `hotloop/agent/models.py`; keys are read from the environment
or `~/hotloop-work/.env`.

Each target repo is a small config in `targets/<name>/target.toml`: the repo, a
pinned commit, the training command, extra pip packages and the dataset. `--setup`
picks what runs: `rules` (a few AST rules), `llm` (the model reads the code and
patches it once without measuring), `llm-kb` (the same with the docs), or `agent`.
`python -m hotloop.scoreboard` compares the setups over everything in `results/`.

`python -m hotloop.survey` rebuilds the static scan of popular PyTorch repos that
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
- The agent tries one change per round and keeps only measured wins, so it is
  cautious. With gpt-6-luna it rarely attempted the bigger changes that won most in
  the hand fixes, such as switching on flash attention or compiling a library's
  code; gpt-6.1-sol did, but neither model tried bf16 on a GAN with a gradient
  penalty.

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
