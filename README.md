# Hotloop

An agent that finds and fixes slow PyTorch training code in real open-source repos.
It proves every fix by rerunning the training on a GPU: a change is kept only if
it is faster beyond run-to-run noise and the loss stays the same.

## Results

Every number below comes from a run saved in `results/`, and
`python -m hotloop.scoreboard` rebuilds these tables. Each one compares a repo's
original code with the changed code in pairs: both run in the same container on the
same GPU, once per seed (two seeds), for 50 measured steps after 10 warm-up steps.

### The agent on 12 real repos

The agent ran with gpt-6-luna, a small OpenAI model; all its runs together used
$0.29 of API credit. For each repo where it kept a change, the original and the
changed code were measured again on an A100:

| Repo | Stars | GPU | Before (ms/step) | After (ms/step) | Speedup | Loss | Fixes | Pull request |
|---|---|---|---|---|---|---|---|---|
| [monologg/JointBERT](https://github.com/monologg/JointBERT) | 749 | A100-SXM4-40GB | 67.4 | 17.3 | 3.90x | within noise | bf16 autocast, fused AdamW, torch.compile with CUDA graphs | not sent: no outside pull request merged in 12 months |
| [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) | 4,660 | A100-SXM4-40GB | 13.7 | 4.0 | 3.42x | within noise | DataLoader workers with pinned memory, torch.compile | not sent: no outside pull request merged in 12 months |
| [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) | 6,057 | A100-SXM4-80GB | 80.8 | 39.4 | 2.05x | within noise | TF32 matmuls | not sent: no outside pull request merged in 12 months |
| [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) | 17,457 | A100-SXM4-40GB | 24.5 | 12.0 | 2.04x | within noise | DataLoader workers with pinned memory | not sent: no outside pull request merged in 12 months |
| [Zeleni9/pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan) | 809 | A100-SXM4-40GB | 39.5 | 28.7 | 1.38x | within noise | DataLoader workers with pinned memory | not sent: no outside pull request merged in 12 months |
| [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) | 25,553 | A100-SXM4-80GB | 19.8 | 15.8 | 1.22x | within noise | DataLoader workers with pinned memory, fused Adam | drafted, not sent |
| [w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) | 646 | A100-SXM4-40GB | 144.7 | 118.3 | 1.22x | within noise | bf16 autocast for the forward pass and loss | not sent: no outside pull request merged in 12 months |
| [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | 489 | A100 80GB PCIe | 245.9 | 202.3 | 1.22x | within noise | DataLoader workers with pinned memory | issue drafted, not sent |
| [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR) | 825 | A100-SXM4-40GB | 178.1 | 175.8 | 1.01x (1.01x is within noise) | within noise | bf16 autocast, channels_last | not sent: no outside pull request merged in 12 months |
| [giddyyupp/ganilla](https://github.com/giddyyupp/ganilla) | 482 | L4 | 130.5 |  | nothing kept |  |  | not sent: no outside pull request merged in 12 months |
| [lucidrains/transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) | 1,408 | L4 | 85.3 |  | nothing kept |  |  | not sent |
| [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers) | 5,960 | L4 | 717.8 |  | nothing kept |  |  | not sent |

- **GPU** is the card Modal assigned to an A100-40GB request; every request was
  billed as an A100-40GB. Both sides of a pair always ran on the same card.
- **SimCLR:** the same changes were 2.06x faster on an L4. On the A100 the GPU work
  shrank until image augmentation on 8 CPU cores set the pace (data wait rose from
  4% to 36% of each step).
- **Nothing kept** means every change the agent tried crashed, was slower, or was
  within noise, so the original code stands.
- **Pull requests:** none sent yet. 8 of the 12 repos haven't merged a pull request
  from outside their team in the last 12 months.

### Against a hand fix

The same repos with the usual fixes applied by hand and measured the same way, on an
L4, as a reference for how much the agent leaves on the table:

| Repo | GPU | Original (ms/step) | Agent | Agent's changes | Hand fix | Hand fix's changes |
|---|---|---|---|---|---|---|
| x-transformers | L4 | 707.7 | nothing kept |  | 4.13x | batches built on the CPU and copied once, flash attention, bf16 |
| rectified-flow-pytorch | L4 | 439.4 | 1.17x | DataLoader workers with pinned memory | 3.12x | bf16, torch.compile through accelerate, loader workers |
| simclr | L4 | 416.6 | 2.06x | bf16 autocast, channels_last | 2.44x | bf16 autocast, channels_last, cudnn autotuning, pinned copies, no per-step epoch-loss sync |
| efficient-kan | L4 | 17.3 | 3.40x | DataLoader workers with pinned memory, torch.compile | 2.36x | num_workers=4, pin_memory=True on the training DataLoader (by hand, as a reference for the agent) |
| pytorch-ddpm | L4 | 549.2 | 1.29x | bf16 autocast for the forward pass and loss | 2.28x | bf16 autocast, compiled trainer, fused EMA update, one loss sync, pinned copies |
| jointbert | L4 | 137.9 | 2.65x | bf16 autocast, fused AdamW, torch.compile with CUDA graphs | 2.20x | bf16 autocast, fused AdamW, no per-step loss sync |
| pytorch-gan-dcgan | L4 | 28.9 | 2.03x | DataLoader workers with pinned memory | 1.90x | pass the existing --n_cpu option (default 8) to the DataLoader as num_workers |
| vit-pytorch | L4 | 26.7 | 1.37x | DataLoader workers with pinned memory, fused Adam | 1.74x | loader workers, torch.compile, one sync for the printed losses |
| transfusion-pytorch | L4 | 92.5 | nothing kept |  | 1.55x | loader workers, compiled transformer forward |
| nerf-pytorch | L4 | 191.7 | 1.18x | TF32 matmuls | 1.30x | TF32, no per-step host copies or syncs, ray sampling on the GPU |
| pytorch-wgan | L4 | 77.7 | 1.18x | DataLoader workers with pinned memory | 1.15x | add a --workers option (default 4) and pass it to the DataLoaders |
| ganilla | L4 | 123.2 | nothing kept |  | 1.13x | compiled networks, cudnn autotuning, pinned non-blocking input copies |

The agent matches or beats the hand fix on the standard fixes: mixed precision,
DataLoader workers, fused optimizers, and torch.compile where it compiles cleanly. It
falls short where the win needs a bigger or riskier change, such as flash attention
behind a library flag (x-transformers) or compiling library code (rectified-flow,
pytorch-ddpm, transfusion).

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

Yes, for the bigger changes. On x-transformers, where the agent with gpt-6-luna kept
nothing, the same agent with gpt-6.1-sol (a stronger model at 20 times the price)
made training 5.52x faster on an L4, with the loss within noise, for $0.44 of API
credit. It turned on the library's flash attention, bf16 through Accelerate,
torch.compile and fused Adam, and beat the hand fix (4.13x). Like the hand
experiment, it also tried copying whole batches to the GPU instead of single
samples, measured no gain, and reverted it.

It also kept torch.compile on rectified-flow-pytorch and pytorch-ddpm, which
gpt-6-luna never got to work. The month's GPU credit ran out before the final
paired measurement of those two runs, so they aren't counted here yet.

| Target | gpt-6-luna | gpt-6.1-sol |
|---|---|---|
| x-transformers | nothing kept | 5.52x |

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
  cautious. It rarely attempts the bigger changes that won most in the hand fixes,
  such as switching on flash attention or compiling a library's code.

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
