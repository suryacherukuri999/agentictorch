# Hotloop

An agent that finds and fixes slow PyTorch training code in real open-source repos.
It proves every fix by rerunning the training on a GPU: a change is kept only if
it is faster beyond run-to-run noise and the loss stays the same.

## Results

Runs on the real repos are in progress. This section will hold a table of measured
results from `results/`.

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
hotloop run pytorch-ddpm --setup agent --model ornith-9b
```

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
