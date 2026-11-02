# Qwen3.8-27B on a free Kaggle GPU (T4 x2)

**Serve Qwen3.8-27B (AWQ-INT4) on Kaggle's free GPU quota, with a public
OpenAI-compatible endpoint — a batch/regression lane that leaves your TPU quota alone.**

The 54 GB bf16 checkpoint cannot fit the 32 GB of a T4 x2 session, so this recipe serves
the community AWQ-INT4 quant (~16 GB, `cyankiwi/Qwen3.8-27B-AWQ-INT4`) with tensor
parallelism 2 and stock vLLM 0.29.0 (AWQ dequant runs on Turing/SM75; the
flashinfer/marlin paths need SM80+, and vllm-tpu's MTP patching does not apply here).

| What | Measured (T4 x2, AWQ-INT4, TP=2) |
|---|---|
| Time to live endpoint | **5:29** with the compile-cache dataset (13:26 cold) |
| Decode, 4 streams | **61 tok/s** aggregate |
| Decode, single stream | 19.4–23.4 tok/s |
| E2E TTFT | 0.41 s; 36.9 tok/s @ 2 streams |
| Interactive gate | **FAILED** at c8 (p99 16.8 s) — use the TPU or Modal lanes for chat/agents |
| Quota | GPU: 30 h/week (TPU is 20 h/week); sessions cap at 12 h |

**Use this lane for** regression suites, batch evals, soak tests — anything that values
throughput over tail latency and wants to preserve TPU hours.

## Run it

Notebook: open `notebook/qwen38-gpu-serve.ipynb` on Kaggle (accelerator **GPU T4 x2**,
Internet ON), run all cells.

CLI:

```bash
python launch.py serve --model qwen38-27b-gpu                          # defaults (16k ctx, 4 seqs)
python launch.py serve --model qwen38-27b-gpu --gpu-cache-dataset <owner>/<slug>   # warm start
python launch.py serve --model qwen38-27b-gpu --weights-dataset <owner>/<slug>     # skip the ~16 GB download
```

## Compile-cache flywheel

On exit the kernel tars the torch.compile cache and writes a `manifest.json` into the
session output. Publish both as a (private) dataset and attach it on the next run —
engine init drops from 665 s to 182 s measured. The manifest records the exact
vllm / torch / python versions; a mismatch still runs but warns loudly that the
cache may miss (the cache is content-addressed: it can miss, not corrupt).

## Good to know / limits

- **Interactive tail latency is the known weakness**: the concurrency-8 p99 was 16.8 s.
  Route humans and agents to the TPU notebook or the Modal appliance.
- **16k context × 4 sequences is the measured fit.** 32k × 8 OOM'd both T4s
  (14.56 GiB usable per card). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (set by the kernel) removes a ~150 MB allocator-fragmentation OOM at `create_weights`.
- **No speculative decoding, no vision**: stock vLLM on Turing, and the kernel launches
  vLLM text-only (`--limit-mm-per-prompt` zeroes images/video).
- **AWQ is a third-party quant**: run your own accuracy spot-checks before trusting
  outputs for anything careful.
- **Sessions cap at 12 h**, and the kernel also exits after `idle_min` (10) idle minutes.

## Provenance

The kernel and the measured numbers come from the Burst-Compute-Testing GPU campaign
(G1/G2, 2026-09-10; session-7 re-run 2026-10-14 at pinned commit
`b668e2d725018a9f062b653725986fead434f847`). See
`docs/research/amortyx-stack-extraction.md` for the full extraction assessment.
