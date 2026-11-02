# Modal — serve Qwen3.8-27B without a Kaggle queue

Kaggle's TPU is free but queues (3–9 min mornings, 80–95 min evenings) and caps sessions at
9 hours. [Modal](https://modal.com) is the complement: cold starts in seconds-to-minutes,
billed per container-second ($30/month free credit at signup), and the GPU stays warm while
idle only if you pay for it.

This directory holds a standalone vLLM appliance for **Qwen3.8-27B GPTQ-Int4** (~19 GiB of
weights in a Modal Volume), derived from the recipe validated on A10G and A100-40GB in the
[Burst-Compute-Testing](https://github.com/aliasfoxkde/Burst-Compute-Testing) campaigns.

## Measured reference points (Burst BM0 campaign)

| GPU | Verdict |
| --- | --- |
| A100-40GB | Full profile — graphs on, c1 ~55 tok/s, p50 TTFT ~215 ms |
| A10G (24 GiB) | Validated with a shrunken profile (ctx 6144, eager) — ~19 GiB weights leave ~0.7 GiB KV |
| T4 (16 GiB) | **OOM** — the artifact dies 86 MiB short. Do not use |
| L40S (48 GiB) | Configured (full 262K context) but not benched; requires a payment method on file |

## Setup (once)

```bash
pip install modal
modal setup                     # or: modal token new

# Weights into a Volume — one-time, ~19 GiB download:
python modal/populate_qwen38_volume.py
```

## Deploy

```bash
# Default: L40S, full 262,144-token context
modal deploy modal/serve_qwen38_modal.py

# A10G (free credits, shrunken profile):
BURST_GPU=A10G BURST_MAX_MODEL_LEN=6144 BURST_GPU_MEM_UTIL=0.98 \
BURST_MAX_NUM_SEQS=4 BURST_MAX_NUM_BATCHED_TOKENS=1024 BURST_ENFORCE_EAGER=1 \
modal deploy modal/serve_qwen38_modal.py
```

The deploy prints a public URL. Set an API key via a Modal secret so the endpoint is not
open to the internet:

```bash
modal secret create qwen38-serve-key BURST_SERVE_KEY=sk-your-secret
```

(or pass `BURST_SERVE_KEY` through `modal deploy --env`).

## Cost discipline

- The container scales to zero after `BURST_SCALEDOWN_SECONDS` (default 600 s) idle — GPU
  billing stops with it.
- End a deployment entirely with `modal app stop burst-qwen38`.
- Idle GPU seconds burn credits just like idle TPU minutes burn Kaggle quota.

## Connecting an agent

OpenAI-compatible: base URL `https://<workspace>--burst-qwen38-serve.us-east.modal.direct/v1`
with the serve key as the bearer. In Claude Code via the OpenAI-compatible surface, or point
any client's `OPENAI_BASE_URL` at it.

## Files

- `populate_qwen38_volume.py` — one-time weights download into the Volume
- `serve_qwen38_modal.py` — the Modal app (vLLM + idle scaledown)
- `notebook/qwen38-modal-serve.ipynb` — the same flow as a runnable notebook with
  per-cell instructions
