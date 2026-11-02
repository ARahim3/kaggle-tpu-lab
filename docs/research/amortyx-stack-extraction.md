# Amortyx-Stack / Burst-Compute-Testing extraction assessment

Reviewed 2026-10-31. Sources: `/nas/Temp/repos/Amortyx-Stack/docs/burst-notebooks/`
(recipe pages pinned to Burst-Compute-Testing commit `b668e2d725018a9f062b653725986fead434f847`,
written 2026-10-30, not re-benched since) and `/nas/Temp/repos/Burst-Compute-Testing/kernels/`.
Builds on `docs/research/deepseek-v4.1-flash-kaggle.md`, which settled the model-feasibility
questions; this page settles what the *platform* evidence contributes to this lab.

## What the Burst campaigns measured that this lab had not

| Lane | Verdict (evidence) | Extraction value |
| --- | --- | --- |
| GLM-5.3-Flash on Kaggle TPU | **Validated end-to-end** 2026-09-15/16: TTUT 16–18 min, c1 52.9–82.1 tok/s, 0.4 s TTFT, 269k-doc resume in 0.43 s, knee at 3 streams ≈ 80 tok/s agg | Confirms this repo's glm53-flash recipe shape (streams=3, 262k ctx). Nothing to port — this repo IS the reference engine; the Burst copy was harvested from it |
| Qwen3.8-27B on Kaggle TPU | **Validated** (B0 byte-identical reproduction of this repo's `serve_qwen38.py`) | Already merged; the Burst fork tracks us, not the reverse |
| Qwen3.8-27B GPTQ-Int4 on Modal | **Validated on A10G and A100-40GB**; L40S configured but unbenched; T4 measured OOM (86 MiB short) | Extracted: `modal/` appliance (standalone, no Burst control-plane dependency) + `modal/notebook/qwen38-modal-serve.ipynb` |
| Qwen3.8-27B AWQ-INT4 on Kaggle GPU T4×2 | **Validated G1/G2 + session-7 re-run**: READY 5:29 warm cache, 61 tok/s @ 4 streams; **interactive gate FAILED** at c8 p99 16.8 s | Extracted as a third (batch/regression) lane — see below |

## Verdicts

1. **Modal lane — adopt (done).** The measured evidence transfers to a standalone
   appliance: pinned `vllm/vllm-openai:v0.28.0` registry image (flashinfer JIT for the
   hybrid GDN attention needs nvcc — two boots died on `debian_slim`), weights in a
   Volume, JIT caches in a second Volume (first boot pays ~2 min of compile, the Volume
   commits on exit), `--api-key` as the auth layer, `scaledown_window` 600 s. The lab's
   copy drops the Burst gateway/reporting seam and `BURST_SERVE_PROFILE` machinery —
   this repo is self-contained, not a Burst node.
2. **Kaggle GPU T4×2 lane — adopt as a clearly-labeled batch leaf.** It is the only
   measured lane that touches GPU quota (30 h/week) instead of TPU quota (20 h/week),
   and the warm-cache READY of 5:29 beats the TPU's ~16–22 min. But its interactive
   gate FAILED (c8 p99 16.8 s), so the recipe must advertise batch/regression use only
   and route interactive traffic to the TPU or Modal lanes. Key mechanics worth
   carrying: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the ~150 MB
   create_weights fragmentation OOM), TP=2, 16k ctx × 4 seqs (32k×8 measured OOM on
   14.56 GiB/card), and the version-guarded compile-cache dataset (engine init 665 →
   182 s).
3. **GLM-5.3-Flash — no extraction needed.** The Burst validation is *of* this repo's
   engine. The one open Burst cell (discriminating compile-cache run without the serve
   dataset) maps to this repo's cold-warm-up path and needs a Kaggle session to close,
   not code.
4. **Qwen3.8-Flash-Next — unchanged verdict, no new evidence.** The recipe pages do not
   touch Flash-Next; `docs/research/deepseek-v4.1-flash-kaggle.md` remains authoritative:
   TPU/vLLM explicitly unsupported upstream, optional T4×2 llama.cpp Q2_0 POC only.

## What was deliberately not extracted

- The Burst control-plane integration (gateway adoption, `burst bench`, BURST_STAGE
  telemetry). This repo's kernels already carry the `burst_gateway`/`burst_instance`
  config seam, so a user who *does* run the Burst controller can adopt these sessions;
  the lab itself does not depend on it.
- Modal GPU memory snapshots (alpha; 21–26 s restores on A100) — measured only through
  a direct deploy, not the adapter shape this lab ships, and they interact with
  `torch.compile` incompatibly without `enforce_eager`.
- The G3–G5 GPU cold-start plan (venv dataset, AWQ weights dataset, graph-capture
  binding) — designed with decision rules but never measured (Kaggle sessions blocked
  by server-side token loss at Burst `b668e2d`); citing its targets as results would be
  fabrication.

## Provenance and staleness rule

Every number quoted here comes from the Amortyx-Stack recipe pages, which quote the
Burst campaign evidence at the pinned commit; nothing was re-benched on 2026-10-30/31.
Kaggle queue windows, image versions, and Modal pricing drift. Before relying on a
lane, budget one throwaway session to confirm the READY timings, and record drift back
into the relevant recipe page (Amortyx-Stack) or this repo's README.
