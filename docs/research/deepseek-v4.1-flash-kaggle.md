# DeepSeek V4.1 Flash and Qwen3.8-Flash-Next on Kaggle

**Research date:** 2026-10-30
**Status:** feasibility assessment; no new Kaggle runtime was started.

## Recommendation

Pursue an **experimental Qwen3.8-Flash-Next GGUF on Kaggle T4x2** proof of concept first. Use the Q2_0 build and llama.cpp with partial GPU offload, a short context, and one request at a time. This is plausible from the published memory figures, but it is not yet verified on Kaggle; the Q2_0 transformer shard is larger than the two T4s' combined VRAM, so CPU offload and Kaggle's 29 GB host RAM must be validated.

Do **not** add either model as a Kaggle TPU recipe yet. Current vLLM recipes explicitly exclude TPU for Qwen3.8-Flash-Next, and the Kaggle Qwen 3.8-27B vLLM-TPU setup does not establish support for this different architecture. DeepSeek's native checkpoint is far beyond Kaggle accelerator memory. A DeepSeek GGUF could theoretically be explored as a **CPU-only job on a Kaggle TPU session's large host**, but it would use none of the TPU's compute and would entail a 231–266 GB model download, storage, and uncertain CPU speed. Treat that as a later experiment only if CPU-only Kaggle inference is valuable.

For reliable Kaggle TPU serving today, keep the existing Qwen3.8-27B recipe as the supported path. Qwen3.8-Flash-Next is a separate architecture and should get a separate, explicitly experimental GPU recipe if the proof of concept succeeds.

## Kaggle resources that constrain the decision

Kaggle currently documents T4x2 sessions as **two 16 GB GPUs plus 29 GB host RAM**, and TPU 1VM sessions as **96 CPU cores and 330 GB host RAM**. Notebook time is capped at 12 hours for CPU/GPU and 9 hours for TPU; `/kaggle/working` has 20 GB of autosaved output, with additional scratch space whose size is not specified in the public spec. P100 was retired from Kaggle on 2026-09-15, so a new GPU recipe should target T4x2 rather than P100. [Kaggle notebook resources](https://www.kaggle.com/docs/notebooks), [Kaggle P100 retirement / T4x2 notice](https://www.kaggle.com/product-announcements/735239)

A TPU v5e chip has **16 GB HBM**; an eight-chip v5e-8 therefore has 128 GB nominal HBM in total, with some HBM reserved by the runtime. This is the accelerator in the existing Kaggle recipe. [Google Cloud TPU v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e)

Kaggle currently allows up to **200 GB per dataset** and 200 GB total across private datasets. The DeepSeek GGUF's smallest complete build is larger than one dataset; it would need multiple public datasets or an external download, and public hosting would be a deliberate licensing/distribution choice. [Kaggle dataset technical specifications](https://www.kaggle.com/docs/datasets)

## DeepSeek-V4.1-Flash-1017

The official checkpoint is a multimodal `DeepseekV41ForCausalLM` model using dynamic FP8 quantization with FP4 experts, and has a 1M context configuration. The vLLM model recipe lists **614 GB minimum VRAM** for the 1017 checkpoint and validates it on large NVIDIA/AMD accelerators; that excludes Kaggle's T4x2 and v5e-8 by a wide margin. [Official 1017 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), [vLLM DeepSeek V4.1 recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)

ISTA-DASLab's newer GGUF variant is much smaller but still enormous as a complete download. It contains a 551.9B-parameter backbone plus a 196.6B-parameter Engram lookup file. The provided builds are:

| Quant | Backbone | Engram | Total | Kaggle implication |
| --- | ---: | ---: | ---: | --- |
| IQ1_M | 120.72 GB | 110.60 GB | 231.32 GB | Smallest; still too large for T4x2 resident memory |
| IQ2_XXS | 138.24 GB | 110.60 GB | 248.84 GB | Recommended quality; too large for T4x2 resident memory |
| IQ2_XS | 155.23 GB | 110.60 GB | 265.83 GB | Largest of these builds |

The model card says the Engram file is sparse and can stay memory-mapped on disk. It describes a Strata execution mode that keeps the backbone in host memory while using a 12 GB GPU, and documents llama.cpp support from b11870 or newer. That is useful evidence for a CPU/RAM-tier design, but it is not evidence that Kaggle's 29 GB host RAM plus T4x2 can hold this backbone. Kaggle's TPU host has 330 GB RAM, so a CPU-only llama.cpp run there is the only plausible Kaggle configuration on the published numbers. Actual setup time, CPU throughput, scratch-disk capacity, and tool-call behavior remain unknown. [ISTA-DASLab GGUF model card and memory table](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)

**Verdict:** do not spend a Kaggle TPU allocation on DeepSeek now. If we later test it, use the IQ1_M CPU-only route on the TPU host, attach the two files as read-only public datasets or verify scratch capacity before downloading, start at 2K–4K context, and record cold-start time, RSS, first-token latency, decode tokens/s, and a tool-call smoke test. Stop if it cannot load comfortably below the 330 GB host-memory ceiling or cannot complete a small prompt within the session window. This is an inference from published capacity figures, not a tested result.

## Qwen3.8-Flash-Next

The Qwen model card describes a different model from the repo's existing Qwen3.8-27B: 125B main parameters, 51B learned per-layer n-gram embedding (PLE), about 6B activated per token, and native 262K context. Its official vLLM recipe says to use vLLM stable 0.30+ or nightly, and explicitly lists TPU/XPU as unsupported in the initial implementation. Its FP8 checkpoint is 172.78 GiB, and the recipe's minimum VRAM is 250 GB. That rules out the native FP8/vLLM path on Kaggle GPU or TPU. [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [vLLM Qwen3.8-Flash-Next recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml)

The current ISTA-DASLab GGUF is a promising alternative. It has a 512-expert, 48-layer model and a separate 28.8 GB quantized PLE shard that can be memory-mapped from disk. The fastest Q2_0 build has a **37.6 GB transformer shard**, 28.8 GB PLE shard, and 66.4 GB total download. IQ3_XXS is 47.0 + 28.8 GB; IQ3_S is 54.8 + 28.8 GB. The card says these are standard GGUFs supported by llama.cpp/Ollama/LM Studio, with `-lm mmap --lazy-mode on` to keep PLE on disk. [ISTA-DASLab Qwen GGUF model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)

On T4x2, the 37.6 GB Q2_0 shard exceeds the two GPUs' combined 32 GB VRAM, but it is smaller than the combined nominal GPU plus host memory (61 GB). Partial CPU/GPU layer placement may therefore fit at low context and one sequence. This is a **capacity-based inference only**: memory fragmentation, llama.cpp CUDA placement, PLE page faults, and Kaggle's small four-core host CPU may still cause OOM or very low throughput. The GGUF card's sample `-ngl 99` command requests all layers on GPU and should not be copied unchanged to T4x2.

The local `Burst-Compute-Testing/docs/QWEN-NEXT-FLASH-NGRAM.md` note is dated 2026-09-13 and says llama.cpp did not yet ship PLE support. The current ISTA GGUF card now documents the n-gram shard and llama.cpp mmap path, so that statement is stale and should be revised before using the note as current setup guidance. The Burst repo's `kernels/serve_qwen38.py` and this repo's `qwen38-27b/kernel/serve_qwen38.py` serve Qwen3.8-27B, not Flash-Next; they cannot be treated as proof of Flash-Next runtime compatibility.

**Verdict:** pursue an optional T4x2 llama.cpp POC for Qwen3.8-Flash-Next Q2_0, not a TPU/vLLM recipe. Use a Kaggle input dataset for the ~66 GB GGUF if available (it fits the current per-dataset cap), keep the PLE shard on the mounted input or scratch storage, and begin at 4K context with batch/concurrency 1. Validate the real available VRAM, RAM, filesystem capacity, launch flags, a short text prompt, one tool-call-shaped response, and one streaming request before publishing a recipe. If partial offload is unstable or too slow, stop; do not shift to BF16/FP8 or raise context on Kaggle.

## Minimal proof-of-concept sequence

1. Wait until the existing Kaggle TPU session releases its slot; do not stop or replace a user's queued/running job for this test.
2. Start a private **T4x2** kernel, not the existing vLLM TPU kernel. Check `nvidia-smi`, available RAM, `/kaggle/working`, and scratch capacity before downloading or attaching model data.
3. Use the Qwen3.8-Flash-Next GSQ-RCO **Q2_0** GGUF and current llama.cpp, with PLE memory-mapped and partial GPU offload. Keep context 4K, one sequence, no speculative decoding.
4. Require successful startup, `/v1/models`, a short completion, a stream, tool-call-shaped output, and a context follow-up. Record cold start, peak VRAM/RSS, TTFT, decode tokens/s, and whether PLE access causes stalls.
5. Publish an experimental Kaggle GPU recipe only if this passes repeatably. Keep the current Qwen3.8-27B TPU recipe independent.
6. Consider DeepSeek only as a separate CPU-only TPU-host experiment after the Qwen test, with the same resource receipts and an explicit pass/fail time budget.

## Primary references

- [Kaggle notebook runtime and accelerators](https://www.kaggle.com/docs/notebooks)
- [Kaggle T4x2 availability and P100 retirement](https://www.kaggle.com/product-announcements/735239)
- [Google Cloud TPU v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e)
- [Kaggle dataset limits](https://www.kaggle.com/docs/datasets)
- [DeepSeek V4.1 Flash 1017 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)
- [DeepSeek V4.1 vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
- [ISTA-DASLab DeepSeek V4.1 Flash 1017 GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
- [Qwen3.8-Flash-Next official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [vLLM Qwen3.8-Flash-Next recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml)
- [ISTA-DASLab Qwen3.8-Flash-Next GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- Internal cross-reference: `Burst-Compute-Testing/docs/QWEN-NEXT-FLASH-NGRAM.md` (research note dated 2026-09-13; refresh its llama.cpp support statement).
