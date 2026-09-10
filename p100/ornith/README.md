# Ornith-1.5-35B-A3B (CRACK) on Kaggle P100 (llama.cpp)

Serve `dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF` (CRACK-abliterated MoE) through an
OpenAI-compatible API on Kaggle's free **GPU P100** via llama.cpp. Text-only (no mmproj
— it would eat the VRAM headroom this needs for 256k context).

## Files

- `ornith-1.5-35b-a3b-p100.ipynb` — the notebook. Import at kaggle.com
  (Code → New Notebook → File → Import Notebook), set **Accelerator = GPU P100** and
  **Internet = ON**, then run all.
- `serve_ornith.py` + `kernel-metadata.json` — same pipeline as a pushable script kernel:
  ```bash
  kaggle kernels push -p ornith/    # needs ~/.kaggle/kaggle.json
  kaggle kernels status mikiasendale/ornith-p100-serve
  ```
  Progress (and the endpoint once live) publishes to ntfy topic
  `ktl-ornith-3f9c2b7e51a04d68`.

## What to expect

| | Ornith P100 (this folder) | Qwen3.8 P100 (`../`) | TPU v5e-8 (repo root) |
|---|---|---|---|
| Model | GGUF Q2_K, 13.25 GB (35B MoE, 8+1 experts active ≈ 3B/token) | GGUF UD-Q2_K_XL, 10 GB (27B dense) | bf16 safetensors |
| Decode | ~40-90 tok/s (est.) | ~15-22 tok/s (measured) | ~130 tok/s (measured, Qwen3.8) |
| Context | 262144 native (q4_0 KV ≈ 1.3 GB — only 10 full-attn layers × 2 KV heads) | 128000 | 262144 |
| Time to READY | ~6-8 min | ~6-8 min | ~22 min (different arch: unverified) |

The ~2-4x decode gain over the 27B dense comes from reading only ~0.9 GB of active
weights per token instead of the whole 14 GB model. Prefill stays Pascal-slow
(~1-2k tok/s est.), so very long prompts still take a while to ingest.

## VRAM math (why the ladder matters)

13.25 GB model + 1.34 GB KV (q4_0 @ 262144) + compute buffers ≈ 15-16 GB — right at the
P100's limit. The launcher tries `-fa on + q4_0 KV` at ngl 99 → 90 → 80 → 60, then
`-fa off` (f16 KV, lower ngl) before giving up. Don't load the mmproj alongside 256k.

## Wiring into opencode

Same key as the other kernels; alias is `Ornith-1.5-35B-A3B-CRACK`:

```jsonc
"baseURL": "https://<xxx>.trycloudflare.com/v1",
"apiKey":  "1e0afcc97b0ba77076ef35a63664d578",
"model":   "Ornith-1.5-35B-A3B-CRACK"
```
