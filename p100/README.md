# Qwen3.8-27B abliterated on Kaggle P100 (llama.cpp fallback)

Serve `huihui-ai/Huihui-Qwen3.8-27B-abliterated` through an OpenAI-compatible API on
Kaggle's free **GPU P100** via llama.cpp — the fallback for when **TPU v5e-8** is
unavailable (see the TPU path in the repo root; it's ~6x faster when it works).

## Files

- `qwen3.8-27b-abliterated-p100.ipynb` — the notebook. Import at kaggle.com
  (Code → New Notebook → File → Import Notebook), set **Accelerator = GPU P100** and
  **Internet = ON**, then run all.
- `serve_p100.py` + `kernel-metadata.json` — same pipeline as a pushable script kernel.
  Run without the browser:
  ```bash
  kaggle kernels push -p p100/     # needs ~/.kaggle/kaggle.json
  # then watch progress / fetch the log:
  kaggle kernels status mikiasendale/p100-serve
  kaggle kernels output mikiasendale/p100-serve -p <dir>   # after it finishes/errors
  ```
  The script publishes each phase to an ntfy topic, including the endpoint once live.

## What to expect

| | P100 (this folder) | TPU v5e-8 (repo root) |
|---|---|---|
| Model format | GGUF UD-Q2_K_XL (10.0 GB, 2-bit) | bf16 safetensors (55.6 GB) |
| Decode speed | ~15-22 tok/s | ~130 tok/s (MTP) |
| Context | 128000 (q4_0 KV cache, near-full GPU offload) | 262144 native |
| Time to READY | ~6-8 min (cold llama.cpp build ~5, download ~2) | ~22 min |

## Session behavior

- The model and llama.cpp build live in `/kaggle/tmp` — **nothing persists**; every
  session re-downloads (~1-2 min at Kaggle speeds) and rebuilds (~5 min on 4 vCPUs).
- The endpoint URL **changes every session** (Cloudflare quick tunnel).
- Use **Run → Background execution** so closing the browser tab doesn't kill the server
  (hard limit: 9 h GPU session, ~30 h/week P100 quota).

## Kaggle quirks this handles

- **No `libcuda.so` anywhere on the image** — even the CUDA toolkit ships without
  stubs, which breaks llama.cpp's `CUDA::cuda_driver` CMake target. Build with
  `-DGGML_CUDA_NO_VMM=ON` (drops the direct driver link; only loses cuMemCreate VMM,
  which is irrelevant on Pascal).
- The old `huggingface-cli download` subcommand is gone from current `huggingface_hub`
  (folded into the new `hf` CLI). Downloads use `snapshot_download(...)` instead.
- `pkill` must use `-x` (exact process name): with `-f`, a server-restart ladder whose
  script text contains the word `llama-server` kills its own shell.
- If `P100` shows "unavailable" too: phone-verify the account (Settings → Phone) —
  GPU/TPU/internet silently don't attach to unverified accounts.

## Wiring into opencode

Same key and model alias as the Colab version, so the existing `colab-qwen` provider
just needs its URL swapped:

```bash
cd ~/Documents/Default\ Project/colab-qwen
./set-colab-url.sh https://<xxx>.trycloudflare.com
```

The endpoint + key are printed by the tunnel cell (or the ntfy `p100-serve` messages)
when the server is live.
