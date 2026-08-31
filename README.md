# kaggle-tpu-lab

**Serve Qwen3.8-27B — a frontier-class 27B hybrid-attention model — on Kaggle's free
TPU v5e-8, with a public OpenAI-compatible endpoint you can plug into Claude Code,
Codex CLI, opencode, or anything else that speaks the OpenAI API.**

No paid GPU, no cloud account, no quantization. Full bf16 weights, up to the model's
native **262,144-token context**, and real speed:

| What | Measured (TPU v5e-8, bf16, TP=8) |
|---|---|
| Decode, single stream | **104 tok/s** with MTP speculative decoding (78 without) |
| Decode, 16 concurrent streams | **~900 tok/s aggregate** (~60 tok/s each) |
| Prefill | **10,300 tok/s** — a 105k-token prompt in ~10 s |
| Native 262k context | works — 225k-token prompt prefilled in ~28 s |
| Time to live endpoint | ~30 min (~50 min without the compile-cache dataset) |
| Output correctness with MTP | **verified lossless** — 12/12 greedy prompts exactly match non-speculative |

## Why this works (the one-paragraph version)

Qwen3.8-27B is a hybrid: 48 of its 64 layers are **gated-DeltaNet linear attention**, only
16 are classic full attention. That makes its KV cache tiny (~64 KB/token), which is why a
27B can serve 131k+ contexts on 8×16 GB TPU chips with room to spare. Until recently no TPU
stack could run the DeltaNet layers — [vllm-tpu](https://github.com/vllm-project/tpu-inference)
0.28.0 shipped native Pallas kernels for them (Aug 2026), and this repo is the recipe that
puts it all together on Kaggle's free tier: pinned versions, pre-mirrored weights, a
pre-built XLA compile cache, MTP speculative decoding, and a tunnel to the outside world.

## Quick start A — from your terminal

You need Python 3.9+ and a Kaggle account with **TPU access** (Settings → phone-verify
your account if you haven't; free tier includes ~20 TPU hours/week).

```bash
# 1. Kaggle CLI + API token (one-time)
pip install kaggle
# kaggle.com → Settings → API → Create New Token, then place the file:
#   Linux/macOS: ~/.kaggle/kaggle.json     Windows: %USERPROFILE%\.kaggle\kaggle.json

# 2. Get this repo and launch
git clone https://github.com/ARahim3/kaggle-tpu-lab
cd kaggle-tpu-lab
python launch.py serve
```

That's it. The launcher pushes a script kernel to your Kaggle account, and streams
progress to your terminal so you always know what's happening:

```
[14:05]  Pushing kernel you/qwen38-tpu-serve (TPU v5e-8)...
[14:06]  Kaggle: queued — waiting for a TPU v5e-8 slot...
[14:08]  Kaggle: provisioning the VM and attaching datasets (a few minutes)...
[14:14]  Installing vllm-tpu on the TPU VM (~10 min)...
[14:24]  XLA compile cache restored — startup will be ~18 min instead of ~35.
[14:24]  Weights found mounted (no download needed).
[14:24]  Starting vLLM — loading 55 GB of weights + XLA compile...
[14:32]  Compiling / loading... 8 min elapsed (cold ~35 min, cached ~18 min)
[14:42]  Server is HEALTHY after 18 min.
[14:42]  Quick benchmark: 108.3 tok/s single-stream decode

==================================================================
  YOUR ENDPOINT IS LIVE
  base URL : https://xxxx-yyyy.trycloudflare.com/v1
  API key  : sk-....
  model    : qwen3.8-27b   (context: 262144)
==================================================================
```

`Ctrl-C` detaches without stopping the server. Re-attach with
`python launch.py status -f`; kill the TPU session with `python launch.py stop`.

Useful flags (the default launch serves the **full native 262k context**, 4 concurrent
sequences):

```bash
python launch.py serve --max-model-len 131072 --max-num-seqs 16  # many parallel streams (~900 tok/s aggregate)
python launch.py serve --reasoning-effort medium                 # server-side default
python launch.py serve --keepalive-min 120                       # auto-stop after 2 h
```

## Quick start B — as a Kaggle notebook

Prefer clicking? Open [`notebook/qwen38-tpu-serve.ipynb`](notebook/qwen38-tpu-serve.ipynb)
in Kaggle (upload it, or copy the published version), set **Accelerator = TPU VM v5e-8**,
**Internet = ON**, attach the two datasets named in the first cell, and run top to bottom.
The last cell *is* the server — the endpoint URL and API key appear in its output.

## Using it with coding agents

The endpoint is standard OpenAI API, with tool calling enabled
(`--enable-auto-tool-choice`, hermes parser) and the `qwen3` reasoning parser, so
thinking content is separated properly.

**Codex CLI / opencode / aider / anything OpenAI-compatible:**

```bash
export OPENAI_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
export OPENAI_API_KEY="sk-<your-key>"
# model name: qwen3.8-27b
```

**Claude Code** — the bundled vLLM also exposes an Anthropic-compatible API, so:

```bash
export ANTHROPIC_BASE_URL="https://<your-tunnel>.trycloudflare.com"
export ANTHROPIC_API_KEY="sk-<your-key>"
export ANTHROPIC_MODEL="qwen3.8-27b"
claude
```

(If your vLLM build lacks the Anthropic route, put [LiteLLM](https://github.com/BerriAI/litellm)
in front — one-liner proxy from Anthropic to OpenAI format.)

### Reasoning effort

Qwen3.8 has three thinking levels: **xhigh** (default), **medium**, **low** — plus off.
Set a per-request level (works from any client that lets you add request fields):

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

or turn thinking off entirely with `{"enable_thinking": false}`. To change the
*server-side default* (for clients that can't pass extra fields), launch with
`--reasoning-effort medium`.

## What's actually in this repo

```
launch.py                        the CLI: serve / status / stop, with live progress
kernel/serve_qwen38.py           the Kaggle kernel: install → patch → cache → weights → vLLM → tunnel
notebook/qwen38-tpu-serve.ipynb  the same flow as a run-it-yourself notebook
patches/mtp-rollback-v0280.diff  GDN state-rollback fix (port of tpu-inference PR #3178)
tools/embed_patch.py             re-embeds the patch into the kernel script after edits
```

Plus two public Kaggle datasets the kernel attaches:

- **`rahim3/qwen3-8-27b-bf16`** — mirror of `Qwen/Qwen3.8-27B` (55.6 GB safetensors).
  Attaching it skips the HF download entirely.
- **`rahim3/qwen38-xla-cache-v5e8`** — pre-populated JAX/XLA compile cache for exactly
  this stack (vllm-tpu 0.28.0 on v5e-8). Cuts cold-start compile roughly in half. If
  versions ever drift it's simply ignored (cache miss → normal compile).

## Good to know / limits

- **One TPU session at a time** per Kaggle account, sessions cap at 9 h, free quota is
  ~20 TPU-hours/week. The server auto-stops after `--keepalive-min` so a forgotten
  session doesn't eat your quota.
- **The endpoint is public** (random cloudflared URL) but protected by the generated
  API key. Treat the URL+key pair like a secret; a new launch gets fresh ones.
- **Prefix caching is off for now**: vllm-tpu 0.28.0 deliberately disables automatic
  prefix caching for hybrid linear-attention models on TPU. Upstream merged the fix
  two days after the 0.28.0 release, so a near-future version bump should enable it.
  Until then, multi-turn agent sessions re-prefill each turn — at 10k tok/s that's
  ~5 s for a 50k-token conversation, noticeable but fine.
- **First request after idle** can be a touch slower (scheduler warm-up); throughput
  numbers above are steady-state.
- **MTP speculative decoding: on by default, and there's a story.** Qwen3.8 ships a
  native MTP draft head, but stock vllm-tpu 0.28.0 **corrupts outputs** with it on
  TPU — rejected draft tokens advance the gated-DeltaNet recurrent state and are never
  rolled back (0/12 greedy prompts matched in our verification, with visible garbage).
  The fix exists as a stalled upstream PR
  ([tpu-inference #3178](https://github.com/vllm-project/tpu-inference/pull/3178));
  this repo bundles a port of it onto 0.28.0
  ([`patches/mtp-rollback-v0280.diff`](patches/mtp-rollback-v0280.diff)), applied to
  the installed wheel before serving. With the patch: **12/12 greedy prompts match the
  non-speculative outputs exactly**, at +34 % decode speed (104 vs 78 tok/s; healthy
  acceptance profile of 87/66/52 % per draft position). If the patch ever fails to
  apply (e.g. a future vllm-tpu version), the script disables MTP automatically rather
  than serve corrupted outputs. `--mtp 0` turns it off; k=4 fails to start.
- Kaggle's preinstalled `torchaudio` is incompatible with vllm-tpu's torch — the script
  uninstalls it automatically. Don't be alarmed by the log line.

## Credits

- [vLLM](https://github.com/vllm-project/vllm) and
  [tpu-inference](https://github.com/vllm-project/tpu-inference) teams — the GDN Pallas
  kernels and the MTP drafter are theirs; this repo is packaging and measurement.
- [Qwen](https://huggingface.co/Qwen) for releasing Qwen3.8-27B (Apache-2.0) with a
  native MTP head.
- Kaggle for the free TPUs.

## License

MIT for everything in this repo. Model weights follow the upstream
[Qwen3.8-27B license](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
