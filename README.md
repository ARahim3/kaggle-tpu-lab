# kaggle-tpu-lab

**Serve Qwen3.8-27B — a frontier-class 27B hybrid-attention model — on Kaggle's free
TPU v5e-8, with a public OpenAI-compatible endpoint you can plug into Claude Code,
Codex CLI, opencode, or anything else that speaks the OpenAI API.**

No paid GPU, no cloud account, no quantization. Full bf16 weights, up to the model's
native **262,144-token context**, and real speed:

| What | Measured (TPU v5e-8, bf16, TP=8) |
|---|---|
| Decode, single stream | **~130 tok/s** with MTP speculative decoding (78 without, measured before the token-bucket change) |
| Decode, 8 concurrent streams | **~540 tok/s aggregate** (~107 tok/s each, MTP on) |
| Decode, 16 concurrent streams | **~900 tok/s aggregate** (with `--mtp 0` — see tuning note below) |
| Prefill | **10,300 tok/s** — a 105k-token prompt in ~10 s |
| Native 262k context | works — 225k-token prompt prefilled in ~28 s |
| Time to live endpoint | **~22 min** with the env dataset attached (~12 with `--text-only`, ~6 with `--fast-start`; ~40 without the dataset) |
| Output correctness with MTP | **verified lossless** — 12/12 greedy prompts exactly match non-speculative |

Tuning note: speculative decoding pays off up to ~8 concurrent streams and fades beyond
that (verification competes with batch compute). Serving many users? Launch with
`--max-model-len 131072 --max-num-seqs 16 --mtp 0` for max aggregate throughput.

## Why this works (the one-paragraph version)

Qwen3.8-27B is a hybrid: 48 of its 64 layers are **gated-DeltaNet linear attention**, only
16 are classic full attention. That makes its KV cache tiny (~64 KB/token), which is why a
27B can serve 131k+ contexts on 8×16 GB TPU chips with room to spare. Until recently no TPU
stack could run the DeltaNet layers — [vllm-tpu](https://github.com/vllm-project/tpu-inference)
0.28.0 shipped native Pallas kernels for them (Aug 2026), and this repo is the recipe that
puts it all together on Kaggle's free tier: a pre-built Python runtime with pinned
versions, pre-mirrored weights, a pre-built XLA compile cache, MTP speculative decoding,
and a tunnel to the outside world.

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
[14:14]  Building the Python runtime with uv (~30 s)...
[14:15]  Runtime ready.
[14:15]  XLA compile cache restored for this exact config — fast start.
[14:15]  Weights found mounted (no download needed).
[14:15]  Starting vLLM — loading 55 GB of weights, then TPU graph compile...
[14:15]  Endpoint URL reserved: https://xxxx-yyyy.trycloudflare.com/v1  (not live yet — wait for the banner)
[14:19]  Loading / compiling... 4 min elapsed (typically ~20 min with the env dataset, ~35 min without)
[14:35]  Server is HEALTHY after 20 min.

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
python launch.py serve --text-only                               # skip the vision tower: ~10 min faster, no image inputs
python launch.py serve --fast-start                              # live in ~6 min; common shapes warmed after, rare ones stall ~1 min once
```

## Quick start B — as a Kaggle notebook

Prefer clicking? Copy the published Kaggle notebook —
[**kaggle.com/code/rahim3/qwen38-tpu-serve**](https://www.kaggle.com/code/rahim3/qwen38-tpu-serve)
— or upload [`notebook/qwen38-tpu-serve.ipynb`](notebook/qwen38-tpu-serve.ipynb) yourself.
Set **Accelerator = TPU VM v5e-8**, **Internet = ON**, attach the two datasets named in
the first cell, and run top to bottom. The last cell *is* the server — the endpoint URL
and API key appear in its output.

## Using it with coding agents

The endpoint is standard OpenAI API, with tool calling enabled
(`--enable-auto-tool-choice`, `qwen3_coder` parser — matches Qwen3.8's XML tool format)
and the `qwen3` reasoning parser, so thinking content is separated properly. The
launcher/notebook print the endpoint URL and API key in a banner when the server is up.

**Codex CLI / opencode / aider / anything OpenAI-compatible:**

```bash
export OPENAI_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
export OPENAI_API_KEY="sk-<your-key>"
# model name: qwen3.8-27b
```

**Claude Code** — the bundled vLLM also exposes an Anthropic-compatible `/v1/messages`
(reasoning arrives as proper `thinking` blocks; we verified Claude Code end-to-end
against it). One gotcha: the server authenticates with a Bearer header only, so use
`ANTHROPIC_AUTH_TOKEN`, **not** `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_BASE_URL="https://<your-tunnel>.trycloudflare.com"
export ANTHROPIC_AUTH_TOKEN="sk-<your-key>"
export ANTHROPIC_MODEL="qwen3.8-27b"
export ANTHROPIC_SMALL_FAST_MODEL="qwen3.8-27b"
claude
```

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
kernel/serve_qwen38.py           the Kaggle kernel: runtime → cache → weights → vLLM → tunnel → READY
notebook/qwen38-tpu-serve.ipynb  the same flow as a run-it-yourself notebook
patches/mtp-rollback-v0280.diff  GDN state-rollback fix (port of tpu-inference PR #3178)
tools/embed_patch.py             re-embeds the patch into the kernel script after edits
```

Plus two public Kaggle datasets the kernel attaches:

- **`rahim3/qwen3-8-27b-bf16`** — mirror of `Qwen/Qwen3.8-27B` (55.6 GB safetensors).
  Attaching it skips the HF download entirely.
- **`rahim3/qwen38-tpu-env-v5e8`** — the JAX/XLA compile cache for the documented
  configs (262k/4 and 131k/16 with images on, plus 262k/4 text-only; all MTP k=3), a
  `cloudflared` binary, and a `manifest.json` recording the build date and versions.
  The kernel pins its `uv` dependency resolution to that build date so the cache keeps
  matching. If the Python or vllm-tpu version ever drifts the cache is ignored and the
  graphs compile cold — slower, never broken.

### Where the startup time goes (and went)

The first version of this recipe took ~50 min from "Run" to a live URL. Measured now: ~22
(~12 with `--text-only`, ~6 with `--fast-start`):

| Step | Before | Now | How |
|---|---|---|---|
| pip install | 11 min | **30 s** | `uv` into a fresh venv with CPU torch (the PyPI default is the CUDA build + 3 GB of NVIDIA libs), resolution pinned to the cache's build date |
| Weights → TPU | 3 min | 3 min | reading 55 GB; unchanged |
| Vision-tower graphs | 13–23 min | ~8 min (0 with `--text-only`) | the compile part is now cached; the rest is tpu-inference tracing the vision encoder at warm-up, which no cache can skip |
| Text graphs at all in `--fast-start` | — | 0 at startup | precompile skipped; the script warms the common shapes right after READY, rare shapes stall ~1 min once |
| Text graphs | 8 buckets, ~19 min cold | 6 buckets, ~4 min | cache built for the *shipped* config (the old cache never matched, so every run compiled cold), `MIN_TOKEN_BUCKET=64`, 4 parallel compile threads |
| Tunnel | after the self-test | in parallel with the server start | URL is printed early, banner marks readiness |

Maintainers rebuild the bundle with `python launch.py build-env` (serves each config
once on a TPU kernel, ~2 h) and create/version the dataset from the kernel's output
folder in the Kaggle UI (Output tab → New Dataset).

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
- **Images work, with a one-time cost per image size.** Qwen3.8 is a vision-language model
  and the endpoint accepts `image_url` content (verified: it reads text and shapes from
  screenshots, with MTP on — that needed one more hunk in our patch, since tpu-inference
  0.28.0 crashes the engine on image + speculative decoding). tpu-inference compiles the
  vision encoder per image grid size, so the **first** image at a new resolution takes
  ~1 min and may even time out at the tunnel (HTTP 524) — just retry; every later image
  of that size is instant. Coding agents send screenshots at a consistent size, so this
  is paid once. `--text-only` drops image support and ~8 min of startup.
- **MTP speculative decoding: on by default, and there's a story.** Qwen3.8 ships a
  native MTP draft head, but stock vllm-tpu 0.28.0 **corrupts outputs** with it on
  TPU — rejected draft tokens advance the gated-DeltaNet recurrent state and are never
  rolled back (0/12 greedy prompts matched in our verification, with visible garbage).
  The fix exists as a stalled upstream PR
  ([tpu-inference #3178](https://github.com/vllm-project/tpu-inference/pull/3178));
  this repo bundles a port of it onto 0.28.0
  ([`patches/mtp-rollback-v0280.diff`](patches/mtp-rollback-v0280.diff)), applied to
  the installed wheel before serving. With the patch: **12/12 greedy prompts match the
  non-speculative outputs exactly**, at +34 % decode speed in the A/B test (104 vs 78 tok/s at the time; the shipped config now measures ~130; healthy
  acceptance profile of 87/66/52 % per draft position). If the patch ever fails to
  apply (e.g. a future vllm-tpu version), the script disables MTP automatically rather
  than serve corrupted outputs. `--mtp 0` turns it off; k=4 fails to start.
- **Harmless log noise.** vLLM prints a few scary-looking lines on every TPU start:
  `Unable to poll the TPU GCE Metadata` (Kaggle isn't a GCE VM), `Failed to import
  from vllm._C` (that's the CUDA extension), `Triton ... 0 active driver(s)`, and
  `Transparent hugepages are not enabled`. None of them matter. The kernel log only
  shows real progress lines; the full vLLM output is saved to `vllm.log` next to the
  script (`"verbose": true` / `--verbose` prints it live).

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
